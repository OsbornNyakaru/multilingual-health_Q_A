"""experiment.py — the ONE file the autoresearch loop edits (Karpathy's train.py role).

The frozen molab runner (notebooks/molab_runner.py) downloads this repo at the spec's git SHA,
imports this module fresh, and calls:

    answers, meta = run(config, eval_df, pool_df, ctx)

    config   the spec's "config" dict (see DEFAULT_CONFIG for every key)
    eval_df  rows to answer: ID, input, subset (+ output on held-out; never read it here)
    pool_df  rows with known answers that retrieval / few-shot may use:
             work_train for held-out runs, Train + Val for test runs. Never contains eval rows.
    ctx      services from the runner:
             ctx.get_model(model_id, precision, adapter) -> (tokenizer, model)   cached across runs
             ctx.get_embedder(name) -> SentenceTransformer                       cached across runs
             ctx.cache: dict          kernel-lifetime cache (e.g. pool embeddings)
             ctx.resume: dict[ID, answer]  answers already produced by an interrupted attempt
             ctx.save(answers)        checkpoint (the runner throttles uploads)
             ctx.should_stop() -> bool  True when the run hits its time budget
             ctx.log(msg)

    answers  dict[ID, str] for every eval row
    meta     dict[ID, dict] optional per-row diagnostics (e.g. retrieval similarity, route)

Scoring lives in prepare.py (frozen, hash-checked by the runner). Never score or read
eval_df["output"] here.
"""

from __future__ import annotations

import hashlib

import pandas as pd

ID_COL, INPUT_COL, OUTPUT_COL, SUBSET_COL = "ID", "input", "output", "subset"

DEFAULT_CONFIG: dict = {
    # retrieval | zero_shot | few_shot | rag_few_shot | router | lora_rag
    "mode": "retrieval",
    # retrieval (used by retrieval, rag_few_shot, router)
    "embedder": "BAAI/bge-m3",   # any sentence-transformers id, or "tfidf-char" (CPU)
    "query_prefix": "",          # e.g. "query: " for intfloat/multilingual-e5-*
    "passage_prefix": "",        # e.g. "passage: " for intfloat/multilingual-e5-*
    "retrieve_on": "input",      # match eval questions against pool questions
    "embed_batch": 64,
    # embedder_train: fine-tune the embedder on the training pool (anchor = a question, positive = another
    # question with the same answer, + 1 retrieved hard negative; in-batch negatives, contrastive loss)
    "embedder_train": False,
    "embedder_train_lr": 1e-5,
    "embedder_train_epochs": 1.0,
    "embedder_train_batch": 32,
    "embedder_train_max_len": 128,
    "embedder_train_subsets": None,  # e.g. ["Lug_Uga"]; None = every subset in the pool
    "hybrid_with": None,         # e.g. "tfidf-char": blend a second retriever's similarity in
    "hybrid_alpha": 0.5,         # weight of hybrid_with in the blend
    "rerank_model": None,        # e.g. "BAAI/bge-reranker-v2-m3": cross-encoder over the top rerank_k
    "rerank_k": 20,
    "rerank_on": "question",     # question (paraphrase check) | answer (relevance check) | both (question || answer)
    "rerank_answer_chars": 400,  # answer prefix length used when rerank_on is "answer" or "both"
    "rerank_batch": 64,
    # rerank_train: fine-tune the cross-encoder (rerank_model = its starting point) on the training pool:
    # query = a training question; positive = another training question with the SAME answer; hard
    # negatives = retrieved neighbours with a different answer. Listwise softmax loss.
    "rerank_train": False,
    "rerank_train_negs": 7,
    "rerank_train_epochs": 1.0,
    "rerank_train_lr": 2e-5,
    "rerank_train_batch": 8,      # query groups per step (each = 1 positive + rerank_train_negs negatives)
    "rerank_train_max_len": 256,
    "rerank_train_graded": False,  # targets = answer overlap with the gold (soft), not 1-positive/rest-negative
    "rerank_train_tau": 0.1,       # temperature of the soft targets
    "rerank_ensemble": 1,          # train K selectors (different seeds: data order + sampled negatives), average scores
    "select": "top1",            # top1 | vote: sum sim**vote_power over neighbours sharing an answer
    "vote_k": 50,
    "vote_power": 4.0,
    # generation (zero_shot, few_shot, rag_few_shot, router)
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "precision": "auto",         # auto | bf16 | 4bit
    "adapter": None,             # local path, or "run:<run_id>" = the adapter a lora_rag run uploaded
    "num_beams": 1,
    "infer_batch": 32,
    "max_input_tokens": 1024,
    "no_repeat_ngram": 0,         # 3 pushed Qwen into Chinese and garbled Akan (EXP-020/021)
    "length_penalty": 1.0,
    # MBR: besides the greedy answer, sample gen_samples answers at each of gen_temps and keep the medoid
    # (the candidate with the highest summed ROUGE-1/L to all the others; 1st place's selection rule)
    "gen_samples": 0,            # per temperature; 0 = greedy only
    "gen_temps": [0.7, 1.0, 1.3],
    "gen_top_p": 0.95,
    "gen_sample_batch": 64,      # sequences per sampling call (prompts per call = this // gen_samples)
    # "vllm": generate with vLLM in a subprocess (own venv, base + LoRA served directly); 5-10x faster than HF generate
    "gen_engine": "hf",          # hf | vllm
    "vllm_version": "0.31.0",
    "vllm_mem": 0.85,            # gpu_memory_utilization (the kernel keeps the embedder on the GPU)
    "vllm_chunk": 256,           # prompts per vLLM call; answers are checkpointed after each
    "gen_seed": 0,
    "few_shot_k": 2,             # fixed examples per subset (few_shot) or neighbours (rag_few_shot)
    "few_shot_max_chars": 150,   # cap on fixed few-shot example answers
    "rag_max_chars": 400,        # cap on retrieved example answers in rag_few_shot
    # router: copy the nearest answer when similarity >= threshold, else generate
    "router_threshold": 0.85,
    "router_generate_mode": "rag_few_shot",  # zero_shot | few_shot | rag_few_shot
    # answer length: "pool_p95" caps max_new_tokens at the 95th percentile of pool answers measured
    # with THIS model's tokenizer (the old word-count bounds truncated ~94% of Amharic answers);
    # "fixed" uses LENGTH_BOUNDS as-is.
    "length_mode": "pool_p95",
    "min_len_pct": 0,            # >0: min_new_tokens = this percentile of pool answer lengths (stops too-short answers)
    "fallback_below_frac": 0.0,  # >0: a generation shorter than this fraction of the subset's median pool
                                 # answer (in words) is replaced by the retrieved answer
    "checkpoint_every": 10,      # batches
    "diag_k": 0,                 # >0: record each row's top-k candidate pool IDs (before + after rerank) for recall@k
    # lora_rag: fine-tune a LoRA adapter on RAG-enriched prompts (each training row sees its few_shot_k
    # nearest OTHER training Q&A pairs, target = its own answer), then generate like rag_few_shot.
    # Defaults = the 11th-place reference recipe.
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.5,
    "lora_lr": 2e-4,
    "lora_epochs": 3.0,
    "lora_batch": 4,
    "lora_grad_acc": 1,
    "lora_warmup": 0.03,
    "lora_max_len": 2048,
    "lora_data_frac": 1.0,       # fraction of the training pool (stratified by subset) to train on
    "lora_train_subsets": None,  # e.g. ["Aka_Gha", "Eng_Gha", "Amh_Eth"]; None = every subset in the pool
    "lora_bits": 16,             # 16 = bf16 LoRA; 4 = QLoRA (4-bit base), for 27B+ models
    "lora_optim": "adamw_torch",
    "lora_save_steps": 500,      # also uploads the checkpoint to HF so training survives a molab restart
}

SUBSET_TO_LANG = {
    "Aka_Gha": "aka", "Amh_Eth": "amh", "Eng_Eth": "eng", "Eng_Gha": "eng",
    "Eng_Ken": "eng", "Eng_Uga": "eng", "Lug_Uga": "lug", "Swa_Ken": "swa",
}
SYSTEM_INSTRUCTIONS = {
    "eng": "You are an expert health worker. Answer the health question accurately and concisely in English, in the same style as a clinical reference answer. Never refuse to answer.",
    "swa": "Wewe ni mtaalamu wa afya. Jibu swali la afya kwa usahihi na kwa ufupi kwa Kiswahili. Usiseme huwezi kujibu.",
    "lug": "Oli omukugu mu by'obulamu. Ddamu ekibuuzo ky'obulamu mu bujjuvu era mu Luganda. Togamba nti toyinza kuddamu.",
    "aka": "Woyɛ apɔmuden ho nimdefoɔ. Bua apɔmuden asɛmmisa no yiye wɔ Akan kasa mu. Nnka sɛ wontumi mmua so.",
    "amh": "እርስዎ የጤና ባለሙያ ነዎት። ይህን የጤና ጥያቄ በትክክል እና በአጭሩ በአማርኛ ይመልሱ። መመለስ አልችልም አይበሉ።",
}
PROMPT_TEMPLATES = {
    "eng": "Question: {question}\nAnswer:",
    "swa": "Swali: {question}\nJibu:",
    "lug": "Ekibuuzo: {question}\nEky'okuddamu:",
    "aka": "Asɛmmisa: {question}\nMmuaeɛ:",
    "amh": "ጥያቄ: {question}\nመልስ:",
}
ANSWER_MARKERS = {"eng": "Answer:", "swa": "Jibu:", "lug": "Eky'okuddamu:", "aka": "Mmuaeɛ:", "amh": "መልስ:"}
# Per-subset new-token bounds from autoresearch_nlp/tools/length_calibrate.py (June 2026).
LENGTH_BOUNDS = {
    "Aka_Gha": {"min_new_tokens": 35, "max_new_tokens": 266},
    "Amh_Eth": {"min_new_tokens": 11, "max_new_tokens": 58},
    "Eng_Eth": {"min_new_tokens": 15, "max_new_tokens": 65},
    "Eng_Gha": {"min_new_tokens": 42, "max_new_tokens": 184},
    "Eng_Ken": {"min_new_tokens": 30, "max_new_tokens": 226},
    "Eng_Uga": {"min_new_tokens": 26, "max_new_tokens": 273},
    "Lug_Uga": {"min_new_tokens": 22, "max_new_tokens": 226},
    "Swa_Ken": {"min_new_tokens": 29, "max_new_tokens": 246},
}
DEFAULT_BOUNDS = {"min_new_tokens": 8, "max_new_tokens": 160}
# NOTE: these were calibrated in words, not tokens; only used with length_mode="fixed".

GENERATIVE_MODES = {"zero_shot", "few_shot", "rag_few_shot", "lora_rag"}


def lang_of(subset: str) -> str:
    return SUBSET_TO_LANG.get(subset, "eng")


def postprocess(text: str, subset: str) -> str:
    text = str(text).strip()
    marker = ANSWER_MARKERS.get(lang_of(subset), "")
    if marker and marker in text:
        text = text.split(marker, 1)[-1].strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text


# ── retrieval ────────────────────────────────────────────────────────────────


def _pool_key(name: str, prefix: str, cfg: dict, pool_df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(pool_df[[ID_COL]], index=False).values.tobytes()).hexdigest()[:12]
    return f"pool_emb|{name}|{prefix}|{cfg['retrieve_on']}|{h}"


def _encode(name: str, cfg: dict, eval_df: pd.DataFrame, pool_df: pd.DataFrame, ctx):
    """(query vectors, pool vectors) for one retriever; rows L2-normalised so dot = cosine."""
    dense = name != "tfidf-char"
    qp, pp = (cfg["query_prefix"], cfg["passage_prefix"]) if dense else ("", "")
    key = _pool_key(name, pp, cfg, pool_df)
    pool_texts = [pp + str(t) for t in pool_df[cfg["retrieve_on"]]]
    queries = [qp + str(t) for t in eval_df[INPUT_COL]]
    if not dense:
        # CPU retriever: char 3-5-gram TF-IDF.
        from sklearn.feature_extraction.text import TfidfVectorizer

        if key not in ctx.cache:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
            ctx.cache[key] = (vec, vec.fit_transform(pool_texts))
        vec, pool_vecs = ctx.cache[key]
        return vec.transform(queries), pool_vecs
    emb = ctx.get_embedder(name)
    if key not in ctx.cache:
        ctx.log(f"embedding pool: {len(pool_df):,} rows with {name}")
        ctx.cache[key] = emb.encode(
            pool_texts, batch_size=int(cfg["embed_batch"]), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
    q_vecs = emb.encode(
        queries, batch_size=int(cfg["embed_batch"]), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return q_vecs, ctx.cache[key]


def neighbours(cfg: dict, eval_df: pd.DataFrame, pool_df: pd.DataFrame, ctx, k: int) -> dict[str, list[tuple[int, float]]]:
    """Top-k pool rows (positional index into pool_df, similarity) per eval ID, within the same subset.

    With hybrid_with set, similarity = (1 - hybrid_alpha) * embedder + hybrid_alpha * hybrid_with.
    """
    import numpy as np

    parts = [(_encode(cfg["embedder"], cfg, eval_df, pool_df, ctx), 1.0)]
    if cfg["hybrid_with"]:
        a = float(cfg["hybrid_alpha"])
        parts = [(parts[0][0], 1.0 - a), (_encode(cfg["hybrid_with"], cfg, eval_df, pool_df, ctx), a)]
    pool_subsets = pool_df[SUBSET_COL].to_numpy()
    out: dict[str, list[tuple[int, float]]] = {}
    for subset in eval_df[SUBSET_COL].unique():
        qi = (eval_df[SUBSET_COL] == subset).to_numpy().nonzero()[0]
        pi = (pool_subsets == subset).nonzero()[0]
        if len(pi) == 0:  # unseen subset: fall back to the whole pool
            pi = np.arange(len(pool_df))
        sims = 0.0
        for (q_vecs, pool_vecs), w in parts:
            block = q_vecs[qi] @ pool_vecs[pi].T
            sims = sims + w * (block.toarray() if hasattr(block, "toarray") else np.asarray(block))
        kk = min(k, len(pi))
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for row, q in enumerate(qi):
            order = top[row][np.argsort(-sims[row, top[row]])]
            out[str(eval_df[ID_COL].iloc[q])] = [(int(pi[j]), float(sims[row, j])) for j in order]
    return out


def candidate_text(cfg: dict, question: str, answer: str) -> str:
    """What the cross-encoder sees for one pool row (training and inference use the same form)."""
    n = int(cfg["rerank_answer_chars"])
    if cfg["rerank_on"] == "answer":
        return str(answer)[:n]
    if cfg["rerank_on"] == "both":
        return f"{question} || {str(answer)[:n]}"
    return str(question)


def rerank(cfg: dict, eval_df: pd.DataFrame, pool_df: pd.DataFrame, nn: dict, ctx) -> dict[str, list[tuple[int, float]]]:
    """Re-score each eval row's top rerank_k candidates with a cross-encoder; returns them sorted by that score.

    rerank_on="question" scores (eval question, neighbour's question): a paraphrase check.
    rerank_on="answer" scores (eval question, neighbour's answer): a relevance check.
    """
    models = list(cfg["rerank_model"]) if isinstance(cfg["rerank_model"], (list, tuple)) else [cfg["rerank_model"]]
    ces = []
    for m in models:
        key = ("cross_encoder", m)
        if key not in ctx.cache:
            import torch
            from sentence_transformers import CrossEncoder

            ctx.cache[key] = CrossEncoder(m, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
        ces.append(ctx.cache[key])
    questions = dict(zip(eval_df[ID_COL].astype(str), eval_df[INPUT_COL].astype(str)))
    k = int(cfg["rerank_k"])
    pairs, owners = [], []
    for i, hits in nn.items():
        for j, _ in hits[:k]:
            pairs.append((questions[i], candidate_text(cfg, pool_df[INPUT_COL].iloc[j], pool_df[OUTPUT_COL].iloc[j])))
            owners.append((i, j))
    ctx.log(f"reranking {len(pairs):,} pairs with {len(ces)} model(s) on {cfg['rerank_on']}")
    import numpy as np

    scores = np.mean([np.asarray(ce.predict(pairs, batch_size=int(cfg["rerank_batch"]), show_progress_bar=False), dtype=float)
                      for ce in ces], axis=0)
    out: dict[str, list[tuple[int, float]]] = {i: [] for i in nn}
    for (i, j), sc in zip(owners, scores):
        out[i].append((j, float(sc)))
    return {i: sorted(v, key=lambda t: -t[1]) for i, v in out.items()}


# ── prompting + generation ───────────────────────────────────────────────────


def fixed_examples(cfg: dict, pool_df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """k deterministic, refusal-free examples per subset (classic few-shot)."""
    k, cap = int(cfg["few_shot_k"]), int(cfg["few_shot_max_chars"])
    bank: dict[str, list[tuple[str, str]]] = {}
    for subset, grp in pool_df.groupby(SUBSET_COL):
        good = grp[
            grp[OUTPUT_COL].str.len().between(50, 400)
            & ~grp[OUTPUT_COL].str.lower().str.contains("cannot|sorry|apologize", regex=True)
        ]
        good = good if len(good) >= k else grp
        sample = good.sample(min(k, len(good)), random_state=42)
        bank[str(subset)] = [(str(r[INPUT_COL]).strip()[:300], str(r[OUTPUT_COL]).strip()[:cap]) for _, r in sample.iterrows()]
    return bank


def build_messages(question: str, subset: str, examples: list[tuple[str, str]]) -> list[dict]:
    lang = lang_of(subset)
    msgs = [{"role": "system", "content": SYSTEM_INSTRUCTIONS[lang]}]
    for ex_q, ex_a in examples:
        msgs.append({"role": "user", "content": PROMPT_TEMPLATES[lang].format(question=ex_q)})
        msgs.append({"role": "assistant", "content": ex_a})
    msgs.append({"role": "user", "content": PROMPT_TEMPLATES[lang].format(question=question.strip())})
    return msgs


def render_prompt(tok, msgs: list[dict]) -> str:
    if getattr(tok, "chat_template", None):
        try:
            # enable_thinking=False: Qwen3.x / Gemma 4 templates otherwise open a reasoning block; others ignore it
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            pass  # e.g. templates that reject a system role
    return "\n\n".join(m["content"] for m in msgs)


def length_bounds(cfg: dict, tok, pool_df: pd.DataFrame) -> dict[str, dict]:
    if cfg["length_mode"] == "fixed":
        return LENGTH_BOUNDS
    import numpy as np

    out = {}
    for subset, grp in pool_df.groupby(SUBSET_COL):
        sample = grp[OUTPUT_COL].sample(min(500, len(grp)), random_state=0).tolist()
        lens = [len(x) for x in tok(sample, add_special_tokens=False)["input_ids"]]
        lo = int(np.percentile(lens, float(cfg["min_len_pct"]))) if float(cfg["min_len_pct"]) > 0 else 1
        out[str(subset)] = {"min_new_tokens": max(1, lo), "max_new_tokens": int(np.percentile(lens, 95)) + 8}
    return out


def mbr_pick(cands: list[str]) -> int:
    """Index of the medoid: the candidate with the highest summed answer_overlap to all the others."""
    if len(cands) <= 2:
        return 0
    best, best_i = -1.0, 0
    for i, a in enumerate(cands):
        u = sum(answer_overlap(a, b) for j, b in enumerate(cands) if j != i)
        if u > best:
            best, best_i = u, i
    return best_i


def sample_candidates(cfg: dict, tok, model, prompts: list[str], bounds: dict) -> list[list[str]]:
    """gen_samples sampled answers per prompt at each of gen_temps (raw text, before postprocess)."""
    import torch

    n = int(cfg["gen_samples"])
    out: list[list[str]] = [[] for _ in prompts]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg["max_input_tokens"])).to(model.device)
    for t in cfg["gen_temps"]:
        with torch.inference_mode():
            gen = model.generate(
                **enc, do_sample=True, temperature=float(t), top_p=float(cfg["gen_top_p"]), top_k=0,
                num_return_sequences=n, num_beams=1,
                min_new_tokens=bounds["min_new_tokens"], max_new_tokens=bounds["max_new_tokens"],
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for k, txt in enumerate(texts):
            out[k // n].append(txt)
        del gen
        torch.cuda.empty_cache()  # each sampling call holds n x prompts KV caches; 31B models OOM'd at 32 (EXP-064)
    return out


def generate(cfg: dict, rows: pd.DataFrame, examples_for, ctx, answers: dict[str, str], pool_df: pd.DataFrame,
             meta: dict | None = None) -> None:
    """Fill answers[ID] for every row not already present. examples_for(ID, subset) -> examples.
    With gen_samples > 0, the answer is the MBR medoid of greedy + sampled candidates (all kept in meta["cands"])."""
    if cfg["gen_engine"] == "vllm":
        return generate_vllm(cfg, rows, examples_for, ctx, answers, pool_df, meta)
    import torch

    tok, model = ctx.get_model(cfg["model_id"], cfg["precision"], cfg["adapter"])
    bounds_by_subset = length_bounds(cfg, tok, pool_df)
    ctx.log("max_new_tokens per subset: " + ", ".join(f"{k}={v['max_new_tokens']}" for k, v in sorted(bounds_by_subset.items())))
    todo = rows[~rows[ID_COL].astype(str).isin(answers.keys())].reset_index(drop=True)
    if todo.empty:
        return
    prompts = {
        str(r[ID_COL]): render_prompt(tok, build_messages(str(r[INPUT_COL]), str(r[SUBSET_COL]), examples_for(str(r[ID_COL]), str(r[SUBSET_COL]))))
        for _, r in todo.iterrows()
    }
    batch_size, every = int(cfg["infer_batch"]), int(cfg["checkpoint_every"])
    batches: list[tuple[str, list[str]]] = []
    for subset, grp in todo.groupby(SUBSET_COL):
        ids = sorted(grp[ID_COL].astype(str), key=lambda i: len(prompts[i]))
        batches += [(str(subset), ids[s : s + batch_size]) for s in range(0, len(ids), batch_size)]
    ctx.log(f"generating {len(todo):,} rows in {len(batches)} batches with {cfg['model_id']}")
    for b, (subset, ids) in enumerate(batches, 1):
        if ctx.should_stop():
            ctx.log("time budget reached; stopping generation")
            return
        bounds = bounds_by_subset.get(subset, DEFAULT_BOUNDS)
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, truncation=True,
                  max_length=int(cfg["max_input_tokens"])).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc, num_beams=int(cfg["num_beams"]), do_sample=False,
                no_repeat_ngram_size=int(cfg["no_repeat_ngram"]), length_penalty=float(cfg["length_penalty"]),
                min_new_tokens=bounds["min_new_tokens"], max_new_tokens=bounds["max_new_tokens"],
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if int(cfg["gen_samples"]) > 0:
            import json

            per_call = max(1, int(cfg["gen_sample_batch"]) // int(cfg["gen_samples"]))
            sampled = []
            for s0 in range(0, len(ids), per_call):
                sampled += sample_candidates(cfg, tok, model, [prompts[i] for i in ids[s0 : s0 + per_call]], bounds)
            for i, g, smp in zip(ids, texts, sampled):
                cands = [postprocess(x, subset) for x in [g] + smp]
                answers[i] = cands[mbr_pick(cands)]
                if meta is not None:
                    meta.setdefault(i, {})["cands"] = json.dumps(cands, ensure_ascii=False)
        else:
            for i, t in zip(ids, texts):
                answers[i] = postprocess(t, subset)
        if b % every == 0:
            ctx.save(answers)
        ctx.log(f"batch {b}/{len(batches)} ({subset})", progress=(b, len(batches)))
    ctx.save(answers)


# ── vLLM generation (gen_engine="vllm") ──────────────────────────────────────

VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def clean_env() -> dict:
    """The kernel's environment minus whatever points Python at the kernel's packages: molab sets
    PYTHONPATH to its own (3.13) site-packages, which the vLLM venv would otherwise import (EXP-066)."""
    import os

    drop = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONUSERBASE"}
    return {k: v for k, v in os.environ.items() if k not in drop}


def ensure_vllm(cfg: dict, ctx) -> str:
    """Python of a venv holding vLLM (built once per kernel). Separate from the kernel so vLLM's
    pinned torch never replaces the one the runner already imported."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    env = Path.cwd() / "runner_work" / f"vllm-{cfg['vllm_version']}"
    py = env / "bin" / "python"
    if (env / ".ok").exists():
        return str(py)
    if env.exists():  # a failed earlier attempt: start over
        import shutil

        shutil.rmtree(env)
    t0 = time.time()
    ctx.log(f"installing vLLM {cfg['vllm_version']} into {env}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    uv = [sys.executable, "-m", "uv"]
    subprocess.run(uv + ["venv", "--seed", "--python", "3.12", str(env)], check=True)
    r = subprocess.run(uv + ["pip", "install", "--python", str(py), "--torch-backend=auto", f"vllm=={cfg['vllm_version']}"],
                       capture_output=True, text=True, env=clean_env())
    if r.returncode:
        raise RuntimeError("vLLM install failed:\n" + (r.stdout + r.stderr)[-3000:])
    v = subprocess.run([str(py), "-I", "-c", "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
                       capture_output=True, text=True, env=clean_env())
    ctx.log(f"vLLM ready in {time.time() - t0:.0f}s: {v.stdout.strip() or v.stderr[-1500:]}")
    if v.returncode:
        raise RuntimeError("vLLM import failed:\n" + v.stderr[-3000:])
    (env / ".ok").touch()
    return str(py)


def free_gpu_for_vllm(cfg: dict, ctx) -> float:
    """Release what the kernel holds on the GPU and return the gpu_memory_utilization vLLM can ask for.
    An interrupted run can leave its HF model alive through the saved traceback (EXP-068: 21 of 95 GB free)."""
    import gc
    import sys

    if hasattr(ctx, "free_model"):
        ctx.free_model()
    for name in ("last_traceback", "last_value", "last_exc", "last_type"):
        if hasattr(sys, name):
            setattr(sys, name, None)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return float(cfg["vllm_mem"])
    mem = min(float(cfg["vllm_mem"]), free / total - 0.03)
    ctx.log(f"GPU before vLLM: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free -> gpu_memory_utilization {mem:.2f}")
    if mem < 0.75:
        raise RuntimeError(f"only {free / 2**30:.1f} GiB of GPU memory free for vLLM; something in the kernel still "
                           "holds a model. Restart the molab kernel and press Run queue.")
    return mem


def merge_vllm_output(rec: dict, subset: str) -> tuple[str, list[str]]:
    """(answer, candidates) for one worker record: greedy alone, or the MBR medoid of greedy + samples."""
    cands = [postprocess(x, subset) for x in [rec["greedy"]] + rec["samples"]]
    return cands[mbr_pick(cands)], cands


def generate_vllm(cfg: dict, rows: pd.DataFrame, examples_for, ctx, answers: dict[str, str], pool_df: pd.DataFrame,
                  meta: dict | None = None) -> None:
    import json
    import subprocess
    import time
    from pathlib import Path

    from transformers import AutoTokenizer

    todo = rows[~rows[ID_COL].astype(str).isin(answers.keys())].reset_index(drop=True)
    if todo.empty:
        return
    mem = free_gpu_for_vllm(cfg, ctx)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    bounds_by_subset = length_bounds(cfg, tok, pool_df)
    ctx.log("max_new_tokens per subset: " + ", ".join(f"{k}={v['max_new_tokens']}" for k, v in sorted(bounds_by_subset.items())))
    subset_of = dict(zip(todo[ID_COL].astype(str), todo[SUBSET_COL].astype(str)))
    reqs = []
    for _, r in todo.iterrows():
        i, subset = str(r[ID_COL]), str(r[SUBSET_COL])
        prompt = render_prompt(tok, build_messages(str(r[INPUT_COL]), subset, examples_for(i, subset)))
        ids = tok(prompt, truncation=True, max_length=int(cfg["max_input_tokens"]))["input_ids"]  # same ids as the HF path
        b = bounds_by_subset.get(subset, DEFAULT_BOUNDS)
        reqs.append({"id": i, "ids": ids, "max_tokens": b["max_new_tokens"], "min_tokens": b["min_new_tokens"]})

    adapter, rank = resolve_adapter(cfg["adapter"], ctx), 0
    if adapter:
        r_ = int(json.load(open(Path(adapter) / "adapter_config.json"))["r"])
        rank = next(x for x in VLLM_LORA_RANKS if x >= r_)
    py = ensure_vllm(cfg, ctx)
    work = Path(_work_dir(ctx, cfg))
    work.mkdir(parents=True, exist_ok=True)
    out, log_path, job_path = work / "vllm_out.jsonl", work / "vllm_log.txt", work / "vllm_job.json"
    max_len = max(len(r["ids"]) for r in reqs) + max(r["max_tokens"] for r in reqs) + 16
    job = {"model": cfg["model_id"], "adapter": adapter, "lora_rank": rank, "max_model_len": max_len,
           "mem": mem, "temps": list(cfg["gen_temps"]), "n": int(cfg["gen_samples"]),
           "top_p": float(cfg["gen_top_p"]), "seed": int(cfg["gen_seed"]), "chunk": int(cfg["vllm_chunk"]),
           "requests": reqs, "out": str(out)}
    job_path.write_text(json.dumps(job), encoding="utf-8")
    ctx.log(f"vLLM: {len(reqs):,} rows, {1 + len(cfg['gen_temps']) * int(cfg['gen_samples']) if int(cfg['gen_samples']) else 1} "
            f"candidates each, max_model_len {max_len}, LoRA rank {rank or '-'}")

    worker = Path(__file__).with_name("vllm_worker.py")
    seen: set[str] = set()

    def collect() -> int:
        if not out.exists():
            return 0
        new = 0
        with open(out, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                i = rec["id"]
                if i in seen or i not in subset_of:
                    continue
                seen.add(i)
                ans, cands = merge_vllm_output(rec, subset_of[i])
                answers[i] = ans
                if meta is not None and int(cfg["gen_samples"]) > 0:
                    meta.setdefault(i, {})["cands"] = json.dumps(cands, ensure_ascii=False)
                new += 1
        return new

    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen([py, str(worker), str(job_path)], stdout=lf, stderr=subprocess.STDOUT, env=clean_env())
        while proc.poll() is None:
            time.sleep(30)
            if collect():
                ctx.save(answers)
                ctx.log(f"vLLM {len(seen):,}/{len(reqs):,} rows ({time.time() - t0:.0f}s)", progress=(len(seen), len(reqs)))
            if ctx.should_stop():
                proc.terminate()
                proc.wait(timeout=120)
                ctx.log("time budget reached; stopped vLLM")
                break
    collect()
    ctx.save(answers)
    if proc.returncode not in (0, None) and len(seen) < len(reqs) and not ctx.should_stop():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"vLLM worker exited with {proc.returncode}:\n{tail}")
    ctx.log(f"vLLM done: {len(seen):,}/{len(reqs):,} rows in {time.time() - t0:.0f}s")


# ── adapters from earlier runs ───────────────────────────────────────────────


def resolve_artifact(ref, ctx, folder: str = "adapter", marker: str = "adapter_config.json"):
    """'run:<run_id>' -> local copy of the <folder>/ that run uploaded (downloaded once); anything else unchanged."""
    if not ref or not str(ref).startswith("run:"):
        return ref
    import os
    from pathlib import Path

    rid = str(ref)[4:]
    local = Path.cwd() / "runner_work" / f"{folder}s" / rid
    if not (local / marker).exists():
        from huggingface_hub import snapshot_download

        repo = os.environ.get("HF_RUNS_REPO", "nyakaruosborn/afro-health-qa-runs")
        ctx.log(f"downloading {folder} of {rid}")
        tmp = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"runs/{rid}/{folder}/*"])
        src = Path(tmp) / "runs" / rid / folder
        if not (src / marker).exists():
            raise FileNotFoundError(f"no {folder} uploaded for run {rid}")
        import shutil

        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, local, dirs_exist_ok=True)
    return str(local)


def resolve_adapter(adapter, ctx):
    return resolve_artifact(adapter, ctx, "adapter", "adapter_config.json")


# ── embedder fine-tuning (embedder_train) ────────────────────────────────────


def _embedder_key(ctx, cfg: dict) -> tuple:
    return ("trained_embedder",) + _adapter_key(ctx, cfg)[1:]


def embedder_triplets(cfg: dict, pool_df: pd.DataFrame, ctx, seed: int = 0) -> list[tuple[str, str, str]]:
    """(anchor, positive, hard negative) question triplets from the pool alone; positive = another question
    with the same answer, hard negative = the nearest retrieved question with a different answer."""
    import random

    rng = random.Random(seed)
    pool = pool_df.reset_index(drop=True)
    answers = pool[OUTPUT_COL].astype(str).str.strip().to_numpy()
    questions = pool[INPUT_COL].astype(str).to_numpy()
    ids = pool[ID_COL].astype(str).to_numpy()
    subs = cfg["embedder_train_subsets"]
    by_answer: dict[str, list[int]] = {}
    for j, a in enumerate(answers):
        by_answer.setdefault(a, []).append(j)
    rows = [j for j in range(len(pool)) if len(by_answer[answers[j]]) > 1 and (not subs or pool[SUBSET_COL].iloc[j] in subs)]
    if not rows:
        return []
    nn = neighbours(cfg, pool.iloc[rows], pool, ctx, k=12)
    out = []
    for j in rows:
        mate = rng.choice([h for h in by_answer[answers[j]] if h != j])
        neg = next((h for h, _ in nn[ids[j]] if h != j and answers[h] != answers[j]), None)
        if neg is not None:
            out.append((questions[j], questions[mate], questions[neg]))
    return out


def train_embedder(cfg: dict, pool_df: pd.DataFrame, ctx) -> str:
    """Fine-tune the sentence-transformers embedder in cfg['embedder']; returns its local directory."""
    import torch

    out = _work_dir(ctx, cfg) / "embedder"
    upload = getattr(ctx, "upload_folder", None)
    download = getattr(ctx, "download_folder", None)
    if (out / "modules.json").exists():
        return str(out)
    if download and download("embedder", out):
        ctx.log("found a trained embedder for this run on HF; skipping training")
        return str(out)
    trips = embedder_triplets(cfg, pool_df, ctx)
    ctx.log(f"embedder training: {len(trips):,} (anchor, positive, hard negative) triplets")
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(resolve_artifact(cfg["embedder"], ctx, "embedder", "modules.json"), device=dev)
    model.max_seq_length = int(cfg["embedder_train_max_len"])
    model.train()
    bs = int(cfg["embedder_train_batch"])
    steps = int(len(trips) * float(cfg["embedder_train_epochs"]) / bs)
    from transformers import get_cosine_schedule_with_warmup

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["embedder_train_lr"]), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), max(1, steps))
    import random

    order = list(range(len(trips)))
    random.Random(1).shuffle(order)

    def embed(texts):
        feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in model.tokenize(texts).items()}
        return torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)

    step = 0
    while step < steps:
        for b in range(0, len(order) - bs + 1, bs):
            if step >= steps or ctx.should_stop():
                break
            batch = [trips[i] for i in order[b:b + bs]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                q = embed([t[0] for t in batch])
                d = embed([t[1] for t in batch] + [t[2] for t in batch])  # positives, then hard negatives
                logits = (q @ d.T).float() * 20.0
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 50 == 0:
                ctx.log(f"embedder step {step}/{steps} loss {loss.item():.4f}")
        random.Random(step).shuffle(order)
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    if upload:
        upload(out, "embedder")
        ctx.log("embedder uploaded")
    del model, opt
    import gc

    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return str(out)


# ── reranker fine-tuning (rerank_train) ──────────────────────────────────────


def _reranker_key(ctx, cfg: dict) -> tuple:
    return ("trained_reranker",) + _adapter_key(ctx, cfg)[1:]


def answer_overlap(a: str, b: str, _cache: dict = {}) -> float:  # noqa: B006 (deliberate memo)
    """(ROUGE-1 F + ROUGE-L F) / 2 with the competition's whitespace tokenizer: 1.0 = same answer."""
    key = (a, b)
    if key not in _cache:
        from rouge_score import rouge_scorer

        class _WS:
            def tokenize(self, t):
                return str(t).strip().split() if t else []

        sc = _cache.setdefault("_scorer", rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=_WS()))
        r = sc.score(str(b), str(a))
        _cache[key] = (r["rouge1"].fmeasure + r["rougeL"].fmeasure) / 2
    return _cache[key]


def reranker_groups(cfg: dict, pool_df: pd.DataFrame, ctx, seed: int = 0) -> list[tuple[str, list[str], list[float]]]:
    """(query, [positive, negative, ...], labels) groups from the pool alone. Rows whose answer no other pool
    row shares have no positive and are skipped; negatives are retrieved neighbours with a different answer
    (the top ones for seed 0, a random sample of the top 3x for other seeds). labels = 1 for the positive and
    0 (or the answer overlap with the gold, when rerank_train_graded) for the negatives."""
    import random

    rng = random.Random(seed)
    pool = pool_df.reset_index(drop=True)
    answers = pool[OUTPUT_COL].astype(str).str.strip().to_numpy()
    questions = pool[INPUT_COL].astype(str).to_numpy()
    ids = pool[ID_COL].astype(str).to_numpy()
    by_answer: dict[str, list[int]] = {}
    for j, a in enumerate(answers):
        by_answer.setdefault(a, []).append(j)
    shared = [j for j in range(len(pool)) if len(by_answer[answers[j]]) > 1]
    if not shared:
        return []
    q_df = pool.iloc[shared]
    negs_wanted = int(cfg["rerank_train_negs"])
    nn = neighbours(cfg, q_df, pool, ctx, k=negs_wanted * 3 + 8)
    groups = []
    for j in shared:
        hits = [h for h, _ in nn[ids[j]] if h != j]
        mates = [h for h in by_answer[answers[j]] if h != j]
        retrieved_mates = [h for h in hits if answers[h] == answers[j]]
        pos = retrieved_mates[0] if retrieved_mates else rng.choice(mates)
        pool_negs = [h for h in hits if answers[h] != answers[j]]
        if len(pool_negs) < negs_wanted:
            continue
        negs = pool_negs[:negs_wanted] if seed == 0 else rng.sample(pool_negs[: negs_wanted * 3], negs_wanted)
        cand = lambda h: candidate_text(cfg, questions[h], answers[h])  # noqa: E731
        graded = bool(cfg["rerank_train_graded"])
        labels = [1.0] + [answer_overlap(answers[h], answers[j]) if graded else 0.0 for h in negs]
        groups.append((questions[j], [cand(pos)] + [cand(h) for h in negs], labels))
    return groups


def train_rerankers(cfg: dict, pool_df: pd.DataFrame, ctx):
    """One selector, or a list of rerank_ensemble selectors trained with seeds 0..K-1."""
    k = max(1, int(cfg["rerank_ensemble"]))
    paths = [train_reranker(cfg, pool_df, ctx, seed=s) for s in range(k)]
    return paths[0] if k == 1 else paths


def train_reranker(cfg: dict, pool_df: pd.DataFrame, ctx, seed: int = 0) -> str:
    """Fine-tune the cross-encoder in cfg['rerank_model'] on pool-only groups; returns its local directory."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

    name = "reranker" if seed == 0 else f"reranker_{seed}"
    out = _work_dir(ctx, cfg) / name
    upload = getattr(ctx, "upload_folder", None)
    download = getattr(ctx, "download_folder", None)
    if (out / "config.json").exists():
        return str(out)
    if download and download(name, out):
        ctx.log(f"found a trained {name} for this run on HF; skipping training")
        return str(out)
    groups = reranker_groups(cfg, pool_df, ctx, seed=seed)
    ctx.log(f"reranker training: {len(groups):,} query groups (1 positive + {cfg['rerank_train_negs']} negatives each)")
    base = resolve_artifact(cfg["rerank_model"], ctx, "reranker", "config.json")
    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(dev)
    model.train()
    bs, n_neg = int(cfg["rerank_train_batch"]), int(cfg["rerank_train_negs"])
    steps = int(len(groups) * float(cfg["rerank_train_epochs"]) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["rerank_train_lr"]), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), max(1, steps))
    import random

    order = list(range(len(groups)))
    random.Random(1 + seed).shuffle(order)
    torch.manual_seed(seed)
    graded, tau = bool(cfg["rerank_train_graded"]), float(cfg["rerank_train_tau"])
    step = 0
    while step < steps:
        for b in range(0, len(order) - bs + 1, bs):
            if step >= steps or ctx.should_stop():
                break
            batch = [groups[i] for i in order[b:b + bs]]
            qs = [q for q, cands, _ in batch for _ in cands]
            cs = [c for _, cands, _ in batch for c in cands]
            enc = tok(qs, cs, truncation=True, max_length=int(cfg["rerank_train_max_len"]), padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits.view(len(batch), n_neg + 1).float()
            if graded:  # match the score distribution to softmax(labels / tau)
                target = torch.softmax(torch.tensor([lab for _, _, lab in batch], device=dev) / tau, dim=-1)
                loss = -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 50 == 0:
                ctx.log(f"reranker step {step}/{steps} loss {loss.item():.4f}")
        random.Random(step).shuffle(order)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    if upload:
        upload(out, name)
        ctx.log(f"{name} uploaded")
    del model, opt
    import gc

    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return str(out)


# ── LoRA fine-tuning (mode lora_rag) ─────────────────────────────────────────


def _adapter_key(ctx, cfg: dict | None = None) -> tuple:
    # newer runners pass run_id; older ones don't, so fall back to a hash of the config
    rid = getattr(ctx, "run_id", None) or hashlib.sha256(repr(sorted((cfg or {}).items())).encode()).hexdigest()[:12]
    return ("lora_adapter", rid)


def _work_dir(ctx, cfg: dict):
    from pathlib import Path

    return getattr(ctx, "work_dir", None) or Path.cwd() / "runner_work" / "train" / _adapter_key(ctx, cfg)[1]


def setup(config: dict, sets: dict, ctx) -> None:
    """Called once per run before the eval sets. sets = {name: (eval_df, pool_df)}.

    lora_rag trains on the pool of the most honest set present: held_out's pool (work_train), else
    val's (Train), else test's (Train + Val). Eval rows are never in that pool.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["mode"] != "lora_rag" and not cfg["rerank_train"] and not cfg["embedder_train"]:
        return
    name = next(n for n in ("held_out", "val", "test") if n in sets)
    ctx.log(f"training on the {name} pool")
    if cfg["embedder_train"]:
        ctx.cache[_embedder_key(ctx, cfg)] = train_embedder(cfg, sets[name][1], ctx)
        cfg["embedder"] = ctx.cache[_embedder_key(ctx, cfg)]  # the reranker/LoRA below train on its candidates
    if cfg["rerank_train"]:
        ctx.cache[_reranker_key(ctx, cfg)] = train_rerankers(cfg, sets[name][1], ctx)
    if cfg["mode"] == "lora_rag":
        ctx.cache[_adapter_key(ctx, cfg)] = train_lora(cfg, sets[name][1], ctx)


def training_rows(cfg: dict, pool_df: pd.DataFrame, ctx) -> list[tuple[str, str, list[tuple[str, str]], str]]:
    """(question, answer, examples, subset) per training row; examples = k nearest OTHER pool rows, same subset."""
    frac = float(cfg["lora_data_frac"])
    subs = cfg.get("lora_train_subsets")
    rows = pool_df[pool_df[SUBSET_COL].isin(subs)] if subs else pool_df
    rows = rows if frac >= 1 else pd.concat(
        [g.sample(max(1, int(round(len(g) * frac))), random_state=0) for _, g in rows.groupby(SUBSET_COL)]
    )
    rows = rows.reset_index(drop=True)
    k = max(1, int(cfg["few_shot_k"]))
    nn = neighbours(cfg, rows, pool_df, ctx, k=k + 1)
    pool_ids = pool_df[ID_COL].astype(str).to_numpy()
    cap = int(cfg["rag_max_chars"])
    out = []
    for _, r in rows.iterrows():
        rid = str(r[ID_COL])
        hits = [j for j, _ in nn[rid] if pool_ids[j] != rid][:k]
        ex = [(str(pool_df[INPUT_COL].iloc[j]).strip()[:300], str(pool_df[OUTPUT_COL].iloc[j]).strip()[:cap]) for j in reversed(hits)]
        out.append((str(r[INPUT_COL]), str(r[OUTPUT_COL]).strip(), ex, str(r[SUBSET_COL])))
    return out


def lora_targets(model):
    """'all-linear' for text-only models; for multimodal ones (Gemma 3/4, Qwen3.5+) only the language model's
    attention/MLP projections, so no adapter weights go to the unused vision/audio towers."""
    names = [n for n, _ in model.named_modules()]
    if not any(t in n for n in names for t in ("vision", "audio")):
        return "all-linear"
    return r"^(?!.*(vision|audio|multi_modal|embed_)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def train_lora(cfg: dict, pool_df: pd.DataFrame, ctx) -> str:
    """Train (or resume, or reuse) a LoRA adapter; returns its local directory."""
    import dataclasses

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments)

    out = _work_dir(ctx, cfg)
    adapter_dir = out / "adapter"
    upload = getattr(ctx, "upload_folder", None)      # older runners can't upload or resume:
    download = getattr(ctx, "download_folder", None)  # training then simply runs start to finish
    if adapter_dir.joinpath("adapter_config.json").exists():
        ctx.log("found a finished adapter in this session; skipping training")
        return str(adapter_dir)
    if download and download("adapter", adapter_dir):
        ctx.log("found a finished adapter for this run on HF; skipping training")
        return str(adapter_dir)

    examples = training_rows(cfg, pool_df, ctx)
    if hasattr(ctx, "free_model"):
        ctx.free_model()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    max_len, data, skipped = int(cfg["lora_max_len"]), [], 0
    for q, a, ex, subset in examples:
        p_ids = tok(render_prompt(tok, build_messages(q, subset, ex)), add_special_tokens=False)["input_ids"]
        a_ids = tok(a + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p_ids) > max_len - 32:
            skipped += 1
            continue
        ids = (p_ids + a_ids)[:max_len]
        data.append({"input_ids": ids, "labels": ([-100] * len(p_ids) + a_ids)[:max_len]})
    ctx.log(f"{len(data):,} training sequences ({skipped} skipped: prompt longer than {max_len} tokens)")

    import transformers

    ctx.log(f"transformers {transformers.__version__}, torch {torch.__version__}; base {cfg['model_id']} ({int(cfg['lora_bits'])}-bit)")
    load = dict(dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if int(cfg["lora_bits"]) == 4:
        from transformers import BitsAndBytesConfig

        load["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **load)
    model.config.use_cache = False
    if int(cfg["lora_bits"]) == 4:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
        bias="none", task_type="CAUSAL_LM", target_modules=lora_targets(model),
    ))

    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    compat = {"warmup_ratio": float(cfg["lora_warmup"])} if "warmup_ratio" in fields else {"warmup_steps": float(cfg["lora_warmup"])}
    args = TrainingArguments(
        output_dir=str(out / "trainer"), num_train_epochs=float(cfg["lora_epochs"]),
        per_device_train_batch_size=int(cfg["lora_batch"]), gradient_accumulation_steps=int(cfg["lora_grad_acc"]),
        learning_rate=float(cfg["lora_lr"]), lr_scheduler_type="cosine", optim=cfg["lora_optim"], bf16=True,
        logging_steps=20, save_strategy="steps", save_steps=int(cfg["lora_save_steps"]), save_total_limit=1,
        report_to="none", seed=3407, remove_unused_columns=False, **compat,
    )

    class Progress(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                ctx.log(f"step {state.global_step}/{state.max_steps} loss {logs['loss']:.4f}")

        def on_save(self, a, state, control, **kw):
            ck = out / "trainer" / f"checkpoint-{state.global_step}"
            if ck.exists() and upload:
                upload(ck, "train_ckpt")
                ctx.log(f"checkpoint {state.global_step} uploaded")

    trainer = Trainer(model=model, args=args, train_dataset=data, callbacks=[Progress()],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8))
    resume_dir = out / "resume_ckpt"
    resume = str(resume_dir) if download and download("train_ckpt", resume_dir) else None
    if resume:
        ctx.log("resuming training from the checkpoint on HF")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    if upload:
        upload(adapter_dir, "adapter")
        ctx.log("adapter uploaded")
    del trainer, model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return str(adapter_dir)


# ── entry point ──────────────────────────────────────────────────────────────


def apply_fallback(cfg: dict, eval_df, pool_df, answers: dict, best_answer: dict, meta: dict, ctx) -> None:
    """Replace generations far shorter than a typical answer with the retrieved answer."""
    frac = float(cfg["fallback_below_frac"])
    if frac <= 0 or not best_answer:
        return
    median_words = pool_df.groupby(SUBSET_COL)[OUTPUT_COL].apply(lambda x: x.str.split().str.len().median()).to_dict()
    n = 0
    for _, r in eval_df.iterrows():
        i, sub = str(r[ID_COL]), str(r[SUBSET_COL])
        if i in answers and i in best_answer and len(str(answers[i]).split()) < frac * median_words.get(sub, 0):
            answers[i] = best_answer[i]
            meta.setdefault(i, {})["fallback"] = True
            n += 1
    ctx.log(f"fallback to retrieval for {n:,} short generations (< {frac:.0%} of the subset's median length)")


def run(config: dict, eval_df: pd.DataFrame, pool_df: pd.DataFrame, ctx) -> tuple[dict[str, str], dict[str, dict]]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    unknown = set(config or {}) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    mode = cfg["mode"]
    eval_df = eval_df.reset_index(drop=True)
    pool_df = pool_df.reset_index(drop=True)
    answers: dict[str, str] = dict(ctx.resume)
    meta: dict[str, dict] = {}
    cfg["adapter"] = resolve_adapter(cfg["adapter"], ctx)
    if cfg["embedder_train"]:
        ekey = _embedder_key(ctx, cfg)
        if ekey not in ctx.cache:  # older runner without the setup step: train on this set's pool
            ctx.cache[ekey] = train_embedder(cfg, pool_df, ctx)
        cfg["embedder"] = ctx.cache[ekey]
    elif str(cfg["embedder"]).startswith("run:"):
        cfg["embedder"] = resolve_artifact(cfg["embedder"], ctx, "embedder", "modules.json")
    if cfg["rerank_train"]:
        key = _reranker_key(ctx, cfg)
        if key not in ctx.cache:  # older runner without the setup step: train on this set's pool
            ctx.cache[key] = train_rerankers(cfg, pool_df, ctx)
        cfg["rerank_model"] = ctx.cache[key]
    elif cfg["rerank_model"] and not isinstance(cfg["rerank_model"], (list, tuple)):
        cfg["rerank_model"] = resolve_artifact(cfg["rerank_model"], ctx, "reranker", "config.json")

    needs_nn = mode in {"retrieval", "rag_few_shot", "router", "lora_rag"}
    if mode == "lora_rag":
        key = _adapter_key(ctx, cfg)
        if key not in ctx.cache:  # older runner without the setup step: train inline on this set's pool
            ctx.log("no setup step in this runner; training the adapter inline on this set's pool")
            ctx.cache[key] = train_lora(cfg, pool_df, ctx)
        cfg["adapter"] = ctx.cache[key]
    uses_rag = mode in {"rag_few_shot", "lora_rag"} or (mode == "router" and cfg["router_generate_mode"] == "rag_few_shot")
    k_nn = max(1, int(cfg["few_shot_k"])) if uses_rag else 1
    if mode in {"retrieval", "router"} and cfg["select"] == "vote":
        k_nn = max(k_nn, int(cfg["vote_k"]))
    if needs_nn and cfg["rerank_model"]:
        k_nn = max(k_nn, int(cfg["rerank_k"]))
    dk = int(cfg["diag_k"])
    if needs_nn and dk:
        k_nn = max(k_nn, dk)
    nn = neighbours(cfg, eval_df, pool_df, ctx, k=k_nn) if needs_nn else {}
    pool_ids = pool_df[ID_COL].astype(str).to_numpy()
    for i, hits in nn.items():
        meta[i] = {"sim": round(hits[0][1], 4), "nn_id": str(pool_ids[hits[0][0]])}
        if dk:
            meta[i]["ret_ids"] = "|".join(pool_ids[j] for j, _ in hits[:dk])
    if nn and cfg["rerank_model"]:
        nn = rerank(cfg, eval_df, pool_df, nn, ctx)
        for i, hits in nn.items():
            meta[i].update(rerank_score=round(hits[0][1], 4), rerank_id=str(pool_ids[hits[0][0]]))
            if dk:
                meta[i]["cand_ids"] = "|".join(pool_ids[j] for j, _ in hits[:dk])
    best_answer: dict[str, str] = {}
    for i, hits in nn.items():
        if cfg["select"] == "vote":
            votes: dict[str, float] = {}
            for j, sim in hits:
                a = str(pool_df[OUTPUT_COL].iloc[j])
                votes[a] = votes.get(a, 0.0) + max(sim, 0.0) ** float(cfg["vote_power"])
            best_answer[i] = max(votes, key=votes.get)
        else:
            best_answer[i] = str(pool_df[OUTPUT_COL].iloc[hits[0][0]])

    if mode == "retrieval":
        answers.update(best_answer)
        return answers, meta

    def rag_examples(i: str, subset: str):
        cap = int(cfg["rag_max_chars"])
        # nearest first in the list = furthest from the question in the prompt; put nearest last
        return [(str(pool_df[INPUT_COL].iloc[j]).strip()[:300], str(pool_df[OUTPUT_COL].iloc[j]).strip()[:cap])
                for j, _ in reversed(nn[i][: max(1, int(cfg["few_shot_k"]))])]

    bank = fixed_examples(cfg, pool_df) if "few_shot" in (mode, cfg["router_generate_mode"]) else {}
    pickers = {
        "zero_shot": lambda i, s: [],
        "few_shot": lambda i, s: bank.get(s, []),
        "rag_few_shot": rag_examples,
        "lora_rag": rag_examples,  # same prompt shape the adapter was trained on
    }

    if mode in GENERATIVE_MODES:
        if int(cfg["gen_samples"]) > 0:
            generate(cfg, eval_df, pickers[mode], ctx, answers, pool_df, meta=meta)
        else:
            generate(cfg, eval_df, pickers[mode], ctx, answers, pool_df)
        apply_fallback(cfg, eval_df, pool_df, answers, best_answer, meta, ctx)
        return answers, meta

    if mode == "router":
        thr = float(cfg["router_threshold"])
        copy_ids = {i for i, m in meta.items() if m["sim"] >= thr}
        for i in copy_ids:
            answers[i] = best_answer[i]
        for i in meta:
            meta[i]["route"] = "copy" if i in copy_ids else "generate"
        ctx.log(f"router: copying {len(copy_ids):,} / {len(eval_df):,} rows at sim >= {thr}")
        gen_rows = eval_df[~eval_df[ID_COL].astype(str).isin(copy_ids)]
        generate(cfg, gen_rows, pickers[cfg["router_generate_mode"]], ctx, answers, pool_df)
        return answers, meta

    raise ValueError(f"unknown mode {mode!r}")
