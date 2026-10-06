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
    # retrieval | zero_shot | few_shot | rag_few_shot | router
    "mode": "retrieval",
    # retrieval (used by retrieval, rag_few_shot, router)
    "embedder": "BAAI/bge-m3",   # any sentence-transformers id, or "tfidf-char" (CPU)
    "query_prefix": "",          # e.g. "query: " for intfloat/multilingual-e5-*
    "passage_prefix": "",        # e.g. "passage: " for intfloat/multilingual-e5-*
    "retrieve_on": "input",      # match eval questions against pool questions
    "embed_batch": 64,
    "hybrid_with": None,         # e.g. "tfidf-char": blend a second retriever's similarity in
    "hybrid_alpha": 0.5,         # weight of hybrid_with in the blend
    "rerank_model": None,        # e.g. "BAAI/bge-reranker-v2-m3": cross-encoder over the top rerank_k
    "rerank_k": 20,
    "rerank_on": "question",     # question (paraphrase check) | answer (relevance check)
    "rerank_batch": 64,
    "select": "top1",            # top1 | vote: sum sim**vote_power over neighbours sharing an answer
    "vote_k": 50,
    "vote_power": 4.0,
    # generation (zero_shot, few_shot, rag_few_shot, router)
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "precision": "auto",         # auto | bf16 | 4bit
    "adapter": None,
    "num_beams": 1,
    "infer_batch": 32,
    "max_input_tokens": 1024,
    "no_repeat_ngram": 3,
    "length_penalty": 1.0,
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
    "checkpoint_every": 10,      # batches
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

GENERATIVE_MODES = {"zero_shot", "few_shot", "rag_few_shot"}


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


def rerank(cfg: dict, eval_df: pd.DataFrame, pool_df: pd.DataFrame, nn: dict, ctx) -> dict[str, list[tuple[int, float]]]:
    """Re-score each eval row's top rerank_k candidates with a cross-encoder; returns them sorted by that score.

    rerank_on="question" scores (eval question, neighbour's question): a paraphrase check.
    rerank_on="answer" scores (eval question, neighbour's answer): a relevance check.
    """
    key = ("cross_encoder", cfg["rerank_model"])
    if key not in ctx.cache:
        import torch
        from sentence_transformers import CrossEncoder

        ctx.cache[key] = CrossEncoder(cfg["rerank_model"], max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
    ce = ctx.cache[key]
    col = INPUT_COL if cfg["rerank_on"] == "question" else OUTPUT_COL
    questions = dict(zip(eval_df[ID_COL].astype(str), eval_df[INPUT_COL].astype(str)))
    k = int(cfg["rerank_k"])
    pairs, owners = [], []
    for i, hits in nn.items():
        for j, _ in hits[:k]:
            pairs.append((questions[i], str(pool_df[col].iloc[j])))
            owners.append((i, j))
    ctx.log(f"reranking {len(pairs):,} pairs with {cfg['rerank_model']} on {cfg['rerank_on']}")
    scores = ce.predict(pairs, batch_size=int(cfg["rerank_batch"]), show_progress_bar=False)
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
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
        out[str(subset)] = {"min_new_tokens": 1, "max_new_tokens": int(np.percentile(lens, 95)) + 8}
    return out


def generate(cfg: dict, rows: pd.DataFrame, examples_for, ctx, answers: dict[str, str], pool_df: pd.DataFrame) -> None:
    """Fill answers[ID] for every row not already present. examples_for(ID, subset) -> examples."""
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
        for i, t in zip(ids, texts):
            answers[i] = postprocess(t, subset)
        if b % every == 0:
            ctx.save(answers)
        ctx.log(f"batch {b}/{len(batches)} ({subset})", progress=(b, len(batches)))
    ctx.save(answers)


# ── entry point ──────────────────────────────────────────────────────────────


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

    needs_nn = mode in {"retrieval", "rag_few_shot", "router"}
    uses_rag = mode == "rag_few_shot" or (mode == "router" and cfg["router_generate_mode"] == "rag_few_shot")
    k_nn = max(1, int(cfg["few_shot_k"])) if uses_rag else 1
    if mode in {"retrieval", "router"} and cfg["select"] == "vote":
        k_nn = max(k_nn, int(cfg["vote_k"]))
    if needs_nn and cfg["rerank_model"]:
        k_nn = max(k_nn, int(cfg["rerank_k"]))
    nn = neighbours(cfg, eval_df, pool_df, ctx, k=k_nn) if needs_nn else {}
    for i, hits in nn.items():
        meta[i] = {"sim": round(hits[0][1], 4), "nn_id": str(pool_df[ID_COL].iloc[hits[0][0]])}
    if nn and cfg["rerank_model"]:
        nn = rerank(cfg, eval_df, pool_df, nn, ctx)
        for i, hits in nn.items():
            meta[i].update(rerank_score=round(hits[0][1], 4), rerank_id=str(pool_df[ID_COL].iloc[hits[0][0]]))
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
    }

    if mode in GENERATIVE_MODES:
        generate(cfg, eval_df, pickers[mode], ctx, answers, pool_df)
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
