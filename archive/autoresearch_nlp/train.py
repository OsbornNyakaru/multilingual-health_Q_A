"""train.py — afro-health-qa instance (Zindi Multilingual Health QA, due 2026-06-21).

The SINGLE file the agent edits. prepare.py is frozen and already configured for
this competition (ID/input/output/subset; 4-col submission; weights 0.37/0.37/0.26).

One experiment = ONE change to the CONFIG block, then:
    python train.py > run.log 2>&1
    grep "^combined:\\|^rouge1_f1:\\|^rougeL_f1:" run.log

DRY_RUN=True runs end-to-end on CPU with a stub generator (no GPU/model). Flip to
False on a GPU after you've chosen and verified a model (see MODEL_DECISION.md).
"""

from __future__ import annotations

import pandas as pd

import prepare

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — change ONE thing per experiment.
# ════════════════════════════════════════════════════════════════════════════

RUN_NAME = "exp001_zeroshot_baseline"
DRY_RUN = True  # staged. Set False on GPU once a model is chosen + verified.

# MODEL — this is a DAY-1 DECISION, not a default. See MODEL_DECISION.md.
# CohereLabs/aya-expanse-8b is verified to exist BUT is CC-BY-NC (license risk for
# a prize comp) and does NOT list Swahili/Luganda/Akan/Amharic among its languages.
# Run AGENT_PLAYBOOK.md prompt #3 (model survey) before committing real compute.
MODEL_ID = "CohereLabs/aya-expanse-8b"
ADAPTER_DIR = None
LOAD_IN_4BIT = True

# Decoding
NUM_BEAMS = 4
DO_SAMPLE = False
NO_REPEAT_NGRAM = 3
LENGTH_PENALTY = 1.0

MAX_INPUT_TOKENS = 512
INFER_BATCH = 8

# ── Language handling ───────────────────────────────────────────────────────
# 8 subsets (locale codes) map to 5 languages. Answer in the SAME language.
SUBSET_TO_LANG = {
    "Aka_Gha": "aka",
    "Amh_Eth": "amh",
    "Eng_Eth": "eng",
    "Eng_Gha": "eng",
    "Eng_Ken": "eng",
    "Eng_Uga": "eng",
    "Lug_Uga": "lug",
    "Swa_Ken": "swa",
}

SYSTEM_INSTRUCTIONS = {
    "eng": "You are an expert health worker. Answer the health question accurately and concisely in English, in the same style as a clinical reference answer.",
    "swa": "Wewe ni mtaalamu wa afya. Jibu swali la afya kwa usahihi na kwa ufupi kwa Kiswahili.",
    "lug": "Oli omukugu mu by'obulamu. Ddamu ekibuuzo ky'obulamu mu bujjuvu era mu Luganda.",
    "aka": "Woyɛ apɔmuden ho nimdefoɔ. Bua apɔmuden asɛmmisa no yiye wɔ Akan kasa mu.",
    "amh": "እርስዎ የጤና ባለሙያ ነዎት። ይህን የጤና ጥያቄ በትክክል እና በአጭሩ በአማርኛ ይመልሱ።",
}
PROMPT_TEMPLATES = {
    "eng": "Question: {question}\nAnswer:",
    "swa": "Swali: {question}\nJibu:",
    "lug": "Ekibuuzo: {question}\nEky'okuddamu:",
    "aka": "Asɛmmisa: {question}\nMmuaeɛ:",
    "amh": "ጥያቄ: {question}\nመልስ:",
}
ANSWER_MARKERS = {"eng": "Answer:", "swa": "Jibu:", "lug": "Eky'okuddamu:",
                  "aka": "Mmuaeɛ:", "amh": "መልስ:"}

# ── Per-subset length bounds (subword tokens), from tools/length_calibrate.py ──
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

# ════════════════════════════════════════════════════════════════════════════
# Prompt building + post-processing
# ════════════════════════════════════════════════════════════════════════════


def lang_of(subset: str) -> str:
    return SUBSET_TO_LANG.get(subset, "eng")


def build_prompt(question: str, subset: str) -> str:
    lang = lang_of(subset)
    sys = SYSTEM_INSTRUCTIONS[lang]
    user = PROMPT_TEMPLATES[lang].format(question=question.strip())
    return f"{sys}\n\n{user}"


def postprocess(text: str, subset: str) -> str:
    text = str(text).strip()
    marker = ANSWER_MARKERS.get(lang_of(subset), "")
    if marker and marker in text:
        text = text.split(marker, 1)[-1].strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text


# ════════════════════════════════════════════════════════════════════════════
# Generation (batched by subset so length bounds apply correctly)
# ════════════════════════════════════════════════════════════════════════════


def generate(df: pd.DataFrame) -> list[str]:
    df = df.reset_index(drop=True)
    if DRY_RUN:
        # Stub proves the pipeline; replaced by the model when DRY_RUN=False.
        return ["information"] * len(df)

    import torch

    if not hasattr(generate, "cached"):
        generate.cached = load_model()
    tok, model = generate.cached

    out = [""] * len(df)
    for subset, grp in df.groupby(prepare.SUBSET_COL):
        bounds = LENGTH_BOUNDS.get(subset, DEFAULT_BOUNDS)
        idxs = grp.index.tolist()
        for i in range(0, len(idxs), INFER_BATCH):
            chunk = idxs[i:i + INFER_BATCH]
            prompts = [build_prompt(str(df.loc[j, prepare.INPUT_COL]), str(subset)) for j in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_INPUT_TOKENS).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc, num_beams=NUM_BEAMS, do_sample=DO_SAMPLE,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM, length_penalty=LENGTH_PENALTY,
                    min_new_tokens=bounds["min_new_tokens"], max_new_tokens=bounds["max_new_tokens"],
                    pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, text in zip(chunk, tok.batch_decode(new, skip_special_tokens=True)):
                out[j] = postprocess(text, str(subset))
    return out


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if ADAPTER_DIR:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return tok, model


# ════════════════════════════════════════════════════════════════════════════
# Experiment: score on held-out (per subset), write the test submission
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    held = pd.read_csv(prepare.CACHE_DIR / "held_out.csv", dtype=str).fillna("")
    test = pd.read_csv(prepare.DATA_DIR / "Test.csv", dtype=str).fillna("")

    held = held.copy()
    raw_preds = generate(held)
    held["pred"] = [postprocess(p, s) for p, s in zip(raw_preds, held[prepare.SUBSET_COL])]
    s = prepare.score(held["pred"].tolist(), held[prepare.OUTPUT_COL].tolist())
    per = prepare.score_per_subset(held, "pred")
    print(prepare.report(RUN_NAME, s, extra="per_subset:\n" + per.to_string()))

    test_raw = generate(test)
    test_preds = [postprocess(p, s) for p, s in zip(test_raw, test[prepare.SUBSET_COL])]
    sub = prepare.build_submission(test[prepare.ID_COL].tolist(), test_preds)
    prepare.validate_submission(sub, expected_ids=test[prepare.ID_COL].tolist())
    out_path = prepare.CACHE_DIR.parent.parent / "submissions" / f"{RUN_NAME}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"submission: {out_path}")


if __name__ == "__main__":
    main()
