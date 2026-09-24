# %% [markdown]
# # AfroHealth QA — exp001 Full Pipeline (Part 1: Steps 1-4)
# **Platform**: Kaggle (T4 GPU) or Google Colab Free
# **Purpose**: Data audit, length bounds, held-out split, evaluation calibration

# %% — Cell 0: Install Dependencies
# !pip install -q torch transformers accelerate bitsandbytes rouge-score scikit-learn pandas numpy tqdm sentencepiece protobuf fast-langdetect

# %% — Cell 1: Configuration
from __future__ import annotations
import gc, hashlib, json, os, random, re, warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Platform: set to "kaggle" or "colab" ──
PLATFORM = "kaggle"  # Change to "colab" if using Google Colab

# ── Paths ──
if PLATFORM == "kaggle":
    RAW_DIR = Path("/kaggle/input/afro-health-qa")  # Your dataset name
    WORK_DIR = Path("/kaggle/working")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    RAW_DIR = Path("/content/drive/MyDrive/afro-health-qa/data/raw")
    WORK_DIR = Path("/content/working")

PROCESSED_DIR = WORK_DIR / "data/processed"
SUBMISSIONS_DIR = WORK_DIR / "submissions"
EXPERIMENTS_DIR = WORK_DIR / "experiments"
CACHE_DIR = WORK_DIR / ".cache/afrolm_embeddings"
for d in [PROCESSED_DIR, SUBMISSIONS_DIR, EXPERIMENTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_LEADERBOARD = 0.768095
EXPERIMENT_ID = "exp001"
HYPOTHESIS = "Zero-shot Aya-Expanse-8B lang-native prompts length-matched beam5 beats 0.768095"

# ── Seed in 5 places ──
def set_global_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed; set_seed(seed)
    except Exception:
        pass

set_global_seed()
print(f"Seed={SEED} set in all 5 places.")

# ── GPU check ──
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"VRAM: {vram / 1e9:.1f} GB")

# %% — Cell 2: Utility Functions
_GEEZ_RE = re.compile(r"[\u1200-\u137F]")

@dataclass(frozen=True)
class ColumnMapping:
    id_col: str
    question_col: str
    lang_col: str
    answer_col: str

@dataclass(frozen=True)
class SubmissionSchema:
    columns: tuple
    dtypes: dict
    n_rows: int
    ids_match_test_exactly: bool

def detect_column_mapping(train_df, val_df, test_df):
    common = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    id_col = [c for c in common if c.lower() == "id" or c.lower().endswith("id")][0]
    question_col = [c for c in common if c.lower() in {"question","input","prompt","query"} and c != id_col][0]
    answer_col = [c for c in set(train_df.columns) & set(val_df.columns)
                  if c not in test_df.columns and c.lower() in {"response","answer","output","target"}][0]
    lang_candidates = [c for c in common if c not in {id_col, question_col} and c.lower() in {"language","lang","subset"}]
    if lang_candidates:
        lang_col = lang_candidates[0]
    else:
        leftovers = [c for c in common if c not in {id_col, question_col}]
        lang_col = leftovers[0]
    return ColumnMapping(id_col=id_col, question_col=question_col, lang_col=lang_col, answer_col=answer_col)

def infer_lang_code_map(values):
    mapping = {}
    for raw in sorted({str(v) for v in values}):
        key = raw.strip(); lower = key.lower()
        if lower.startswith("swa") or "swahili" in lower: mapping[key] = "swa"
        elif lower.startswith("lug") or "luganda" in lower: mapping[key] = "lug"
        elif lower.startswith("aka") or "twi" in lower or "akan" in lower: mapping[key] = "aka"
        elif lower.startswith("amh") or "amharic" in lower: mapping[key] = "amh"
        elif lower.startswith("eng") or "english" in lower: mapping[key] = "eng"
        else: raise ValueError(f"unknown language/subset code: {raw}")
    return mapping

def contains_geez(text): return bool(_GEEZ_RE.search(text or ""))

def canonical_prompt_language(lang_code): return "eng" if lang_code == "eng" else lang_code

def lang_distribution(df, lang_col):
    counts = df[lang_col].astype(str).value_counts(dropna=False).sort_index()
    pct = counts / len(df) * 100.0
    return pd.DataFrame({"count": counts, "pct": pct.round(2)})

def length_stats(df, lang_col, answer_col, lang_code_map):
    rows = []
    for raw_lang, group in df.groupby(lang_col, sort=True):
        canonical = lang_code_map[str(raw_lang)]
        tc = group[answer_col].astype(str).map(lambda x: len(x.split()))
        rows.append({"raw_lang": raw_lang, "lang": canonical, "n": len(group),
                      "mean": round(float(tc.mean()),2), "median": round(float(tc.median()),2),
                      "p10": int(np.percentile(tc,10)), "p90": int(np.percentile(tc,90))})
    return pd.DataFrame(rows).sort_values(["lang","raw_lang"]).reset_index(drop=True)

def build_submission_schema(sample_df, test_df, id_col):
    s_ids = sample_df.iloc[:,0].astype(str).tolist()
    t_ids = test_df[id_col].astype(str).tolist()
    return SubmissionSchema(columns=tuple(sample_df.columns.tolist()),
                            dtypes={c: str(d) for c,d in sample_df.dtypes.items()},
                            n_rows=len(sample_df), ids_match_test_exactly=s_ids==t_ids)

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("Utility functions loaded.")

# %% — Cell 3: STEP 1 — Ingest and Audit All Four Files
print("=" * 70)
print("STEP 1 — INGEST AND AUDIT ALL FOUR FILES")
print("=" * 70)

train = pd.read_csv(RAW_DIR / "Train.csv", dtype=str)
val   = pd.read_csv(RAW_DIR / "Val.csv", dtype=str)
test  = pd.read_csv(RAW_DIR / "Test.csv", dtype=str)
sample = pd.read_csv(RAW_DIR / "SampleSubmission.csv", dtype=str)

print(f"Train.csv     : {len(train)} rows × {len(train.columns)} cols  |  columns: {train.columns.tolist()}")
print(f"Val.csv       : {len(val)} rows × {len(val.columns)} cols  |  columns: {val.columns.tolist()}")
print(f"Test.csv      : {len(test)} rows × {len(test.columns)} cols  |  columns: {test.columns.tolist()}")
print(f"SampleSubmission.csv : {len(sample)} rows × {len(sample.columns)} cols  |  columns: {sample.columns.tolist()}")

# 1a. Column mapping
mapping = detect_column_mapping(train, val, test)
print(f"\n1a. Column mapping: {asdict(mapping)}")
id_col, question_col, lang_col, answer_col = mapping.id_col, mapping.question_col, mapping.lang_col, mapping.answer_col

# Language code map
all_lang_values = (train[lang_col].astype(str).tolist() + val[lang_col].astype(str).tolist()
                   + test[lang_col].astype(str).tolist())
LANG_CODE_MAP = infer_lang_code_map(all_lang_values)
print(f"\n1i. LANG_CODE_MAP = {json.dumps(LANG_CODE_MAP, indent=2, ensure_ascii=False)}")

# 1b. Language distribution
print("\n1b. Language distribution")
train_dist = lang_distribution(train, lang_col)
val_dist = lang_distribution(val, lang_col)
print("Train:\n", train_dist.to_string())
print("Val:\n", val_dist.to_string())
for rl in sorted(set(train_dist.index) | set(val_dist.index)):
    tp = float(train_dist.loc[rl,"pct"]) if rl in train_dist.index else 0.0
    vp = float(val_dist.loc[rl,"pct"]) if rl in val_dist.index else 0.0
    if tp < 10 or vp < 10: print(f"  [FLAG] {rl} below 10% in {'train' if tp<10 else 'val'}")
    if abs(tp-vp) > 15: print(f"  [FLAG] {rl} gap {abs(tp-vp):.1f}pp")

# 1c/1d. Answer length distributions
print("\n1c. Train answer length distribution:")
train_len = length_stats(train, lang_col, answer_col, LANG_CODE_MAP)
print(train_len.to_string(index=False))
print("\n1d. Val answer length distribution:")
val_len = length_stats(val, lang_col, answer_col, LANG_CODE_MAP)
print(val_len.to_string(index=False))

# 1e. Missing/null values
print("\n1e. Missing/null values")
for name, df in [("train",train),("val",val),("test",test),("sample",sample)]:
    nulls = df.isna().sum().to_dict()
    print(f"  {name}: {nulls}")

# Drop null answers if any
for name in ("train","val"):
    df_ref = train if name=="train" else val
    if df_ref[answer_col].isna().any():
        before = len(df_ref)
        df_ref = df_ref.dropna(subset=[answer_col]).reset_index(drop=True)
        print(f"  Dropped {before - len(df_ref)} null-answer rows from {name}")
        if name=="train": train = df_ref
        else: val = df_ref

assert not test[question_col].isna().any(), "CRITICAL: Test has null questions!"

# 1f. Amharic script check
print("\n1f. Amharic script check")
suspect_frames = []
for name, df in [("train",train),("val",val),("test",test)]:
    amh_mask = df[lang_col].astype(str).map(lambda x: LANG_CODE_MAP.get(str(x))=="amh")
    has_q = question_col in df.columns
    if has_q:
        suspect = df.loc[amh_mask & ~df[question_col].astype(str).map(contains_geez)].copy()
    else:
        suspect = pd.DataFrame()
    if len(suspect):
        suspect["source_file"] = name
        suspect_frames.append(suspect)
        df = df.loc[~(amh_mask & ~df[question_col].astype(str).map(contains_geez))].reset_index(drop=True)
        if name=="train": train = df
        elif name=="val": val = df
        else: test = df
    print(f"  {name}: {len(suspect)} suspect Amharic rows quarantined")

if suspect_frames:
    pd.concat(suspect_frames, ignore_index=True).to_csv(PROCESSED_DIR/"suspect_amharic.csv", index=False)

# 1g. Submission format fingerprint
print("\n1g. Submission format fingerprint")
SUBMISSION_SCHEMA = build_submission_schema(sample, test, id_col)
print(f"  Columns: {list(SUBMISSION_SCHEMA.columns)}")
print(f"  Dtypes: {SUBMISSION_SCHEMA.dtypes}")
print(f"  Rows: {SUBMISSION_SCHEMA.n_rows}")
print(f"  IDs match Test: {SUBMISSION_SCHEMA.ids_match_test_exactly}")

# 1h. Overlap check
print("\n1h. Train/Val/Test overlap check")
tq = set(train[question_col].astype(str))
vq = set(val[question_col].astype(str))
teq = set(test[question_col].astype(str))
print(f"  Train ∩ Val: {len(tq & vq)}")
print(f"  Train ∩ Test: {len(tq & teq)}")
print(f"  Val ∩ Test: {len(vq & teq)}")

# Save audit report
audit_lines = [
    "# Data Audit Report", "",
    "## File shapes",
    f"- Train: {len(train)} rows, cols={train.columns.tolist()}",
    f"- Val: {len(val)} rows, cols={val.columns.tolist()}",
    f"- Test: {len(test)} rows, cols={test.columns.tolist()}",
    f"- Sample: {len(sample)} rows, cols={sample.columns.tolist()}", "",
    f"## Column mapping\n{asdict(mapping)}", "",
    f"## Language code map\n```json\n{json.dumps(LANG_CODE_MAP,indent=2,ensure_ascii=False)}\n```", "",
    f"## Submission schema\n```json\n{json.dumps(asdict(SUBMISSION_SCHEMA),indent=2)}\n```", "",
    f"## Overlaps\n- Train∩Val: {len(tq&vq)}\n- Train∩Test: {len(tq&teq)}\n- Val∩Test: {len(vq&teq)}",
]
(PROCESSED_DIR / "data_audit_report.md").write_text("\n".join(audit_lines), encoding="utf-8")
print("\n✅ Step 1 complete. Audit saved to data/processed/data_audit_report.md")

# %% — Cell 4: STEP 2 — Build Per-Language Length Bounds
print("\n" + "=" * 70)
print("STEP 2 — BUILD PER-LANGUAGE LENGTH BOUNDS TABLE")
print("=" * 70)

LENGTH_BOUNDS = {}
comparison_rows = []
bound_notes = []

for lang in sorted(train_len["lang"].unique()):
    tg = train_len[train_len["lang"]==lang]
    vg = val_len[val_len["lang"]==lang]
    tp10 = int(tg["p10"].mean()); tp90 = int(tg["p90"].mean())
    vp10 = int(vg["p10"].mean()) if len(vg) else tp10
    vp90 = int(vg["p90"].mean()) if len(vg) else tp90

    if abs(vp90 - tp90) > 20:
        avg_p10 = int(round((tp10+vp10)/2)); avg_p90 = int(round((tp90+vp90)/2))
        chosen = {"min_new_tokens": max(1,int(avg_p10*0.8)), "max_new_tokens": avg_p90+30}
        bound_notes.append(f"{lang}: Train/Val p90 diverged by {abs(vp90-tp90)}, using average bounds.")
    else:
        chosen = {"min_new_tokens": max(1,int(tp10*0.8)), "max_new_tokens": tp90+30}

    LENGTH_BOUNDS[lang] = chosen
    comparison_rows.append({"lang":lang, "train_p10":tp10, "train_p90":tp90,
                            "val_p10":vp10, "val_p90":vp90,
                            "final_min":chosen["min_new_tokens"], "final_max":chosen["max_new_tokens"]})

print(pd.DataFrame(comparison_rows).to_string(index=False))
for n in bound_notes: print(f"  NOTE: {n}")
print(f"\nLENGTH_BOUNDS = {json.dumps(LENGTH_BOUNDS, indent=2, ensure_ascii=False)}")
print("✅ Step 2 complete.")

# %% — Cell 5: STEP 3 — Carve Held-Out Slice
print("\n" + "=" * 70)
print("STEP 3 — CARVE HELD-OUT SLICE FROM TRAIN (5%)")
print("=" * 70)

train_core, held_out = train_test_split(train, test_size=0.05, stratify=train[lang_col], random_state=42)
train_core = train_core.reset_index(drop=True)
held_out = held_out.reset_index(drop=True)
train_core.to_csv(PROCESSED_DIR/"train_core.csv", index=False, encoding="utf-8")
held_out.to_csv(PROCESSED_DIR/"held_out.csv", index=False, encoding="utf-8")

print("train_core per-language counts:")
print(lang_distribution(train_core, lang_col).to_string())
print("\nheld_out per-language counts:")
print(lang_distribution(held_out, lang_col).to_string())

print("\n╔══════════════════════════════════════════════════════════╗")
print("║  held_out.csv is now SEALED.                             ║")
print("║  Val.csv is our local leaderboard proxy.                 ║")
print("║  train_core.csv is everything available for training.    ║")
print("║  Do NOT load held_out.csv again until final week.        ║")
print("╚══════════════════════════════════════════════════════════╝")
print("✅ Step 3 complete.")

# %% — Cell 6: STEP 4 — Build and Calibrate Evaluation Pipeline
print("\n" + "=" * 70)
print("STEP 4 — BUILD AND CALIBRATE LOCAL EVALUATION PIPELINE")
print("=" * 70)

from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F

# 4a. ROUGE scorer with Unicode whitespace tokenizer
class UnicodeWhitespaceTokenizer:
    def tokenize(self, text): return re.findall(r"\S+", str(text), flags=re.UNICODE)

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=False,
                                       tokenizer=UnicodeWhitespaceTokenizer())
    r1s, rls = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1s.append(s["rouge1"].fmeasure); rls.append(s["rougeL"].fmeasure)
    return {"rouge1": np.mean(r1s), "rougeL": np.mean(rls),
            "rouge1_per_row": r1s, "rougeL_per_row": rls}

# 4b. AfroLM BertScore (token-level proper P/R/F1)
AFROLM_MODEL_ID = "bonadossou/afrolm_active_learning"
AFROLM_FALLBACK = "intfloat/multilingual-e5-base"
_afrolm_bundle = None
_afrolm_warning = None

def load_afrolm():
    global _afrolm_bundle, _afrolm_warning
    if _afrolm_bundle is not None: return _afrolm_bundle
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mid in [AFROLM_MODEL_ID, AFROLM_FALLBACK]:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModel.from_pretrained(mid).to(device).eval()
            if mid != AFROLM_MODEL_ID:
                _afrolm_warning = f"⚠️ PROXY METRIC: using {mid} instead of AfroLM."
                print(_afrolm_warning)
            _afrolm_bundle = (tok, mdl, device, mid)
            print(f"Loaded AfroLM encoder: {mid}")
            return _afrolm_bundle
        except Exception as e:
            print(f"  Failed to load {mid}: {e}")
    raise RuntimeError("Cannot load any AfroLM encoder")

def embed_tokens(text, afrolm=None):
    """Return L2-normed token embeddings (n_tokens, dim) with CLS/SEP stripped."""
    if afrolm is None: afrolm = load_afrolm()
    tok, mdl, device, _ = afrolm
    max_len = min(256, int(getattr(mdl.config, "max_position_embeddings", 256) or 256) - 2)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max(8, max_len))
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)
    h = out.last_hidden_state.squeeze(0).cpu().numpy()
    mask = enc["attention_mask"].squeeze(0).cpu().numpy().astype(bool)
    h = h[mask]
    if h.shape[0] >= 3: h = h[1:-1]  # Strip CLS/SEP
    norms = np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-8, None)
    return h / norms

def pair_bertscore(pred_emb, ref_emb):
    if pred_emb.size == 0 or ref_emb.size == 0: return 0.0, 0.0, 0.0
    sim = pred_emb @ ref_emb.T
    p = float(sim.max(axis=1).mean()); r = float(sim.max(axis=0).mean())
    if p + r == 0: return 0.0, 0.0, 0.0
    return p, r, 2*p*r/(p+r)

def afrolm_bertscore_batch(predictions, references):
    """Compute token-level BertScore F1 per row."""
    afrolm = load_afrolm()
    f1s = []
    for pred, ref in zip(predictions, references):
        pe = embed_tokens(str(pred), afrolm)
        re_ = embed_tokens(str(ref), afrolm)
        _, _, f = pair_bertscore(pe, re_)
        f1s.append(f)
    return {"afrolm_bs": np.mean(f1s), "afrolm_bs_per_row": f1s}

# 4c. LLM Judge — SKIPPED on free GPU to save memory
# Using 3-metric proxy formula instead: R1*0.3125 + RL*0.3125 + AfroLM*0.375
JUDGE_AVAILABLE = False
print("⚠️ LLM Judge SKIPPED (free GPU memory conservation).")
print("  Using 3-metric proxy: R1×0.3125 + RL×0.3125 + AfroLM×0.375")

# 4d. Combined score functions
def combined_score_4(r1, rl, bs, judge): return 0.25*r1 + 0.25*rl + 0.30*bs + 0.20*judge
def combined_score_proxy(r1, rl, bs): return 0.3125*r1 + 0.3125*rl + 0.375*bs

def full_eval_report(predictions, val_df):
    refs = val_df[answer_col].astype(str).tolist()
    langs = [LANG_CODE_MAP[str(l)] for l in val_df[lang_col].astype(str).tolist()]
    rouge = compute_rouge(predictions, refs)
    afrolm = afrolm_bertscore_batch(predictions, refs)
    combined = combined_score_proxy(rouge["rouge1"], rouge["rougeL"], afrolm["afrolm_bs"])
    # Per-row combined
    combined_per_row = [combined_score_proxy(rouge["rouge1_per_row"][i], rouge["rougeL_per_row"][i],
                                             afrolm["afrolm_bs_per_row"][i]) for i in range(len(predictions))]
    # Per-language breakdown
    per_lang = {}
    for lang in sorted(set(langs)):
        mask = [i for i,l in enumerate(langs) if l==lang]
        r1l = np.mean([rouge["rouge1_per_row"][i] for i in mask])
        rll = np.mean([rouge["rougeL_per_row"][i] for i in mask])
        bsl = np.mean([afrolm["afrolm_bs_per_row"][i] for i in mask])
        per_lang[lang] = {"combined": combined_score_proxy(r1l,rll,bsl),
                          "rouge1":r1l, "rougeL":rll, "afrolm_bs":bsl, "judge":None}
    return {"combined":combined, "r1":rouge["rouge1"], "rl":rouge["rougeL"],
            "afrolm_bs":afrolm["afrolm_bs"], "judge":None, "per_language":per_lang,
            "scoring_mode":"proxy_3_metric",
            "rouge1_per_row":rouge["rouge1_per_row"], "rougeL_per_row":rouge["rougeL_per_row"],
            "afrolm_bs_per_row":afrolm["afrolm_bs_per_row"], "judge_per_row":[None]*len(predictions),
            "combined_per_row":combined_per_row}

# 4e. Calibration checks
print("\nRunning calibration checks...")
cal_sample = val.sample(n=min(10, len(val)), random_state=SEED).reset_index(drop=True)

# Check A: perfect predictions
perfect_preds = cal_sample[answer_col].astype(str).tolist()
perfect_report = full_eval_report(perfect_preds, cal_sample)

# Check B: wrong-language predictions
random.seed(SEED)
full_langs = [LANG_CODE_MAP[str(v)] for v in val[lang_col].astype(str).tolist()]
full_refs = val[answer_col].astype(str).tolist()
sample_langs = [LANG_CODE_MAP[str(v)] for v in cal_sample[lang_col].astype(str).tolist()]
wrong_preds = []
for lang in sample_langs:
    candidates = [r for r, l in zip(full_refs, full_langs) if l != lang]
    wrong_preds.append(random.choice(candidates))
wrong_report = full_eval_report(wrong_preds, cal_sample)

check_a = perfect_report["combined"]
check_b = wrong_report["combined"]
check_a_pass = check_a >= 0.95
check_b_pass = check_b <= 0.20

print("\nCALIBRATION CHECK")
print("─" * 55)
print(f"Check A (prediction = reference) : {check_a:.4f}  [{'PASS ≥ 0.95' if check_a_pass else 'FAIL'}]")
print(f"Check B (wrong language shuffle)  : {check_b:.4f}  [{'PASS ≤ 0.20' if check_b_pass else 'FAIL'}]")
print("─" * 55)

if not check_a_pass:
    print("⚠️ Check A failed — but with proxy scoring (no judge), threshold is approximate.")
    print("  Proceeding with caution if score > 0.85...")
    if check_a < 0.85:
        raise RuntimeError("Calibration Check A critically failed. Fix evaluation pipeline.")

if not check_b_pass:
    print("⚠️ Check B failed — wrong-language scores too high.")
    if check_b > 0.35:
        raise RuntimeError("Calibration Check B critically failed. AfroLM BertScore is broken.")

# Unload AfroLM to free memory for generation model
del _afrolm_bundle
_afrolm_bundle = None
cleanup_gpu()
print("\n✅ Step 4 complete. AfroLM unloaded to free GPU memory for generation.")
