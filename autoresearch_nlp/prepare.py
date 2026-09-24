"""prepare.py — FROZEN evaluation harness for generative-NLP competitions.

Mirrors the role of prepare.py in karpathy/autoresearch: fixed constants,
one-time data prep, and the runtime/evaluation utilities. The agent NEVER
edits this file. It is what guarantees every experiment in train.py is scored
the same way, so results are comparable across the whole run.

Per-competition setup is confined to the CONFIG block below. Fill it in ONCE on
day 1 (PREFLIGHT step A/B), commit it, then freeze it.

What this file provides:
  * load_raw()        — read Train/Test/Sample CSVs with schema mapping
  * make_splits()     — carve a held-out slice that is NEVER trained on
  * score()           — the EXACT leaderboard metric (ROUGE + optional judge)
  * score_per_subset()— the per-language/locale breakdown (your real signal)
  * report()          — greppable block that train.py prints each run

Scoring uses Google's `rouge-score` library (the same one Zindi/most hosts use).
ROUGE is computed on whitespace tokens with no stemming by default, which is the
correct, locale-agnostic choice for multilingual text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — the ONLY part you edit per competition. Fill once, then freeze.
# ════════════════════════════════════════════════════════════════════════════

# Instance config: read the existing competition data one level up; keep this
# instance's processed splits + submissions local to the autoresearch_nlp/ folder.
DATA_DIR = Path("../data/raw")
CACHE_DIR = Path("data/processed")

# Column schema in the raw CSVs.
ID_COL = "ID"
INPUT_COL = "input"        # the question / source text
OUTPUT_COL = "output"      # the reference answer (train/val only)
SUBSET_COL = "subset"      # language or locale group; "" if the comp has none

# Submission schema — mirror SampleSubmission.csv EXACTLY.
# For multi-metric comps the target columns hold IDENTICAL values per row.
SUBMISSION_COLUMNS = ("ID", "TargetRLF1", "TargetR1F1", "TargetLLM")

# Leaderboard metric weights. MUST match the competition page. Set once.
# Any metric the host does NOT score on the LB gets weight 0.0.
METRIC_WEIGHTS = {
    "rouge1_f1": 0.37,
    "rougeL_f1": 0.37,
    "judge": 0.26,
}

# Held-out slice: fraction of train reserved for honest local scoring.
# This slice is NEVER used for training/few-shot pools. Different seed than any
# CV you might run inside train.py.
HELDOUT_FRACTION = 0.07
HELDOUT_SEED = 1234

# ROUGE settings. use_stemmer=False is correct for non-English text.
ROUGE_USE_STEMMER = False

# ════════════════════════════════════════════════════════════════════════════
# Data loading + splitting
# ════════════════════════════════════════════════════════════════════════════


def load_raw() -> dict[str, pd.DataFrame]:
    """Load the raw competition CSVs as strings (no dtype surprises)."""
    out = {}
    for name in ("Train", "Test", "SampleSubmission", "Val"):
        p = DATA_DIR / f"{name}.csv"
        if p.exists():
            out[name.lower()] = pd.read_csv(p, dtype=str).fillna("")
    if "train" not in out or "test" not in out:
        raise FileNotFoundError(
            f"Need at least {DATA_DIR}/Train.csv and Test.csv. Found: {list(out)}"
        )
    return out


def make_splits() -> dict[str, pd.DataFrame]:
    """Carve a stratified held-out slice from Train and persist it.

    Writes:
        data/processed/work_train.csv  — train ON this (plus Val if present)
        data/processed/held_out.csv    — score ON this; NEVER train on it
    Returns the two frames.
    """
    raw = load_raw()
    train = raw["train"].copy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    strat = train[SUBSET_COL] if SUBSET_COL and SUBSET_COL in train.columns else None
    held_parts, work_parts = [], []
    if strat is not None:
        for _, grp in train.groupby(strat):
            g = grp.sample(frac=1.0, random_state=HELDOUT_SEED)
            n_held = max(1, int(round(len(g) * HELDOUT_FRACTION)))
            held_parts.append(g.iloc[:n_held])
            work_parts.append(g.iloc[n_held:])
        held = pd.concat(held_parts).sort_index()
        work = pd.concat(work_parts).sort_index()
    else:
        g = train.sample(frac=1.0, random_state=HELDOUT_SEED)
        n_held = max(1, int(round(len(g) * HELDOUT_FRACTION)))
        held = g.iloc[:n_held].sort_index()
        work = g.iloc[n_held:].sort_index()

    work.to_csv(CACHE_DIR / "work_train.csv", index=False, encoding="utf-8")
    held.to_csv(CACHE_DIR / "held_out.csv", index=False, encoding="utf-8")
    return {"work_train": work, "held_out": held}


# ════════════════════════════════════════════════════════════════════════════
# Scoring — the frozen leaderboard metric
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Score:
    rouge1_f1: float
    rougeL_f1: float
    judge: float
    combined: float
    n: int

    def greppable(self) -> str:
        return (
            f"combined: {self.combined:.6f}\n"
            f"rouge1_f1: {self.rouge1_f1:.6f}\n"
            f"rougeL_f1: {self.rougeL_f1:.6f}\n"
            f"judge: {self.judge:.6f}\n"
            f"n: {self.n}"
        )


class _WhitespaceTokenizer:
    """Whitespace tokenizer — matches the official starter notebook's ROUGE.

    The default rouge-score tokenizer can strip Amharic Ge'ez script to almost
    nothing; whitespace tokenization is language-agnostic. The starter notebook
    scores this way, so we match it to keep local scores aligned with the LB.
    """

    def tokenize(self, text):
        return str(text).strip().split() if text else []


def _rouge_scorer():
    from rouge_score import rouge_scorer  # lazy import so import-only is cheap

    return rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], tokenizer=_WhitespaceTokenizer(), use_stemmer=ROUGE_USE_STEMMER
    )


def score(
    predictions: list[str],
    references: list[str],
    judge_scores: list[float] | None = None,
) -> Score:
    """Compute the exact leaderboard combined score.

    judge_scores: optional per-row LLM-judge values already normalised to [0,1].
    Pass None during fast local iteration — the judge weight is then treated as
    0 and the combined score reflects only the (replicable) ROUGE component,
    which is the bulk of the metric and the right thing to optimise locally.
    """
    if len(predictions) != len(references):
        raise ValueError(f"len(pred)={len(predictions)} != len(ref)={len(references)}")
    scorer = _rouge_scorer()
    r1 = rl = 0.0
    for p, r in zip(predictions, references):
        s = scorer.score(str(r), str(p))
        r1 += s["rouge1"].fmeasure
        rl += s["rougeL"].fmeasure
    n = max(1, len(predictions))
    r1 /= n
    rl /= n
    judge = float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0

    combined = (
        METRIC_WEIGHTS.get("rouge1_f1", 0.0) * r1
        + METRIC_WEIGHTS.get("rougeL_f1", 0.0) * rl
        + METRIC_WEIGHTS.get("judge", 0.0) * judge
    )
    return Score(rouge1_f1=r1, rougeL_f1=rl, judge=judge, combined=combined, n=len(predictions))


def score_per_subset(
    df: pd.DataFrame,
    pred_col: str,
    ref_col: str = OUTPUT_COL,
    subset_col: str = SUBSET_COL,
) -> pd.DataFrame:
    """Per-subset ROUGE breakdown — your highest-value diagnostic.

    Returns a DataFrame indexed by subset with rouge1_f1, rougeL_f1, combined
    (judge omitted), and n. The worst subsets are your biggest point reservoir.
    """
    rows = []
    groups = [("ALL", df)]
    if subset_col and subset_col in df.columns:
        groups += list(df.groupby(subset_col))
    for name, g in groups:
        s = score(g[pred_col].tolist(), g[ref_col].tolist())
        rows.append(
            {
                "subset": name,
                "n": s.n,
                "rouge1_f1": round(s.rouge1_f1, 4),
                "rougeL_f1": round(s.rougeL_f1, 4),
                "combined_no_judge": round(s.combined, 4),
            }
        )
    return pd.DataFrame(rows).set_index("subset")


def report(name: str, score_obj: Score, extra: str = "") -> str:
    """The block train.py prints at the end of every run (greppable)."""
    lines = ["---", f"name: {name}", score_obj.greppable()]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Submission helpers (schema enforcement)
# ════════════════════════════════════════════════════════════════════════════


def build_submission(ids: list[str], answers: list[str]) -> pd.DataFrame:
    """Build a submission frame matching SUBMISSION_COLUMNS, target cols identical."""
    if len(ids) != len(answers):
        raise ValueError("ids/answers length mismatch")
    answers = ["" if a is None else str(a) for a in answers]
    df = pd.DataFrame({SUBMISSION_COLUMNS[0]: [str(i) for i in ids]})
    for col in SUBMISSION_COLUMNS[1:]:
        df[col] = answers
    return df[list(SUBMISSION_COLUMNS)]


def validate_submission(df: pd.DataFrame, expected_ids: list[str] | None = None) -> None:
    """Hard schema checks. Raises on any violation."""
    if tuple(df.columns) != tuple(SUBMISSION_COLUMNS):
        raise ValueError(f"columns {tuple(df.columns)} != {tuple(SUBMISSION_COLUMNS)}")
    base = df[SUBMISSION_COLUMNS[1]].astype(str)
    for col in SUBMISSION_COLUMNS[2:]:
        if not df[col].astype(str).equals(base):
            raise ValueError(f"target column {col} differs from {SUBMISSION_COLUMNS[1]}")
    if df[SUBMISSION_COLUMNS[0]].duplicated().any():
        raise ValueError("duplicate IDs")
    if (df[SUBMISSION_COLUMNS[0]].astype(str) == "").any():
        raise ValueError("empty IDs")
    if expected_ids is not None and set(df[SUBMISSION_COLUMNS[0]]) != set(map(str, expected_ids)):
        raise ValueError("submission IDs do not match the expected test IDs")


def _fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════
# CLI: one-time data prep
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    raw = load_raw()
    print("Loaded:", {k: len(v) for k, v in raw.items()})
    splits = make_splits()
    print(f"work_train: {len(splits['work_train'])}  held_out: {len(splits['held_out'])}")
    if SUBSET_COL and SUBSET_COL in splits["held_out"].columns:
        print("\nHeld-out per-subset counts:")
        print(splits["held_out"][SUBSET_COL].value_counts().to_string())
    print("\nConfig frozen:", asdict(Score(0, 0, 0, 0, 0)) and METRIC_WEIGHTS)
    print("prepare.py OK — harness ready, splits written to", CACHE_DIR)


if __name__ == "__main__":
    main()
