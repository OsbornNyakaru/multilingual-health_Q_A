"""Local replica of the leaderboard metric, broken down by ``subset``.

    total      = 0.37 * ROUGE-1 F1 + 0.37 * ROUGE-L F1 + 0.26 * judge   (NaN if no judge)
    rouge_only = 0.37 * ROUGE-1 F1 + 0.37 * ROUGE-L F1                  (judge = 0, lower bound)

Usage (CPU, no judge):

    python -m afro_health_qa.evaluation.scorer \\
        --predictions submissions/my_preds.csv --references data/processed/held_out.csv

Predictions CSV: ``ID`` plus either one answer column (``--pred-col``, or
auto-detected: prediction/answer/response/generated) or a 4-column Zindi
submission (``TargetR1F1`` → ROUGE-1, ``TargetRLF1`` → ROUGE-L, ``TargetLLM`` →
judge, exactly as the platform scores it). References CSV: ``ID, input, output,
subset`` (Train/Val/held_out). Add ``--judge vllm`` on molab for the judge term,
and ``--compare-tokenizers`` to see how much tokenization moves each subset.
See vault/facts/metric-replica.md.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from afro_health_qa.evaluation.combined import W_JUDGE, W_R1, W_RL
from afro_health_qa.evaluation.judge import DEFAULT_JUDGE_MODEL_ID, JudgeFn, make_judge, run_judge
from afro_health_qa.evaluation.rouge import DEFAULT_TOKENIZER, TOKENIZER_CHOICES, score_rouge_per_example

SUBMISSION_COLS = ("TargetR1F1", "TargetRLF1", "TargetLLM")
_PRED_COL_CANDIDATES = ("prediction", "answer", "response", "generated", "pred")


def subset_from_id(row_id: str) -> str:
    """``ID_TS_Amh_Eth_ABC123`` -> ``Amh_Eth``."""
    parts = str(row_id).split("_")
    return f"{parts[2]}_{parts[3]}" if len(parts) >= 5 else "UNKNOWN"


def load_predictions(path: str | Path, pred_col: str | None = None, id_col: str = "ID") -> pd.DataFrame:
    """Return ``ID, pred_r1, pred_rl, pred_llm`` (the three are equal unless it's a submission file)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if pred_col is None and all(c in df.columns for c in SUBMISSION_COLS):
        out = df[[id_col, *SUBMISSION_COLS]].rename(
            columns={"TargetR1F1": "pred_r1", "TargetRLF1": "pred_rl", "TargetLLM": "pred_llm"}
        )
    else:
        if pred_col is None:
            lower = {c.lower(): c for c in df.columns}
            found = [lower[c] for c in _PRED_COL_CANDIDATES if c in lower]
            if not found:
                raise ValueError(f"cannot find a prediction column in {list(df.columns)}; pass --pred-col")
            pred_col = found[0]
        out = pd.DataFrame({id_col: df[id_col], "pred_r1": df[pred_col], "pred_rl": df[pred_col], "pred_llm": df[pred_col]})
    if out[id_col].duplicated().any():
        raise ValueError(f"{int(out[id_col].duplicated().sum())} duplicate IDs in predictions")
    return out.rename(columns={id_col: "ID"})


def load_references(path: str | Path, id_col: str = "ID") -> pd.DataFrame:
    """Return ``ID, question, reference, subset`` from a Train/Val/held_out-style CSV."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "output" not in df.columns:
        raise ValueError(f"references file {path} has no 'output' column (Test.csv has no answers)")
    out = pd.DataFrame(
        {
            "ID": df[id_col],
            "question": df["input"] if "input" in df.columns else "",
            "reference": df["output"],
            "subset": df["subset"] if "subset" in df.columns else df[id_col].map(subset_from_id),
        }
    )
    return out


def join(preds: pd.DataFrame, refs: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Inner-join on ID. Missing predictions are reported (and raise if ``strict``)."""
    merged = refs.merge(preds, on="ID", how="inner")
    missing = len(refs) - len(merged)
    if len(merged) == 0:
        raise RuntimeError("no overlapping IDs between predictions and references")
    if missing:
        msg = f"{missing} of {len(refs)} reference rows have no prediction; scoring the {len(merged)} overlap"
        if strict:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
    return merged.reset_index(drop=True)


def _judge_sample(df: pd.DataFrame, max_per_subset: int | None, seed: int) -> pd.Index:
    if max_per_subset is None:
        return df.index
    idx: list = []
    for _, g in df.groupby("subset", sort=True):
        idx.extend(g.sample(n=min(len(g), max_per_subset), random_state=seed).index)
    return pd.Index(idx)


def score_rows(
    df: pd.DataFrame,
    tokenizer: str = DEFAULT_TOKENIZER,
    judge_fn: JudgeFn | None = None,
    judge_max_per_subset: int | None = None,
    seed: int = 1234,
) -> pd.DataFrame:
    """Per-row ``rouge1``, ``rougeL``, ``judge`` (NaN when not judged) on a joined frame."""
    df = df.copy()
    df["rouge1"] = [r1 for r1, _ in score_rouge_per_example(df["pred_r1"].tolist(), df["reference"].tolist(), tokenizer)]
    df["rougeL"] = [rl for _, rl in score_rouge_per_example(df["pred_rl"].tolist(), df["reference"].tolist(), tokenizer)]
    df["judge"] = float("nan")
    if judge_fn is not None:
        idx = _judge_sample(df, judge_max_per_subset, seed)
        sub = df.loc[idx]
        res = run_judge(judge_fn, sub["question"].tolist(), sub["reference"].tolist(), sub["pred_llm"].tolist(), sub["subset"].tolist())
        df.loc[idx, "judge"] = res.per_row
        df.attrs["judge_parse_failures"] = res.parse_failures
        df.attrs["judge_model_id"] = res.model_id
    return df


def _summ(g: pd.DataFrame) -> dict:
    judged = g["judge"].dropna()
    judge = float(judged.mean()) if len(judged) else float("nan")
    r1, rl = float(g["rouge1"].mean()), float(g["rougeL"].mean())
    rouge_only = W_R1 * r1 + W_RL * rl
    return {
        "n": len(g),
        "rouge1": r1,
        "rougeL": rl,
        "judge": judge,
        "judge_n": len(judged),
        "rouge_only": rouge_only,
        "total": rouge_only + W_JUDGE * judge if not math.isnan(judge) else float("nan"),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-subset table plus an ``ALL`` row (row-weighted, like the leaderboard).

    If the judge was sampled per subset, ALL's judge is the subset-size-weighted
    mean of subset judge means, so small subsets are not over-weighted.
    """
    table = pd.DataFrame({s: _summ(g) for s, g in rows.groupby("subset", sort=True)}).T
    overall = _summ(rows)
    per = table[table["judge_n"] > 0]
    if len(per) and per["judge_n"].sum() < len(rows):
        j = float((per["judge"] * per["n"]).sum() / per["n"].sum())
        overall["judge"] = j
        overall["total"] = overall["rouge_only"] + W_JUDGE * j
    table.loc["ALL"] = overall
    table["n"] = table["n"].astype(int)
    table["judge_n"] = table["judge_n"].astype(int)
    return table[["n", "rouge1", "rougeL", "judge", "judge_n", "rouge_only", "total"]]


def compare_tokenizers(df: pd.DataFrame, tokenizers: Sequence[str] = TOKENIZER_CHOICES) -> pd.DataFrame:
    """ROUGE-only partial total per subset under each tokenizer (columns = tokenizer)."""
    cols = {}
    for tok in tokenizers:
        cols[tok] = summarize(score_rows(df, tokenizer=tok))["rouge_only"]
    return pd.DataFrame(cols)


def evaluate(
    predictions: str | Path,
    references: str | Path,
    pred_col: str | None = None,
    tokenizer: str = DEFAULT_TOKENIZER,
    judge_fn: JudgeFn | None = None,
    judge_max_per_subset: int | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, join, score. Returns ``(per_subset_table, per_row_frame)``."""
    merged = join(load_predictions(predictions, pred_col), load_references(references), strict=strict)
    rows = score_rows(merged, tokenizer=tokenizer, judge_fn=judge_fn, judge_max_per_subset=judge_max_per_subset)
    return summarize(rows), rows


def format_table(table: pd.DataFrame) -> str:
    return table.to_string(float_format=lambda x: "  nan" if math.isnan(x) else f"{x:.4f}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Score predictions with the local leaderboard-metric replica, per subset.")
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--references", required=True, type=Path, help="CSV with ID,input,output,subset (Val.csv, held_out.csv)")
    p.add_argument("--pred-col", default=None, help="answer column; default auto-detects / uses 4-column submission")
    p.add_argument("--tokenizer", choices=TOKENIZER_CHOICES, default=DEFAULT_TOKENIZER)
    p.add_argument("--judge", choices=["none", "transformers", "vllm"], default="none")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL_ID)
    p.add_argument("--judge-max-per-subset", type=int, default=None)
    p.add_argument("--compare-tokenizers", action="store_true")
    p.add_argument("--strict", action="store_true", help="fail if any reference row lacks a prediction")
    p.add_argument("--out", type=Path, default=None, help="save the per-subset table as CSV")
    p.add_argument("--rows-out", type=Path, default=None, help="save per-row scores as CSV")
    args = p.parse_args()

    judge_fn = make_judge(args.judge, args.judge_model)
    table, rows = evaluate(
        args.predictions, args.references, args.pred_col, args.tokenizer, judge_fn, args.judge_max_per_subset, args.strict
    )
    print(f"predictions: {args.predictions}\nreferences:  {args.references}\ntokenizer:   {args.tokenizer}")
    print(f"judge:       {rows.attrs.get('judge_model_id') or 'none (judge=NaN; total=NaN; rouge_only = judge-free lower bound)'}")
    if rows.attrs.get("judge_parse_failures"):
        print(f"[warn] judge parse failures: {rows.attrs['judge_parse_failures']} (excluded as NaN)")
    print(f"weights:     R1 {W_R1} / RL {W_RL} / judge {W_JUDGE}\n")
    print(format_table(table))
    if args.compare_tokenizers:
        merged = join(load_predictions(args.predictions, args.pred_col), load_references(args.references))
        print("\nROUGE-only partial total by tokenizer:")
        print(format_table(compare_tokenizers(merged)))
    if args.out:
        table.to_csv(args.out, index_label="subset")
        print(f"\n[write] {args.out}")
    if args.rows_out:
        rows.to_csv(args.rows_out, index=False)
        print(f"[write] {args.rows_out}")


if __name__ == "__main__":
    _cli()
