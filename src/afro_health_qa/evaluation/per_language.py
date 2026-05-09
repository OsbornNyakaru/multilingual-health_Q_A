"""Per-language breakdown reporter.

Given predictions + references + languages, produces a table:
    lang | n | rouge1 | rougeL | afrolm_bs | judge_sampled | combined

The max-min combined gap across languages is a primary selection criterion in
scripts/select_final.py — a model that's great in Swahili but collapses in
Amharic is risky under the private-LB's 70% weight.
"""

from __future__ import annotations

import pandas as pd

from afro_health_qa.evaluation.afrolm_bertscore import score_afrolm_bertscore
from afro_health_qa.evaluation.combined import combined_score
from afro_health_qa.evaluation.rouge import score_rouge


def per_language_breakdown(
    predictions: list[str],
    references: list[str],
    languages: list[str],
    judge_scores: list[float] | None = None,
) -> pd.DataFrame:
    """Compute per-language ROUGE-1, ROUGE-L, AfroLM-BS, and combined.

    Args:
        predictions / references / languages: parallel lists.
        judge_scores: optional per-row judge scores in [0, 1]. If None, the
            judge column is NaN and combined is computed with 0 — a lower bound.

    Returns:
        DataFrame sorted by language code.
    """
    if not (len(predictions) == len(references) == len(languages)):
        raise ValueError("predictions, references, languages must have the same length.")

    df = pd.DataFrame(
        {
            "prediction": predictions,
            "reference": references,
            "language": languages,
            "judge": judge_scores if judge_scores is not None else [float("nan")] * len(predictions),
        }
    )

    rows = []
    for lang, group in df.groupby("language", sort=True):
        preds = group["prediction"].tolist()
        refs = group["reference"].tolist()
        if not preds:
            continue
        rouge = score_rouge(preds, refs)
        afrolm = score_afrolm_bertscore(preds, refs)
        judge_val = float(group["judge"].dropna().mean()) if group["judge"].notna().any() else 0.0
        rows.append(
            {
                "language": lang,
                "n": len(preds),
                "rouge1": rouge.rouge1,
                "rougeL": rouge.rouge_l,
                "afrolm_bs": afrolm.f1,
                "judge": judge_val,
                "combined": combined_score(rouge.rouge1, rouge.rouge_l, afrolm.f1, judge_val),
            }
        )

    return pd.DataFrame(rows).set_index("language").round(4)


def language_gap(breakdown: pd.DataFrame, col: str = "combined") -> float:
    """Max-min gap across languages. Lower = more robust."""
    if breakdown.empty:
        return 0.0
    return float(breakdown[col].max() - breakdown[col].min())
