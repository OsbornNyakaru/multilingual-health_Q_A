"""Weighted combiner matching Zindi's leaderboard formula.

Leaderboard = 0.37 * ROUGE-1 F1 + 0.37 * ROUGE-L F1 + 0.26 * LLM-judge (normalized).
Source of truth: vault/facts/F-001-competition-metric.md and
autoresearch_nlp/prepare.py. The old 0.25/0.25/0.30 AfroLM/0.20 split was a
pre-release guess; AfroLM BertScore is NOT on the leaderboard (weight 0, it is a
diagnostic only). DO NOT change these without new evidence from the host.

The per-subset scorer + CLI lives in ``afro_health_qa.evaluation.scorer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

W_R1 = 0.37
W_RL = 0.37
W_JUDGE = 0.26
W_AFROLM_BS = 0.0  # host-side secondary check only; never on the leaderboard

WEIGHTS = {"rouge1": W_R1, "rougeL": W_RL, "judge": W_JUDGE}


@dataclass
class CombinedScore:
    rouge1: float
    rouge_l: float
    llm_judge: float  # NaN when no judge was run
    combined: float  # NaN when no judge was run
    rouge_only: float
    n: int


def rouge_only_score(r1: float, rl: float) -> float:
    """Partial total with the judge term set to 0 — a strict lower bound on the LB score."""
    return W_R1 * r1 + W_RL * rl


def combined_score(r1: float, rl: float, afrolm_bs: float = 0.0, judge: float = float("nan")) -> float:
    """The weighted sum Zindi computes. Returns NaN if ``judge`` is NaN/None.

    ``afrolm_bs`` is accepted for backward compatibility (per_language.py) but
    carries weight 0.
    """
    if judge is None or (isinstance(judge, float) and math.isnan(judge)):
        return float("nan")
    return W_R1 * r1 + W_RL * rl + W_AFROLM_BS * afrolm_bs + W_JUDGE * judge


def report(score: CombinedScore) -> str:
    """Human-readable breakdown — shows which sub-metric is dragging."""
    judge_txt = "not run" if math.isnan(score.llm_judge) else f"{score.llm_judge:.4f}  → {W_JUDGE * score.llm_judge:.4f}"
    total_txt = "n/a (no judge)" if math.isnan(score.combined) else f"{score.combined:.4f}"
    lines = [
        f"Leaderboard total: {total_txt}   ROUGE-only partial (judge=0): {score.rouge_only:.4f}  (n={score.n})",
        f"  ROUGE-1 F1 ×{W_R1} = {score.rouge1:.4f}  → {W_R1 * score.rouge1:.4f}",
        f"  ROUGE-L F1 ×{W_RL} = {score.rouge_l:.4f}  → {W_RL * score.rouge_l:.4f}",
        f"  LLM-judge  ×{W_JUDGE} = {judge_txt}",
    ]
    return "\n".join(lines)
