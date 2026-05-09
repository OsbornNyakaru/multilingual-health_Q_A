"""ROUGE-1 and ROUGE-L F1 using the ``rouge_score`` library.

We use RougeScorer directly (not ``evaluate``) because Zindi's scorer
uses the same library and our local scores need to line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class RougeResult:
    rouge1: float
    rouge_l: float
    n: int


def score_rouge(predictions: list[str], references: list[str]) -> RougeResult:
    """Compute per-example ROUGE-1 F1 and ROUGE-L F1, return the averages.

    Uses rouge_score's default stemmer=True for English behaviour. Stemming
    has no effect on Amharic script; for Latin-script African languages
    it won't match real morphology but it's what Zindi's reference implementation
    does, so we match exactly.

    Args:
        predictions: list of generated strings.
        references: list of gold strings, same length as predictions.

    Returns:
        RougeResult with averaged rouge1 F1, rougeL F1, and n.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must match length: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return RougeResult(rouge1=0.0, rouge_l=0.0, n=0)

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    r1_scores = []
    rl_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return RougeResult(
        rouge1=float(mean(r1_scores)),
        rouge_l=float(mean(rl_scores)),
        n=len(predictions),
    )


def score_rouge_per_example(
    predictions: list[str], references: list[str]
) -> list[tuple[float, float]]:
    """Return per-row (rouge1_f1, rougeL_f1). Useful for per-language breakdowns."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    out = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        out.append((scores["rouge1"].fmeasure, scores["rougeL"].fmeasure))
    return out
