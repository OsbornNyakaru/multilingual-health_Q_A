"""Weighted combiner matching Zindi's scoring formula."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Zindi weights — see docs/SCORING.md. DO NOT change without a very good reason.
W_R1 = 0.25
W_RL = 0.25
W_AFROLM_BS = 0.30
W_JUDGE = 0.20


@dataclass
class CombinedScore:
    rouge1: float
    rouge_l: float
    afrolm_bertscore: float
    llm_judge: float
    combined: float
    n: int


def combined_score(r1: float, rl: float, afrolm_bs: float, judge: float) -> float:
    """The exact weighted mean Zindi computes.

    Args:
        r1: ROUGE-1 F1 (0-1)
        rl: ROUGE-L F1 (0-1)
        afrolm_bs: AfroLM-BertScore F1 (0-1)
        judge: LLM-judge score mapped to (0-1)

    Returns:
        Combined score in (0-1).
    """
    return W_R1 * r1 + W_RL * rl + W_AFROLM_BS * afrolm_bs + W_JUDGE * judge


def report(score: CombinedScore) -> str:
    """Human-readable breakdown — shows which sub-metric is dragging."""
    lines = [
        f"Combined score: {score.combined:.4f}  (n={score.n})",
        f"  ROUGE-1 F1        ×0.25 = {score.rouge1:.4f}  → {W_R1 * score.rouge1:.4f}",
        f"  ROUGE-L F1        ×0.25 = {score.rouge_l:.4f}  → {W_RL * score.rouge_l:.4f}",
        f"  AfroLM-BertScore  ×0.30 = {score.afrolm_bertscore:.4f}  → {W_AFROLM_BS * score.afrolm_bertscore:.4f}",
        f"  LLM-judge         ×0.20 = {score.llm_judge:.4f}  → {W_JUDGE * score.llm_judge:.4f}",
    ]
    return "\n".join(lines)


def score_predictions(
    predictions: list[str],
    references: list[str],
    questions: list[str],
    languages: list[str],
    judge_sample_n: int | None = 50,
    skip_judge: bool = False,
) -> CombinedScore:
    """Run all four sub-metrics and return a CombinedScore."""
    from afro_health_qa.evaluation.afrolm_bertscore import score_afrolm_bertscore
    from afro_health_qa.evaluation.rouge import score_rouge

    rouge = score_rouge(predictions, references)
    afrolm = score_afrolm_bertscore(predictions, references)

    judge_val = 0.0
    if not skip_judge:
        from afro_health_qa.evaluation.judge import score_llm_judge

        judge_res = score_llm_judge(
            questions=questions,
            predictions=predictions,
            references=references,
            languages=languages,
            sample_n=judge_sample_n,
        )
        judge_val = judge_res.mean_score

    combined = combined_score(rouge.rouge1, rouge.rouge_l, afrolm.f1, judge_val)
    return CombinedScore(
        rouge1=rouge.rouge1,
        rouge_l=rouge.rouge_l,
        afrolm_bertscore=afrolm.f1,
        llm_judge=judge_val,
        combined=combined,
        n=rouge.n,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score a predictions CSV against the val split.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references-csv", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--pred-col", default="TargetBert")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--split",
        choices=["val", "heldout"],
        default="val",
        help="Which processed split to score against.",
    )
    args = parser.parse_args()

    import pandas as pd

    # Switch reference file based on --split unless user overrode.
    refs_path = args.references_csv
    if args.split and refs_path == Path("data/processed/val.csv"):
        refs_path = Path(f"data/processed/{args.split}.csv")

    preds = pd.read_csv(args.predictions, dtype=str).fillna("")
    refs = pd.read_csv(refs_path, dtype=str).fillna("")
    merged = refs.merge(preds[[args.id_col, args.pred_col]], on=args.id_col, how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlap between predictions and references.")

    result = score_predictions(
        predictions=merged[args.pred_col].tolist(),
        references=merged["Response"].tolist(),
        questions=merged["Question"].tolist(),
        languages=merged["Language"].tolist(),
        skip_judge=args.skip_judge,
    )
    print(report(result))


if __name__ == "__main__":
    _cli()
