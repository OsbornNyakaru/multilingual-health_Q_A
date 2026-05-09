"""Multi-model candidate selection for ensembling.

Given candidate answers from multiple fine-tuned models (e.g. Aya-Expanse QLoRA
+ Gemma-2 QLoRA + UlizaLlama on Swahili rows), pick the final answer per row.

Two selection modes:
    - ``afrolm_reranker``: reuse AfroLMReranker against a retrieval pool.
    - ``language_routed``: send Swahili to UlizaLlama, others to Aya (cheap).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    mode: str = "afrolm_reranker"   # "afrolm_reranker" | "language_routed"
    language_routes: dict[str, str] | None = None  # lang -> model_name


def combine(
    per_model_candidates: dict[str, list[str]],
    questions: list[str],
    languages: list[str],
    config: EnsembleConfig,
    reranker=None,
) -> list[str]:
    """Pick one final answer per row across ``per_model_candidates`` models.

    Args:
        per_model_candidates: {model_name: list_of_predictions}, each list same length.
        questions / languages: parallel lists of the input rows.
        config: ensemble strategy.
        reranker: AfroLMReranker instance required when mode=="afrolm_reranker".

    Returns:
        list of final predictions, one per row.
    """
    model_names = list(per_model_candidates.keys())
    n = len(questions)
    for name, preds in per_model_candidates.items():
        if len(preds) != n:
            raise ValueError(f"model '{name}' produced {len(preds)} predictions, expected {n}")

    if config.mode == "language_routed":
        if not config.language_routes:
            raise ValueError("language_routed mode needs config.language_routes.")
        out = []
        for i in range(n):
            target = config.language_routes.get(languages[i]) or model_names[0]
            if target not in per_model_candidates:
                target = model_names[0]
            out.append(per_model_candidates[target][i])
        return out

    if config.mode == "afrolm_reranker":
        if reranker is None:
            raise ValueError("afrolm_reranker mode needs a reranker instance.")
        out = []
        for i in range(n):
            cands = [per_model_candidates[name][i] for name in model_names]
            out.append(reranker.rerank(questions[i], languages[i], cands))
        return out

    raise ValueError(f"unknown ensemble mode: {config.mode}")
