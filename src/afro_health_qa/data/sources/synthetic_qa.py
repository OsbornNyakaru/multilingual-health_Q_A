"""Synthetic QA from fact sheets via an open-source generator model.

STUB: generate additional (question, answer) pairs from the WHO fact sheets
using CohereForAI/aya-expanse-8b (local, no API). Closed APIs (OpenAI,
Anthropic, Gemini) are forbidden by Zindi rules — this module must refuse
to run with any closed backend.
"""

from __future__ import annotations

import pandas as pd

from afro_health_qa.data.augment import AugmentationSource, register


def _load(config: dict) -> pd.DataFrame:
    if not config.get("open_models_only", True):
        raise ValueError(
            "synthetic_qa.open_models_only=False is forbidden by Zindi rules. "
            "Closed APIs may not generate competition training data."
        )
    # TODO: implement once who_factsheets data lands.
    #   1. Load who_factsheets parquet.
    #   2. For each fact sheet paragraph, prompt the generator with
    #      "Generate 3 diverse user questions a patient might ask about this content,
    #       and answer each in <language>."
    #   3. Parse and validate; run LID on generated answers; discard any that fail.
    raise NotImplementedError(
        "synthetic_qa is a stub — enable once fact-sheet content is curated."
    )


register(
    AugmentationSource(
        name="synthetic_qa",
        languages=("swa", "lug", "aka", "amh"),
        loader=_load,
    )
)
