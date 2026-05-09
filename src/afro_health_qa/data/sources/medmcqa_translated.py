"""MedMCQA back-translated via NLLB-200 — cross-lingual medical knowledge injection.

STUB: translate MedMCQA Q/A pairs with facebook/nllb-200-3.3B to the four
target languages. Cache translations in data/external/medmcqa_translations/
keyed by (row_id, target_lang).
"""

from __future__ import annotations

import pandas as pd

from afro_health_qa.data.augment import AugmentationSource, register


def _load(config: dict) -> pd.DataFrame:
    # TODO: Implement the NLLB translation pipeline.
    #   1. Load openlifescienceai/medmcqa train split.
    #   2. For each row, translate Q + chosen answer into each target lang.
    #   3. Filter by LID confidence (reject any row where detect_lang(Q_translated)
    #      disagrees with the requested target).
    #   4. Write to data/external/medmcqa_translations/<lang>.parquet and read back.
    raise NotImplementedError(
        "medmcqa_translated is a stub — implement the NLLB translation loop."
    )


register(
    AugmentationSource(
        name="medmcqa_translated",
        languages=("swa", "lug", "aka", "amh"),
        loader=_load,
    )
)
