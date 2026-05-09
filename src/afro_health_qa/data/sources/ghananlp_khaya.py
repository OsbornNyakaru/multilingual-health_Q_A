"""GhanaNLP Khaya corpus loader — Akan/Twi augmentation.

STUB: wire once snapshot selected (ghananlp/khaya on HF Hub). Must return
ID, Question, Language, Response.
"""

from __future__ import annotations

import pandas as pd

from afro_health_qa.data.augment import AugmentationSource, register


def _load(config: dict) -> pd.DataFrame:
    raise NotImplementedError(
        "ghananlp_khaya is a stub — implement once Khaya snapshot is selected."
    )


register(
    AugmentationSource(
        name="ghananlp_khaya",
        languages=("aka",),
        loader=_load,
    )
)
