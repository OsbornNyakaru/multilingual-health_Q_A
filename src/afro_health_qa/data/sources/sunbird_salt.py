"""SunBird AI SALT corpus loader — Luganda augmentation.

STUB: reference wiring for SALT. Hook up once we know which SALT snapshot is
available (sunbirdai/salt on HF Hub). The loader must return a DataFrame with
columns ID, Question, Language, Response.
"""

from __future__ import annotations

import pandas as pd

from afro_health_qa.data.augment import AugmentationSource, register


def _load(config: dict) -> pd.DataFrame:
    # TODO: implement once SALT schema confirmed. Pseudocode:
    #     from datasets import load_dataset
    #     ds = load_dataset("sunbirdai/salt", split="train")
    #     df = ds.to_pandas()
    #     df = df[df["Language"] == "lug"]
    #     df = df.rename(columns={"source_text": "Question", "target_text": "Response"})
    #     df["ID"] = df.index.astype(str)
    #     return df[["ID", "Question", "Language", "Response"]]
    raise NotImplementedError(
        "sunbird_salt source is a stub — implement on data-drop day or when SALT snapshot is finalised."
    )


register(
    AugmentationSource(
        name="sunbird_salt",
        languages=("lug",),
        loader=_load,
    )
)
