"""WHO fact-sheet loader — canonical MSRH content.

STUB: scraping WHO fact sheets is out of scope for bootstrap; we expect a
curated Parquet file dropped into data/external/who_factsheets.parquet by a
human. This loader validates and passes through.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from afro_health_qa.data.augment import AugmentationSource, register

_DEFAULT_PATH = "data/external/who_factsheets.parquet"


def _load(config: dict) -> pd.DataFrame:
    path = Path(config.get("path", _DEFAULT_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"WHO fact-sheet parquet not found at {path}. "
            "Curate manually (MSRH topics, four target languages) before enabling this source."
        )
    df = pd.read_parquet(path)
    required = {"ID", "Question", "Language", "Response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


register(
    AugmentationSource(
        name="who_factsheets",
        languages=("swa", "lug", "aka", "amh"),
        loader=_load,
    )
)
