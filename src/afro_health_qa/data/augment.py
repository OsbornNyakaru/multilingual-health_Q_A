"""Augmentation orchestrator with source tagging.

Each augmentation row gets a provenance tag so we can always ablate by source
(e.g. "what if we drop medmcqa_translated — does Akan ROUGE-L go up?").
Provenance lives in an extra column ``Source`` that is stripped before training
but kept in the processed Parquet for audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

SourceFn = Callable[[dict], pd.DataFrame]


@dataclass(frozen=True)
class AugmentationSource:
    """One registered augmentation source."""

    name: str
    languages: tuple[str, ...]
    loader: SourceFn  # takes config dict, returns DataFrame with ID/Question/Language/Response
    enabled: bool = False
    max_rows: int | None = None


_REGISTRY: dict[str, AugmentationSource] = {}


def register(source: AugmentationSource) -> None:
    if source.name in _REGISTRY:
        raise ValueError(f"augmentation source '{source.name}' already registered")
    _REGISTRY[source.name] = source


def registered_sources() -> dict[str, AugmentationSource]:
    return dict(_REGISTRY)


REQUIRED_COLS = ("ID", "Question", "Language", "Response")


def build_augmented_dataset(
    base: pd.DataFrame,
    config: dict,
    limit_per_source: int | None = None,
) -> pd.DataFrame:
    """Concatenate competition data with enabled augmentation sources.

    Args:
        base: the original competition training DataFrame. Tagged Source="competition".
        config: the parsed data.yaml (needs config["augmentation"]["sources"]).
        limit_per_source: global cap; overrides per-source max_rows if smaller.

    Returns:
        Concatenated DataFrame with columns ID, Question, Language, Response, Source.
        IDs are prefixed with "<source>::" for augmented rows to guarantee uniqueness.
    """
    _validate_cols(base, "base")
    tagged_base = base.copy()
    tagged_base["Source"] = "competition"

    frames: list[pd.DataFrame] = [tagged_base]
    augmentation_cfg = (config.get("augmentation") or {}).get("sources") or {}

    for name, src_cfg in augmentation_cfg.items():
        if not src_cfg.get("enabled", False):
            continue
        if name not in _REGISTRY:
            raise ValueError(
                f"augmentation source '{name}' enabled in config but not registered. "
                f"Known: {sorted(_REGISTRY)}"
            )
        source = _REGISTRY[name]
        df = source.loader(src_cfg)
        _validate_cols(df, name)

        # Cap rows.
        cap = src_cfg.get("max_rows") or source.max_rows
        if limit_per_source is not None:
            cap = min(cap, limit_per_source) if cap else limit_per_source
        if cap is not None and len(df) > cap:
            df = df.head(cap)

        df = df.copy()
        df["Source"] = name
        df["ID"] = f"{name}::" + df["ID"].astype(str)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _validate_cols(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"augmentation source '{name}' missing columns: {missing}")
