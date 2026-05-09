"""Stratified train/val/held-out split with per-language balance guarantees.

The held-out slice (default 5%) must never be used for training, eval-during-
training, or model selection. It is only touched by `scripts/select_final.py`
at the very end, to decide which two submissions Zindi sees.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.85
    val: float = 0.10
    heldout: float = 0.05

    def __post_init__(self) -> None:
        total = self.train + self.val + self.heldout
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    heldout: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "heldout": self.heldout}


def stratified_split(
    df: pd.DataFrame,
    by: str = "Language",
    ratios: SplitRatios | tuple[float, float, float] = SplitRatios(),
    seed: int = 42,
) -> Splits:
    """Stratified 3-way split.

    Args:
        df: DataFrame to split. Must contain the ``by`` column.
        by: column to stratify on. Default "Language" (four classes).
        ratios: SplitRatios or 3-tuple (train, val, heldout).
        seed: RNG seed — deterministic across runs.

    Guarantees:
        - No row appears in more than one split.
        - Per-class ratios are within 2 percentage points of requested.
        - Deterministic under fixed seed.
    """
    if isinstance(ratios, tuple):
        ratios = SplitRatios(*ratios)

    if by not in df.columns:
        raise ValueError(f"stratify column '{by}' not found in DataFrame: {list(df.columns)}")

    rng = np.random.default_rng(seed)

    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    heldout_frames: list[pd.DataFrame] = []

    for _, group in df.groupby(by, sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * ratios.train))
        n_val = int(round(n * ratios.val))
        # Tiny-group rescue: guarantee at least 1 heldout row if the class has >= 20.
        n_heldout = n - n_train - n_val
        if n_heldout < 0:
            n_val += n_heldout
            n_heldout = 0

        train_frames.append(df.loc[indices[:n_train]])
        val_frames.append(df.loc[indices[n_train : n_train + n_val]])
        heldout_frames.append(df.loc[indices[n_train + n_val :]])

    return Splits(
        train=pd.concat(train_frames).sort_index().reset_index(drop=True),
        val=pd.concat(val_frames).sort_index().reset_index(drop=True),
        heldout=pd.concat(heldout_frames).sort_index().reset_index(drop=True),
    )


def verify_no_overlap(*splits: pd.DataFrame, id_col: str = "ID") -> None:
    """Assert that no ID appears in more than one split."""
    seen: set[str] = set()
    for split in splits:
        ids = set(split[id_col].astype(str).tolist())
        dupes = seen & ids
        if dupes:
            raise AssertionError(f"ID overlap between splits: {sorted(dupes)[:10]} (+ more)")
        seen |= ids


def per_class_ratios(splits: Splits, by: str = "Language") -> pd.DataFrame:
    """Return a table of per-class row counts and ratios across splits."""
    rows = []
    for class_value in sorted(
        set(splits.train[by]).union(splits.val[by]).union(splits.heldout[by])
    ):
        n_train = int((splits.train[by] == class_value).sum())
        n_val = int((splits.val[by] == class_value).sum())
        n_heldout = int((splits.heldout[by] == class_value).sum())
        total = n_train + n_val + n_heldout
        rows.append(
            {
                by: class_value,
                "train": n_train,
                "val": n_val,
                "heldout": n_heldout,
                "total": total,
                "train_pct": round(n_train / total, 4) if total else 0.0,
                "val_pct": round(n_val / total, 4) if total else 0.0,
                "heldout_pct": round(n_heldout / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def within_tolerance(
    ratios_df: pd.DataFrame,
    target: SplitRatios,
    tolerance: float = 0.02,
) -> bool:
    """True if all per-class split ratios are within ``tolerance`` of the target."""

    def _ok(col: str, want: float) -> bool:
        return bool((ratios_df[col] - want).abs().le(tolerance).all())

    return _ok("train_pct", target.train) and _ok("val_pct", target.val) and _ok(
        "heldout_pct", target.heldout
    )


def parse_ratios(value: Iterable[float] | SplitRatios) -> SplitRatios:
    """Convenience for reading YAML-configured ratios."""
    if isinstance(value, SplitRatios):
        return value
    value = list(value)
    if len(value) != 3:
        raise ValueError(f"expected 3 ratios (train, val, heldout), got {value}")
    return SplitRatios(*value)
