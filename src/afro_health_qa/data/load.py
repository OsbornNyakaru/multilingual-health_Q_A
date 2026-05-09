"""Load the competition data into normalised DataFrames.

The raw competition files may use scaffold-era names like ``Question`` /
``Language`` / ``Response`` or the actual Zindi names ``input`` / ``subset`` /
``output``. This loader normalises those variants to the internal canonical
columns ``ID``, ``Question``, ``Language``, ``Response``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from afro_health_qa.data.runtime import ALLOW_HELD_OUT, assert_not_loading_held_out


@dataclass(frozen=True)
class CompetitionData:
    """Container for the competition splits as pandas DataFrames.

    Using pandas here rather than `datasets.Dataset` keeps the dependency
    surface small for the submission notebook. Convert lazily when needed.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame | None = None
    sample_submission: pd.DataFrame | None = None

RAW_TO_CANONICAL = {
    "ID": "ID",
    "Question": "Question",
    "Language": "Language",
    "Response": "Response",
    "input": "Question",
    "output": "Response",
    "subset": "Language",
}

REQUIRED_TRAIN_COLS = ("ID", "Question", "Language", "Response")
REQUIRED_TEST_COLS = ("ID", "Question", "Language")


def load_competition_data(
    raw_dir: str | Path = "data/raw",
    train_file: str = "Train.csv",
    val_file: str = "Val.csv",
    test_file: str = "Test.csv",
    sample_submission_file: str = "SampleSubmission.csv",
    strict: bool = True,
) -> CompetitionData:
    """Load the raw competition files from ``raw_dir``.

    Args:
        raw_dir: path to the directory holding the raw Zindi files.
        train_file / test_file / sample_submission_file: file names.
        strict: if True, raise when required columns are missing. If False,
            warn and return whatever is present (useful during early dev
            when we have only a partial sample of the data).

    Returns:
        CompetitionData with normalised train/val/test and optional sample submission.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir does not exist: {raw_dir}. Did you run scripts/download_data.sh?"
        )

    train_path = raw_dir / train_file
    val_path = raw_dir / val_file
    test_path = raw_dir / test_file
    sample_path = raw_dir / sample_submission_file

    if not train_path.exists():
        raise FileNotFoundError(f"missing training file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"missing test file: {test_path}")

    train = _normalise(pd.read_csv(train_path, dtype=str).fillna(""))
    val = _normalise(pd.read_csv(val_path, dtype=str).fillna("")) if val_path.exists() else None
    test = _normalise(pd.read_csv(test_path, dtype=str).fillna(""))

    _check_columns(train, REQUIRED_TRAIN_COLS, name="Train.csv", strict=strict)
    if val is not None:
        _check_columns(val, REQUIRED_TRAIN_COLS, name="Val.csv", strict=strict)
    _check_columns(test, REQUIRED_TEST_COLS, name="Test.csv", strict=strict)

    sample = None
    if sample_path.exists():
        sample = pd.read_csv(sample_path, dtype=str).fillna("")

    return CompetitionData(train=train, val=val, test=test, sample_submission=sample)


def load_processed_split(
    name: str,
    processed_dir: str | Path = "data/processed",
    *,
    allow_held_out: bool = ALLOW_HELD_OUT,
) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}.csv"
    assert_not_loading_held_out(path, allow_held_out=allow_held_out)
    return pd.read_csv(path, dtype=str).fillna("")


def _check_columns(df: pd.DataFrame, required: tuple[str, ...], name: str, strict: bool) -> None:
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return
    msg = f"{name} is missing required columns: {missing}. Found: {list(df.columns)}"
    if strict:
        raise ValueError(msg)
    import warnings

    warnings.warn(msg, stacklevel=2)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={c: RAW_TO_CANONICAL.get(c, c) for c in df.columns})
    return renamed


def to_hf_dataset(df: pd.DataFrame):
    """Convert a DataFrame to a `datasets.Dataset`. Kept thin to allow lazy imports."""
    from datasets import Dataset

    return Dataset.from_pandas(df, preserve_index=False)
