"""Validate a submission CSV before upload.

Catches the common screw-ups:
    - Missing / extra columns
    - Wrong column order
    - Mismatched Target columns
    - BOM on UTF-8 file
    - Empty or duplicate IDs
    - IDs that don't match the test set (when --test-csv provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from afro_health_qa.submission.format import DEFAULT_SUBMISSION_COLUMNS


class SubmissionValidationError(ValueError):
    """Raised when a submission CSV violates the Zindi schema."""


def validate_submission_csv(
    path: str | Path,
    test_csv: str | Path | None = None,
    sample_csv: str | Path | None = None,
) -> dict:
    """Validate ``path``.

    Args:
        path: submission CSV path.
        test_csv: optional path to the Zindi Test.csv — if provided, cross-check
            that submission IDs cover test IDs exactly.

    Returns:
        dict of summary stats.

    Raises:
        SubmissionValidationError if any check fails.
    """
    path = Path(path)
    if not path.exists():
        raise SubmissionValidationError(f"submission file not found: {path}")

    # Explicit BOM check — pandas will silently strip it.
    with path.open("rb") as f:
        head = f.read(3)
    if head == b"\xef\xbb\xbf":
        raise SubmissionValidationError(
            f"{path} starts with a UTF-8 BOM. Zindi's parser often chokes — re-write without BOM."
        )

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    submission_columns = tuple(DEFAULT_SUBMISSION_COLUMNS)
    if sample_csv is not None:
        submission_columns = tuple(pd.read_csv(sample_csv, dtype=str).columns.tolist())

    if tuple(df.columns) != submission_columns:
        raise SubmissionValidationError(
            f"column order wrong: got {tuple(df.columns)}, expected {submission_columns}"
        )

    base = df[submission_columns[1]]
    for c in submission_columns[2:]:
        if not df[c].equals(base):
            raise SubmissionValidationError(
                f"column {c} differs from {submission_columns[1]} — all target columns must be identical."
            )

    if df[submission_columns[0]].duplicated().any():
        raise SubmissionValidationError("duplicate IDs in submission.")
    if (df[submission_columns[0]] == "").any():
        raise SubmissionValidationError("empty ID cells in submission.")

    stats = {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_empty_answers": int((df[submission_columns[1]] == "").sum()),
        "mean_chars": float(df[submission_columns[1]].str.len().mean() or 0.0),
    }

    if test_csv is not None:
        test = pd.read_csv(test_csv, dtype=str).fillna("")
        id_col = submission_columns[0]
        test_ids = set(test[id_col].astype(str))
        sub_ids = set(df[id_col].astype(str))
        missing = test_ids - sub_ids
        extra = sub_ids - test_ids
        if missing:
            raise SubmissionValidationError(
                f"submission missing {len(missing)} test IDs. First few: {sorted(missing)[:5]}"
            )
        if extra:
            raise SubmissionValidationError(
                f"submission has {len(extra)} IDs not in test set. First few: {sorted(extra)[:5]}"
            )

    return stats


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="submission CSV or directory of CSVs")
    parser.add_argument("--test-csv", default="data/raw/Test.csv")
    parser.add_argument("--sample-csv", default="data/raw/SampleSubmission.csv")
    args = parser.parse_args()

    p = Path(args.path)
    targets: list[Path] = []
    if p.is_dir():
        targets = sorted(p.glob("*.csv"))
    else:
        targets = [p]

    test_csv_arg = args.test_csv if Path(args.test_csv).exists() else None
    sample_csv_arg = args.sample_csv if Path(args.sample_csv).exists() else None
    for t in targets:
        try:
            stats = validate_submission_csv(t, test_csv=test_csv_arg, sample_csv=sample_csv_arg)
            print(f"OK  {t}  {stats}")
        except SubmissionValidationError as exc:
            print(f"BAD {t}  {exc}")


if __name__ == "__main__":
    _cli()
