"""Build a Zindi-format submission CSV.

The exact submission schema comes from ``SampleSubmission.csv``. Historically
the challenge scaffold assumed five columns, but the live competition files may
use four. The builder therefore mirrors a provided sample schema exactly and
falls back to the legacy five-column shape only when no sample schema is given.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

DEFAULT_SUBMISSION_COLUMNS: tuple[str, ...] = (
    "ID",
    "TargetBert",
    "TargetRLF1",
    "TargetR1F1",
    "TargetLLM",
)


@dataclass
class SubmissionMetadata:
    """Sidecar metadata written alongside each submission CSV."""

    run_id: str
    hypothesis: str
    model_hf_id: str
    adapter_path: str | None
    decoding_config: str
    n_rows: int
    created_utc: str
    git_hash: str
    local_rouge1: float | None = None
    local_rougeL: float | None = None
    local_afrolm_bs: float | None = None
    local_judge: float | None = None
    local_combined: float | None = None
    public_lb: float | None = None
    notes: str = ""


def build_submission_df(
    ids: list[str],
    answers: list[str],
    submission_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build the submission DataFrame using the provided schema.

    Raises:
        ValueError: length mismatch.
    """
    if len(ids) != len(answers):
        raise ValueError(f"len(ids)={len(ids)} != len(answers)={len(answers)}")
    columns = tuple(submission_columns or DEFAULT_SUBMISSION_COLUMNS)
    if len(columns) < 2:
        raise ValueError(f"submission schema must contain at least ID + one target column, got {columns}")
    answers = [("" if a is None else str(a)) for a in answers]
    ids = [str(i) for i in ids]
    df = pd.DataFrame({columns[0]: ids})
    for target_col in columns[1:]:
        df[target_col] = answers
    return df[list(columns)]


def write_submission_csv(
    df: pd.DataFrame,
    path: str | Path,
    metadata: SubmissionMetadata | None = None,
    submission_columns: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Validate the DataFrame and write it as CSV per Zindi spec.

    Also writes a sibling ``<path>.json`` with the metadata sidecar when provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Hard-enforce schema.
    columns = tuple(submission_columns or tuple(df.columns))
    if tuple(df.columns) != columns:
        raise ValueError(
            f"column order wrong: got {tuple(df.columns)}, expected {columns}"
        )
    # All Target cols must match row-for-row.
    base = df[columns[1]].astype(str)
    for target_col in columns[2:]:
        if not df[target_col].astype(str).equals(base):
            raise ValueError(f"column {target_col} differs from {columns[1]} — must be identical.")
    # No NaNs in ID.
    if df[columns[0]].isna().any() or (df[columns[0]].astype(str) == "").any():
        raise ValueError("empty IDs not allowed.")
    # No duplicate IDs.
    if df[columns[0]].duplicated().any():
        dupes = df[df[columns[0]].duplicated()][columns[0]].tolist()
        raise ValueError(f"duplicate IDs: {dupes[:10]}")

    df.to_csv(path, index=False, encoding="utf-8")

    if metadata is not None:
        sidecar = path.with_suffix(path.suffix + ".json")
        sidecar.write_text(json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8")

    return path


def _cli() -> None:
    """Minimal CLI used by ``make submit`` — reformats a predictions CSV into Zindi shape."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with columns ID, Prediction.")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--pred-col", default="Prediction")
    parser.add_argument("--output", required=True, help="Output submission CSV path.")
    parser.add_argument("--sample-csv", default=None, help="Optional SampleSubmission.csv to mirror exactly.")
    args = parser.parse_args()

    src = pd.read_csv(args.input, dtype=str).fillna("")
    sample_columns = None
    if args.sample_csv:
        sample_columns = tuple(pd.read_csv(args.sample_csv, dtype=str).columns.tolist())
    df = build_submission_df(
        src[args.id_col].tolist(),
        src[args.pred_col].tolist(),
        submission_columns=sample_columns,
    )
    write_submission_csv(df, args.output, submission_columns=sample_columns)
    print(f"Wrote submission CSV with {len(df)} rows to {args.output}")


if __name__ == "__main__":
    _cli()
