from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALLOW_HELD_OUT = False
SEED = 42

_GEEZ_RE = re.compile(r"[\u1200-\u137F]")


@dataclass(frozen=True)
class ColumnMapping:
    id_col: str
    question_col: str
    lang_col: str
    answer_col: str


@dataclass(frozen=True)
class SubmissionSchema:
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    n_rows: int
    ids_match_test_exactly: bool


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass


def assert_not_loading_held_out(path: str | Path, allow_held_out: bool = ALLOW_HELD_OUT) -> None:
    path = Path(path)
    if "held_out" in path.name.lower() and not allow_held_out:
        raise AssertionError(
            "held_out.csv is sealed. Set allow_held_out=True only for an explicit final-week evaluation."
        )


def read_csv(path: str | Path, *, dtype: type | str = str, allow_held_out: bool = ALLOW_HELD_OUT) -> pd.DataFrame:
    assert_not_loading_held_out(path, allow_held_out=allow_held_out)
    return pd.read_csv(path, dtype=dtype)


def detect_column_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> ColumnMapping:
    common = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)

    id_candidates = [c for c in common if c.lower() == "id" or c.lower().endswith("id")]
    if not id_candidates:
        raise ValueError(f"could not infer id column from columns={sorted(common)}")
    id_col = id_candidates[0]

    question_candidates = [
        c for c in common if c.lower() in {"question", "input", "prompt", "query"} and c != id_col
    ]
    if not question_candidates:
        raise ValueError(f"could not infer question column from columns={sorted(common)}")
    question_col = question_candidates[0]

    answer_candidates = [
        c
        for c in set(train_df.columns) & set(val_df.columns)
        if c not in test_df.columns and c.lower() in {"response", "answer", "output", "target"}
    ]
    if not answer_candidates:
        raise ValueError(
            "could not infer answer column from train/val-only columns="
            f"{sorted(set(train_df.columns) & set(val_df.columns) - set(test_df.columns))}"
        )
    answer_col = answer_candidates[0]

    lang_candidates = [
        c for c in common if c not in {id_col, question_col} and c.lower() in {"language", "lang", "subset"}
    ]
    if not lang_candidates:
        leftovers = [c for c in common if c not in {id_col, question_col}]
        if len(leftovers) != 1:
            raise ValueError(f"could not infer language column from columns={sorted(common)}")
        lang_col = leftovers[0]
    else:
        lang_col = lang_candidates[0]

    return ColumnMapping(
        id_col=id_col,
        question_col=question_col,
        lang_col=lang_col,
        answer_col=answer_col,
    )


def infer_lang_code_map(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in sorted({str(v) for v in values}):
        key = raw.strip()
        lower = key.lower()
        if lower.startswith("swa") or "swahili" in lower:
            mapping[key] = "swa"
        elif lower.startswith("lug") or "luganda" in lower:
            mapping[key] = "lug"
        elif lower.startswith("aka") or "twi" in lower or "akan" in lower:
            mapping[key] = "aka"
        elif lower.startswith("amh") or "amharic" in lower:
            mapping[key] = "amh"
        elif lower.startswith("eng") or "english" in lower:
            mapping[key] = "eng"
        else:
            raise ValueError(f"unknown language/subset code: {raw}")
    return mapping


def contains_geez(text: str) -> bool:
    return bool(_GEEZ_RE.search(text or ""))


def build_submission_schema(sample_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str) -> SubmissionSchema:
    sample_ids = sample_df.iloc[:, 0].astype(str).tolist()
    test_ids = test_df[id_col].astype(str).tolist()
    return SubmissionSchema(
        columns=tuple(sample_df.columns.tolist()),
        dtypes={col: str(dtype) for col, dtype in sample_df.dtypes.items()},
        n_rows=int(len(sample_df)),
        ids_match_test_exactly=sample_ids == test_ids,
    )


def canonical_prompt_language(lang_code: str) -> str:
    return "eng" if lang_code == "eng" else lang_code

