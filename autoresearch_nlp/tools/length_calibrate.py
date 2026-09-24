"""length_calibrate.py — per-subset reference-length stats -> LENGTH_BOUNDS.

ROUGE is F1, so generated length materially affects the score. This prints
per-subset reference-answer length quantiles (in WORDS) and a suggested
LENGTH_BOUNDS dict (in subword tokens, using a words->tokens fudge factor) that
you can paste into train.py. Calibrate to the median; bracket with p10/p90.

    python tools/length_calibrate.py            # uses work_train.csv
    python tools/length_calibrate.py --tokens-per-word 1.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(prepare.CACHE_DIR / "work_train.csv"))
    ap.add_argument("--tokens-per-word", type=float, default=1.4,
                    help="Rough words->subword-tokens multiplier for your tokenizer.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str).fillna("")
    sub_col = prepare.SUBSET_COL if prepare.SUBSET_COL in df.columns else None
    groups = [("ALL", df)] + (list(df.groupby(sub_col)) if sub_col else [])

    print(f"{'subset':<12} {'n':>6} {'p10':>5} {'median':>7} {'p90':>5}   (words)")
    bounds: dict[str, dict[str, int]] = {}
    for name, g in groups:
        w = g[prepare.OUTPUT_COL].astype(str).str.split().str.len()
        p10, med, p90 = int(w.quantile(.10)), int(w.median()), int(w.quantile(.90))
        print(f"{str(name):<12} {len(g):>6} {p10:>5} {med:>7} {p90:>5}")
        if name != "ALL":
            tpw = args.tokens_per_word
            bounds[str(name)] = {
                "min_new_tokens": max(4, int(p10 * tpw * 0.8)),
                "max_new_tokens": int(p90 * tpw) + 16,
            }

    print("\n# paste into train.py (subword tokens):")
    print("LENGTH_BOUNDS = {")
    for k, v in bounds.items():
        print(f'    "{k}": {{"min_new_tokens": {v["min_new_tokens"]}, "max_new_tokens": {v["max_new_tokens"]}}},')
    print("}")


if __name__ == "__main__":
    main()
