"""synth_sanity.py — end-to-end CPU smoke test of the harness (no GPU/model).

Mirrors autoresearch's local sanity check. Builds a tiny synthetic competition,
runs prepare.py's split + scoring + submission path, and asserts the invariants:
perfect predictions hit the ROUGE ceiling, empty predictions score ~0, and the
submission schema validates. Run this before burning any GPU time.

    python tools/synth_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare  # noqa: E402


def main() -> None:
    refs = [
        "drink clean water and rest well",
        "take the medicine after meals twice a day",
        "visit the clinic if the fever persists",
        "wash your hands with soap regularly",
    ] * 5
    n = len(refs)

    # 1. Perfect predictions -> ROUGE ceiling (combined == sum of ROUGE weights).
    s_perfect = prepare.score(refs, refs)
    ceiling = prepare.METRIC_WEIGHTS.get("rouge1_f1", 0) + prepare.METRIC_WEIGHTS.get("rougeL_f1", 0)
    assert abs(s_perfect.rouge1_f1 - 1.0) < 1e-9, s_perfect
    assert abs(s_perfect.combined - ceiling) < 1e-9, (s_perfect.combined, ceiling)

    # 2. Empty predictions -> ~0.
    s_empty = prepare.score([""] * n, refs)
    assert s_empty.combined < 1e-6, s_empty

    # 3. Partial overlap is between the two.
    partial = [r.split()[0] for r in refs]  # first word only
    s_partial = prepare.score(partial, refs)
    assert 0 < s_partial.combined < s_perfect.combined, s_partial

    # 4. Submission build + validate.
    ids = [f"ID_{i}" for i in range(n)]
    sub = prepare.build_submission(ids, refs)
    prepare.validate_submission(sub, expected_ids=ids)

    print("synth_sanity OK")
    print(f"  perfect combined = {s_perfect.combined:.4f} (== ROUGE ceiling {ceiling:.2f})")
    print(f"  partial combined = {s_partial.combined:.4f}")
    print(f"  empty   combined = {s_empty.combined:.4f}")
    print(f"  submission cols  = {list(sub.columns)} validated")


if __name__ == "__main__":
    main()
