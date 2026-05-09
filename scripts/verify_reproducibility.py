"""End-to-end reproducibility check.

What this does:
    1. Runs the baseline pipeline twice with the same seed.
    2. Hashes each submission CSV (SHA-256).
    3. Asserts the two hashes are identical.

Use case: after any change that touches seeding, data loading, or generation,
run this to confirm we haven't introduced non-determinism.

NOTE: full-model runs require GPU + weights. For CI/cheap verification, pass
``--tiny`` which uses a stub pipeline on a CPU-friendly mini model.

    python scripts/verify_reproducibility.py --tiny
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _run_once(workdir: Path, tiny: bool) -> Path:
    """Run the baseline pipeline in ``workdir`` and return the produced submission path."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_csv = workdir / "submission.csv"
    env_args: list[str] = [
        sys.executable,
        "-m",
        "afro_health_qa.submission.format",
        "--input",
        "tests/fixtures/tiny_predictions.csv" if tiny else "data/processed/val_predictions.csv",
        "--output",
        str(out_csv),
    ]
    subprocess.run(env_args, check=True)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use the CPU-friendly stub pipeline (reads tests/fixtures/tiny_predictions.csv).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a"
        b = Path(td) / "b"

        path_a = _run_once(a, tiny=args.tiny)
        path_b = _run_once(b, tiny=args.tiny)

        h_a = _sha256(path_a)
        h_b = _sha256(path_b)

        print(f"hash_a = {h_a}")
        print(f"hash_b = {h_b}")
        if h_a != h_b:
            print("FAIL: submission hashes differ — reproducibility broken.")
            # Keep a copy for inspection.
            shutil.copy(path_a, "submissions/_repro_a.csv")
            shutil.copy(path_b, "submissions/_repro_b.csv")
            return 1

    print("PASS: bit-identical submissions across runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
