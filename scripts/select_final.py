"""Final-submission decision matrix.

Zindi lets you select TWO submissions that will count for the private leaderboard.
If you forget, Zindi auto-picks your top-2 public-leaderboard scores — which on
low-data competitions is usually wrong because the public LB is 30% of the test
set (high variance).

Our picking heuristic:
    1. "best_public": highest public LB score.
    2. "best_robust": highest local held-out score AND smallest per-language
       gap (i.e., the model that doesn't collapse on any one language).

Reads submission sidecar JSONs from submissions/ and prints a ranked table.

Usage:
    python scripts/select_final.py
    python scripts/select_final.py --dir submissions/ --top 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    path: Path
    run_id: str = ""
    hypothesis: str = ""
    public_lb: float | None = None
    local_combined: float | None = None
    local_rouge1: float | None = None
    local_rougeL: float | None = None
    local_afrolm_bs: float | None = None
    local_judge: float | None = None
    language_gap: float | None = None
    n_rows: int = 0

    @property
    def robust_score(self) -> float:
        """Penalise per-language gap on top of the combined held-out score."""
        if self.local_combined is None:
            return float("-inf")
        gap_penalty = (self.language_gap or 0.0) * 0.5
        return self.local_combined - gap_penalty


def load_entries(submissions_dir: Path) -> list[Entry]:
    entries: list[Entry] = []
    for sidecar in sorted(submissions_dir.glob("*.csv.json")):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] {sidecar}: {exc}")
            continue
        csv_path = sidecar.with_suffix("")
        entries.append(
            Entry(
                path=csv_path,
                run_id=data.get("run_id", csv_path.stem),
                hypothesis=data.get("hypothesis", ""),
                public_lb=_as_float(data.get("public_lb")),
                local_combined=_as_float(data.get("local_combined")),
                local_rouge1=_as_float(data.get("local_rouge1")),
                local_rougeL=_as_float(data.get("local_rougeL")),
                local_afrolm_bs=_as_float(data.get("local_afrolm_bs")),
                local_judge=_as_float(data.get("local_judge")),
                language_gap=_as_float(data.get("language_gap")),
                n_rows=int(data.get("n_rows", 0)),
            )
        )
    return entries


def _as_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def print_table(entries: list[Entry], top: int = 20) -> None:
    if not entries:
        print("No sidecar metadata found in submissions/.")
        return
    cols = ("run_id", "public", "local", "r1", "rL", "bs", "judge", "gap", "robust")
    print("| " + " | ".join(cols) + " | hypothesis")
    print("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for e in entries[:top]:
        print(
            "| {run} | {pub} | {loc} | {r1} | {rl} | {bs} | {jd} | {gap} | {rob} | {hyp}".format(
                run=e.run_id[:28],
                pub=_fmt(e.public_lb),
                loc=_fmt(e.local_combined),
                r1=_fmt(e.local_rouge1),
                rl=_fmt(e.local_rougeL),
                bs=_fmt(e.local_afrolm_bs),
                jd=_fmt(e.local_judge),
                gap=_fmt(e.language_gap),
                rob=_fmt(e.robust_score if e.local_combined is not None else None),
                hyp=(e.hypothesis[:60] + "...") if len(e.hypothesis) > 60 else e.hypothesis,
            )
        )


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def recommend(entries: list[Entry]) -> dict[str, Entry | None]:
    with_pub = [e for e in entries if e.public_lb is not None]
    with_local = [e for e in entries if e.local_combined is not None]
    best_public = max(with_pub, key=lambda e: e.public_lb) if with_pub else None
    best_robust = max(with_local, key=lambda e: e.robust_score) if with_local else None
    return {"best_public": best_public, "best_robust": best_robust}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="submissions", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"submissions dir does not exist: {args.dir}")
        return

    entries = load_entries(args.dir)
    entries.sort(key=lambda e: e.public_lb or e.local_combined or 0.0, reverse=True)

    print(f"== Decision matrix ({len(entries)} submissions) ==")
    print_table(entries, top=args.top)

    rec = recommend(entries)
    print()
    print("== Recommended final submissions (pick 2) ==")
    for label, e in rec.items():
        if e is None:
            print(f"  {label}: none available")
            continue
        print(f"  {label}: {e.run_id}")
        print(f"     path:          {e.path}")
        print(f"     public LB:     {_fmt(e.public_lb)}")
        print(f"     local combined:{_fmt(e.local_combined)}")
        print(f"     language gap:  {_fmt(e.language_gap)}")
        print(f"     hypothesis:    {e.hypothesis}")


if __name__ == "__main__":
    main()
