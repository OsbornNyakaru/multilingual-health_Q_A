"""combine.py — offline combinations of runs that already finished (CPU only, no molab).

    python scripts/combine.py agree --selector EXP-039 --gen EXP-038 --subsets Eng_Uga,Eng_Ken \\
        [--test-selector EXP-041 --test-gen EXP-044] --desc "..."

agree: keep the selector's answer, except when the generator's answer is one of the stored answers of
the selector's top-k candidates (both methods point at the same canned answer). k is tuned per subset
on held-out; Val is the check. The result is recorded as a normal run ("virtual": no GPU) with its own
EXP id, scored and adopted per subset by scripts/exp.py's rules, and usable by `exp.py submission`
when test predictions of both components are given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp as X  # noqa: E402

SETS = ("held_out", "val")


def agree_pick(sel_pred: str, cand_ids: str, gen: str, answer_of: dict, k: int) -> tuple[str, str]:
    """(answer, source): the generator's answer if it is a stored answer among the top-k candidates."""
    g = str(gen).strip()
    if k > 0 and g and g in {answer_of.get(c) for c in str(cand_ids).split("|")[:k]}:
        return g, "gen"
    return sel_pred, "sel"


def run_dir(exp_id: str) -> Path:
    spec = next((s for s in X.load_specs() if s["exp"] == exp_id), None)
    if not spec or not spec.get("run_id"):
        sys.exit(f"{exp_id}: no submitted spec")
    d = X.RUNS_DIR / spec["run_id"]
    if not d.exists():
        sys.exit(f"{exp_id}: not pulled yet (run `exp.py pull`)")
    return d


def combine_set(sel: pd.DataFrame, gen: pd.DataFrame, answer_of: dict, k_by_subset: dict) -> pd.DataFrame:
    if "cand_ids" not in sel.columns:
        sys.exit("selector predictions have no cand_ids (run it with diag_k > 0)")
    m = sel[["ID", "subset", "pred", "cand_ids"]].merge(gen[["ID", "pred"]].rename(columns={"pred": "gen"}), on="ID", how="left")
    m["gen"] = m["gen"].fillna("")
    picks = [agree_pick(p, c, g, answer_of, k_by_subset.get(s, 0)) for p, c, g, s in zip(m["pred"], m["cand_ids"], m["gen"], m["subset"])]
    m["pred"], m["source"] = [a for a, _ in picks], [b for _, b in picks]
    return m[["ID", "subset", "pred", "source"]]


def cmd_agree(a) -> None:
    subsets = a.subsets.split(",")
    grid = [int(x) for x in a.k_grid.split(",")]
    train = pd.read_csv(X.ROOT / "data" / "raw" / "Train.csv", dtype=str).fillna("")
    val = pd.read_csv(X.ROOT / "data" / "raw" / "Val.csv", dtype=str).fillna("")
    answer_of = dict(zip(train["ID"], train["output"].str.strip()))
    answer_of_test = {**answer_of, **dict(zip(val["ID"], val["output"].str.strip()))}  # test pool = Train + Val
    sd, gd = run_dir(a.selector), run_dir(a.gen)
    load = lambda d, s: pd.read_csv(d / f"{s}_preds.csv", dtype=str).fillna("")  # noqa: E731
    sel = {s: load(sd, s) for s in SETS}
    gen = {s: load(gd, s) for s in SETS}

    # tune k per subset on held-out only
    ref = pd.read_csv(X.REFS["held_out"], dtype=str).fillna("")
    k_by_subset, report = {}, []
    for sub in subsets:
        scores = {}
        for k in [0, *grid]:
            c = combine_set(sel["held_out"][sel["held_out"]["subset"] == sub], gen["held_out"], answer_of, {sub: k})
            m = c.merge(ref[["ID", "output"]], on="ID")
            scores[k] = X.prepare.score(m["pred"].tolist(), m["output"].tolist()).combined
        best_k = max(scores, key=lambda k: (round(scores[k], 6), -k))
        k_by_subset[sub] = best_k
        report.append(f"{sub}: k={best_k} (held-out {scores[0]:.4f} -> {scores[best_k]:.4f})")
    print("tuned on held-out: " + "; ".join(report))

    head = X.git("rev-parse", "HEAD")
    n = X.next_exp_number()
    cfg = {"combine": {"rule": "agree", "selector": sd.name, "gen": gd.name, "k": k_by_subset}}
    key = json.dumps(cfg, sort_keys=True)
    rid = f"exp{n:03d}_combine_agree_sub{len(subsets)}_{hashlib.sha256(key.encode()).hexdigest()[:6]}"
    out = X.RUNS_DIR / rid
    out.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        combine_set(sel[s][sel[s]["subset"].isin(subsets)], gen[s], answer_of, k_by_subset).to_csv(out / f"{s}_preds.csv", index=False)
    if a.test_selector and a.test_gen:
        ts = load(run_dir(a.test_selector), "test")
        tg = load(run_dir(a.test_gen), "test")
        combine_set(ts[ts["subset"].isin(subsets)], tg, answer_of_test, k_by_subset).to_csv(out / "test_preds.csv", index=False)
        print(f"test predictions combined from {a.test_selector} + {a.test_gen}")

    spec = {
        "exp": f"EXP-{n:03d}", "hypothesis": a.hyp, "description": a.desc, "parent": sd.name, "changed": ["combine"],
        "config": cfg, "eval": dict(X.ELIGIBLE_EVAL), "subsets": subsets, "max_minutes": 0, "virtual": True,
        "drafted": X.now(), "created": X.now(), "git_sha": head, "run_id": rid,
    }
    X.save_spec(spec)
    rec = {k: spec.get(k) for k in ("exp", "run_id", "hypothesis", "description", "parent", "changed", "eval", "subsets")}
    rec.update(status="ok", finished=X.now(), seconds=0, gpu={"name": "none (offline combine)"}, error=None, sets={})
    for s in SETS:
        rec["sets"][s] = X.score_set(s, pd.read_csv(out / f"{s}_preds.csv", dtype=str).fillna(""))
    best = X.load_best()
    rec["won"] = X.decide(spec, rec, best)
    X.BEST_JSON.write_text(json.dumps(best, indent=2) + "\n")
    with X.RESULTS_JSONL.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    X.write_vault_note(spec, rec)
    X.write_results_md(X.load_records(), X.load_best())
    per = {s: {sub: round(rec["sets"][s]["per_subset"][sub], 4) for sub in subsets} for s in SETS}
    print(f"{spec['exp']} {rid}: {per}  won: {rec['won'] or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree", help="selector + generator agreement")
    g.add_argument("--selector", required=True, help="EXP id of a run with cand_ids (diag_k > 0)")
    g.add_argument("--gen", required=True, help="EXP id of a generation run on the same subsets")
    g.add_argument("--subsets", required=True)
    g.add_argument("--k-grid", default="1,3,5,10,20")
    g.add_argument("--test-selector", help="EXP id with the selector's test predictions")
    g.add_argument("--test-gen", help="EXP id with the generator's test predictions")
    g.add_argument("--hyp", default="H-011")
    g.add_argument("--desc", required=True)
    a = ap.parse_args()
    {"agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    main()
