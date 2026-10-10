"""combine.py — offline combinations of runs that already finished (CPU only, no molab).

    python scripts/combine.py agree --selector EXP-039 --gen EXP-038 --subsets Eng_Uga,Eng_Ken \\
        [--test-selector EXP-041 --test-gen EXP-044] --desc "..."

    python scripts/combine.py rescore --base EXP-048 --subsets Lug_Uga,Swa_Ken [--test-base EXP-049] --desc "..."

    python scripts/combine.py pool --gen EXP-079 --subsets Eng_Uga,Lug_Uga [--test-gen EXP-080] --desc "..."

    python scripts/combine.py twin --twin EXP-098 --subsets Aka_Gha,Eng_Gha,Amh_Eth [--test-twin EXP-099] --desc "..."

agree: keep the selector's answer, except when the generator's answer is one of the stored answers of
the selector's top-k candidates (both methods point at the same canned answer). k is tuned per subset
on held-out; Val is the check.

rescore: over the selector's top-20 candidates grouped by answer, score each answer as
    -log(1 + best selector rank) + a * -log(1 + best retriever rank) + b * (q_ans - max q_ans)
where q_ans = word TF-IDF similarity between the new question and the candidate ANSWER text (the
question-only selector never sees it). a, b tuned per subset on held-out; Val is the check. The result is recorded as a normal run ("virtual": no GPU) with its own
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

    preds = {s: combine_set(sel[s][sel[s]["subset"].isin(subsets)], gen[s], answer_of, k_by_subset) for s in SETS}
    test = None
    if a.test_selector and a.test_gen:
        ts = load(run_dir(a.test_selector), "test")
        tg = load(run_dir(a.test_gen), "test")
        test = combine_set(ts[ts["subset"].isin(subsets)], tg, answer_of_test, k_by_subset)
    cfg = {"combine": {"rule": "agree", "selector": sd.name, "gen": gd.name, "k": k_by_subset}}
    record_virtual(cfg, "agree", subsets, preds, test, sd.name, a.hyp, a.desc)


def record_virtual(cfg: dict, rule: str, subsets: list, preds: dict, test, parent: str, hyp, desc) -> None:
    """Save the combined predictions as a run and score/decide/record it like any pulled run."""
    head = X.git("rev-parse", "HEAD")
    n = X.next_exp_number()
    key = json.dumps(cfg, sort_keys=True)
    rid = f"exp{n:03d}_combine_{rule}_sub{len(subsets)}_{hashlib.sha256(key.encode()).hexdigest()[:6]}"
    out = X.RUNS_DIR / rid
    out.mkdir(parents=True, exist_ok=True)
    for s, df in preds.items():
        df.to_csv(out / f"{s}_preds.csv", index=False)
    if test is not None:
        test.to_csv(out / "test_preds.csv", index=False)
        print("test predictions combined")
    spec = {
        "exp": f"EXP-{n:03d}", "hypothesis": hyp, "description": desc, "parent": parent, "changed": ["combine"],
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


# ── rescore ──────────────────────────────────────────────────────────────────

TOKEN = r"[^\s?.,!]+"

# ── pool ─────────────────────────────────────────────────────────────────────

POOL_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, float("inf"))


def overlap_matrix(items: list[str]) -> list[list[float]]:
    n = len(items)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = X.experiment.answer_overlap(items[i], items[j])
    return m


def pool_pick(m: list[list[float]], w: float) -> int:
    """Index into [base] + generated candidates: the MBR medoid when the base answer's votes count w times
    (w = inf keeps the base). m = overlap_matrix([base] + cands)."""
    if w == float("inf") or len(m) <= 1:
        return 0
    wt = [w] + [1.0] * (len(m) - 1)
    return max(range(len(m)), key=lambda i: (sum(wt[j] * m[i][j] for j in range(len(m)) if j != i), -i))


def best_preds(subset: str, set_name: str, run_id: str | None = None) -> pd.DataFrame:
    """The current best run's (or run_id's) predictions for one subset (test: the run itself, else its test children)."""
    b = run_id or X.load_best()[subset]["run_id"]
    rids = [b]
    if set_name == "test":
        rids += [r["run_id"] for r in reversed(X.load_records())
                 if r["status"] == "ok" and "test" in (r.get("eval") or {}) and r.get("parent") == b]
    for rid in rids:
        f = X.RUNS_DIR / rid / f"{set_name}_preds.csv"
        if f.exists():
            p = pd.read_csv(f, dtype=str).fillna("")
            if (p["subset"] == subset).any():
                return p[p["subset"] == subset][["ID", "subset", "pred"]]
    sys.exit(f"{subset}: no {set_name} predictions for best run {b}")


def pool_rows(base: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    """base rows + their overlap matrices over [base answer] + generated candidates."""
    g = gen[["ID", "pred"] + (["cands"] if "cands" in gen else [])].rename(columns={"pred": "gen"})
    m = base.merge(g, on="ID", how="left").fillna("")
    items = []
    for b, gp, c in zip(m["pred"], m["gen"], m["cands"] if "cands" in m else [""] * len(m)):
        cands = json.loads(c) if c else ([gp] if gp else [])
        items.append([b] + cands)
    m["items"] = items
    m["mat"] = [overlap_matrix(it) for it in items]
    return m


def pool_apply(m: pd.DataFrame, w_by_subset: dict) -> pd.DataFrame:
    out = m[["ID", "subset"]].copy()
    idx = [pool_pick(mat, w_by_subset.get(s, float("inf"))) for mat, s in zip(m["mat"], m["subset"])]
    out["pred"] = [it[i] for it, i in zip(m["items"], idx)]
    out["source"] = ["base" if i == 0 else "gen" for i in idx]
    return out


def cmd_pool(a) -> None:
    subsets = a.subsets.split(",")
    gd = run_dir(a.gen)
    gen = {s: pd.read_csv(gd / f"{s}_preds.csv", dtype=str).fillna("") for s in SETS}
    rows = {s: pool_rows(pd.concat([best_preds(sub, s) for sub in subsets]), gen[s]) for s in SETS}
    refs = {s: pd.read_csv(X.REFS[s], dtype=str).fillna("")[["ID", "output"]] for s in SETS}

    def score(s, sub, w):
        r = rows[s][rows[s]["subset"] == sub]
        p = pool_apply(r, {sub: w}).merge(refs[s], on="ID")
        return X.prepare.score(p["pred"].tolist(), p["output"].tolist()).combined

    w_by, report = {}, []
    for sub in subsets:
        sc = {w: score("held_out", sub, w) for w in POOL_WEIGHTS}
        w = max(sc, key=lambda w: (round(sc[w], 6), w))  # ties: trust the base more
        w_by[sub] = w
        report.append(f"{sub}: w={w} held-out {sc[float('inf')]:.4f} -> {sc[w]:.4f}, val {score('val', sub, float('inf')):.4f} -> {score('val', sub, w):.4f}")
    print("tuned on held-out:\n  " + "\n  ".join(report))
    preds = {s: pool_apply(rows[s], w_by) for s in SETS}
    test = None
    if a.test_gen:
        tg = pd.read_csv(run_dir(a.test_gen) / "test_preds.csv", dtype=str).fillna("")
        test = pool_apply(pool_rows(pd.concat([best_preds(sub, "test") for sub in subsets]), tg), w_by)
    cfg = {"combine": {"rule": "pool", "gen": gd.name, "base": {sub: X.load_best()[sub]["run_id"] for sub in subsets},
                       "w": {k: (None if v == float("inf") else v) for k, v in w_by.items()}}}
    record_virtual(cfg, "pool", subsets, preds, test, gd.name, a.hyp, a.desc)


# ── twin ─────────────────────────────────────────────────────────────────────

TWIN_SOURCES = {"nllb": "pred", "llm": "llm_answer"}
TWIN_THRESHOLDS = [round(0.6 + 0.01 * i, 2) for i in range(40)] + [None]  # None = never route


def twin_apply(base: pd.DataFrame, twin: pd.DataFrame, rule_by_subset: dict) -> pd.DataFrame:
    """Each row: its twin's translated answer (source per subset) when twin_sim >= the subset's threshold and that
    translation exists, else the base answer. rule_by_subset = {subset: (source, threshold or None)}."""
    t = twin[["ID", "twin_sim", *TWIN_SOURCES.values()]].rename(columns={"pred": "twin_pred"})
    m = base.merge(t, on="ID", how="left")
    out = m[["ID", "subset"]].copy()
    preds, src = [], []
    for _, r in m.iterrows():
        source, thr = rule_by_subset.get(r["subset"], ("nllb", None))
        col = "twin_pred" if source == "nllb" else TWIN_SOURCES[source]
        alt = r[col] if isinstance(r[col], str) and r[col].strip() else ""
        use = thr is not None and alt and float(r["twin_sim"] or 0) >= thr
        preds.append(alt if use else r["pred"])
        src.append(f"twin_{source}" if use else "base")
    out["pred"], out["source"] = preds, src
    return out


def cmd_twin(a) -> None:
    subsets = a.subsets.split(",")
    td = run_dir(a.twin)
    twin = {s: pd.read_csv(td / f"{s}_preds.csv", dtype={"ID": str, "subset": str}) for s in SETS}
    # base per subset: --base Aka_Gha=EXP-077,... (the twin run may itself be the best run by now), else the best run
    base_rid = {k: run_dir(v).name for k, _, v in (x.partition("=") for x in (a.base or "").split(",") if x)}
    base_rid = {sub: base_rid.get(sub) or X.load_best()[sub]["run_id"] for sub in subsets}
    base = {s: pd.concat([best_preds(sub, s, base_rid[sub]) for sub in subsets]) for s in SETS}
    refs = {s: pd.read_csv(X.REFS[s], dtype=str).fillna("")[["ID", "output"]] for s in SETS}

    def score(s, sub, rule):
        p = twin_apply(base[s][base[s]["subset"] == sub], twin[s], {sub: rule}).merge(refs[s], on="ID")
        return X.prepare.score(p["pred"].tolist(), p["output"].tolist()).combined

    rule_by, report = {}, []
    for sub in subsets:
        grid = [(src, t) for src in TWIN_SOURCES for t in TWIN_THRESHOLDS]
        sc = {r: score("held_out", sub, r) for r in grid}
        rule = max(grid, key=lambda r: (round(sc[r], 6), 2.0 if r[1] is None else r[1]))  # ties: route fewer rows
        rule_by[sub] = rule
        off = ("nllb", None)
        report.append(f"{sub}: {rule[0]} sim>={rule[1]} held-out {sc[off]:.4f} -> {sc[rule]:.4f}, "
                      f"val {score('val', sub, off):.4f} -> {score('val', sub, rule):.4f}")
    print("tuned on held-out:\n  " + "\n  ".join(report))
    preds = {s: twin_apply(base[s], twin[s], rule_by) for s in SETS}
    test = None
    if a.test_twin:
        tt = pd.read_csv(run_dir(a.test_twin) / "test_preds.csv", dtype={"ID": str, "subset": str})
        test = twin_apply(pd.concat([best_preds(sub, "test", base_rid[sub]) for sub in subsets]), tt, rule_by)
        print("test rows routed to twin:", test.groupby("subset")["source"].apply(lambda x: f"{(x != 'base').mean():.0%}").to_dict())
    cfg = {"combine": {"rule": "twin", "twin": td.name, "base": base_rid,
                       "route": {k: {"source": v[0], "min_sim": v[1]} for k, v in rule_by.items()}}}
    record_virtual(cfg, "twin", subsets, preds, test, td.name, a.hyp, a.desc)


def rescore_set(base: pd.DataFrame, queries: pd.DataFrame, pool: pd.DataFrame, weights: dict, top: int = 20) -> pd.DataFrame:
    """Re-pick each row's answer among its top candidates (see module doc). weights = {subset: (a, b)}."""
    import math

    from sklearn.feature_extraction.text import TfidfVectorizer

    q_text = dict(zip(queries["ID"], queries["input"].astype(str)))
    rows = []
    for sub, g in base.groupby("subset"):
        a_w, b_w = weights.get(sub, (0.0, 0.0))
        P = pool[pool["subset"] == sub]
        ans_of = dict(zip(P["ID"], P["output"].astype(str).str.strip()))
        vec = TfidfVectorizer(analyzer="word", token_pattern=TOKEN, sublinear_tf=True).fit(
            pd.concat([P["input"].astype(str), P["output"].astype(str).str.strip()]))
        for _, r in g.iterrows():
            cand = [c for c in str(r["cand_ids"]).split("|")[:top] if c in ans_of]
            ret = {c: i for i, c in enumerate(str(r["ret_ids"]).split("|"))}
            if not cand or (a_w == 0 and b_w == 0):
                rows.append((r["ID"], sub, r["pred"]))
                continue
            best: dict[str, dict] = {}
            for rank, c in enumerate(cand):
                d = best.setdefault(ans_of[c], {"ce": rank, "ret": ret.get(c, 50)})
                d["ret"] = min(d["ret"], ret.get(c, 50))
            answers = list(best)
            qa = (vec.transform([q_text[r["ID"]]]) @ vec.transform(answers).T).toarray()[0]
            top_qa = qa.max()
            score = {a: -math.log1p(best[a]["ce"]) + a_w * -math.log1p(best[a]["ret"]) + b_w * (qa[i] - top_qa)
                     for i, a in enumerate(answers)}
            rows.append((r["ID"], sub, max(score, key=score.get)))
    return pd.DataFrame(rows, columns=["ID", "subset", "pred"])


def cmd_rescore(a) -> None:
    subsets = a.subsets.split(",")
    root = X.ROOT / "data"
    train = pd.read_csv(root / "raw" / "Train.csv", dtype=str).fillna("")
    val = pd.read_csv(root / "raw" / "Val.csv", dtype=str).fillna("")
    held = pd.read_csv(X.REFS["held_out"], dtype=str).fillna("")
    work = pd.read_csv(X.ROOT / "autoresearch_nlp" / "data" / "processed" / "work_train.csv", dtype=str).fillna("")
    ctx = {"held_out": (held, work), "val": (val, train)}
    bd = run_dir(a.base)
    base = {s: pd.read_csv(bd / f"{s}_preds.csv", dtype=str).fillna("") for s in SETS}
    for s in SETS:
        if "ret_ids" not in base[s] or "cand_ids" not in base[s]:
            sys.exit(f"{a.base}: predictions need ret_ids and cand_ids (diag_k > 0 with a reranker)")
    grid = [(x, y) for x in (0.0, 0.25, 0.5, 1.0) for y in (0.0, 1.0, 2.0, 4.0, 8.0)]
    weights, report = {}, []
    for sub in subsets:
        b = base["held_out"][base["held_out"]["subset"] == sub]
        q, pool = ctx["held_out"]
        scores = {}
        for w in grid:
            m = rescore_set(b, q, pool, {sub: w}).merge(q[["ID", "output"]], on="ID")
            scores[w] = X.prepare.score(m["pred"].tolist(), m["output"].tolist()).combined
        w = max(scores, key=lambda w: (round(scores[w], 6), -w[0] - w[1]))
        weights[sub] = w
        report.append(f"{sub}: a={w[0]} b={w[1]} (held-out {scores[(0.0, 0.0)]:.4f} -> {scores[w]:.4f})")
    print("tuned on held-out: " + "; ".join(report))
    preds = {s: rescore_set(base[s][base[s]["subset"].isin(subsets)], *ctx[s], weights) for s in SETS}
    test = None
    if a.test_base:
        tb = pd.read_csv(run_dir(a.test_base) / "test_preds.csv", dtype=str).fillna("")
        tq = pd.read_csv(root / "raw" / "Test.csv", dtype=str).fillna("")
        test = rescore_set(tb[tb["subset"].isin(subsets)], tq, pd.concat([train, val], ignore_index=True), weights)
    cfg = {"combine": {"rule": "rescore", "base": bd.name, "weights": {k: list(v) for k, v in weights.items()}, "top": 20}}
    record_virtual(cfg, "rescore", subsets, preds, test, bd.name, a.hyp, a.desc)


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
    r = sub.add_parser("rescore", help="re-pick answers with selector rank + retriever rank + question/answer overlap")
    r.add_argument("--base", required=True, help="EXP id of a selector run with ret_ids and cand_ids")
    r.add_argument("--subsets", required=True)
    r.add_argument("--test-base", help="EXP id with the base run's test predictions")
    r.add_argument("--hyp", default="H-011")
    r.add_argument("--desc", required=True)
    p = sub.add_parser("pool", help="MBR over the best run's answer (weighted) + a generation run's candidates")
    p.add_argument("--gen", required=True, help="EXP id of a generation run with cands (gen_samples > 0)")
    p.add_argument("--subsets", required=True)
    p.add_argument("--test-gen", help="EXP id with the generator's test predictions")
    p.add_argument("--hyp", default="H-014")
    p.add_argument("--desc", required=True)
    t = sub.add_parser("twin", help="twin's translated answer when the twin match is confident, else the best run's")
    t.add_argument("--twin", required=True, help="EXP id of a mode=twin run (held-out + Val)")
    t.add_argument("--subsets", required=True)
    t.add_argument("--base", help="per-subset base runs, e.g. Aka_Gha=EXP-077,Amh_Eth=EXP-061 (default: the best run)")
    t.add_argument("--test-twin", help="EXP id with the twin run's test predictions")
    t.add_argument("--hyp", default="H-019")
    t.add_argument("--desc", required=True)
    a = ap.parse_args()
    {"agree": cmd_agree, "rescore": cmd_rescore, "pool": cmd_pool, "twin": cmd_twin}[a.cmd](a)


if __name__ == "__main__":
    main()
