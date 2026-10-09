"""ltr.py — learned final selection over stored candidates (CPU, offline; 1st-place-style second stage).

    python scripts/ltr.py eval --pickers EXP-055,EXP-051,EXP-054 --gen EXP-038 --subsets Eng_Uga,Lug_Uga,Swa_Ken,Eng_Ken
    python scripts/ltr.py record ... --test-map "EXP-055=EXP-0aa;EXP-051=EXP-052;EXP-038=EXP-044+EXP-042" --desc "..."

For each question, the candidates are the distinct answers in the top-20 lists of the given picker runs.
Features per (question, answer): each picker's best rank (and whether it put the answer first), the
retriever rank, word TF-IDF overlap between the question and the answer (and its gap to the row's best),
char TF-IDF similarity to the answer's sibling questions, log answer frequency, relative length, and
ROUGE overlap with a fine-tuned generator's output. A gradient-boosted regressor predicts each
candidate's overlap with the gold answer; the highest prediction is picked.

--feats v2 adds (after 1st place's selector features): each answer's near-duplicate mass in the pool
(rows whose answer has bag-of-words cosine >= 0.8 with it), its mean/max similarity to and number of
near-duplicates among the row's other candidates, and per-question z-scores and gaps to the row's best for
the main signals. --per-subset fits one model per subset instead of one pooled model.

Honest protocol: Val scores = 5-fold GroupKFold out-of-fold (held-out rows always in training);
held-out scores = model trained on Val only; test = model trained on held-out + Val.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combine as C  # noqa: E402
import exp as X  # noqa: E402

sys.path.insert(0, str(X.ROOT / "autoresearch_nlp"))
from experiment import answer_overlap  # noqa: E402

TOP = 20
FEATURES_FIXED = ["logret", "q_ans", "q_ans_d", "sib_char", "lfreq", "len_ratio", "gen_overlap", "n_top1", "subset_code"]
FEATURES_V2 = ["l_echo", "cand_mean", "cand_max", "cand_echo"]
RELATIVE = ["logret", "q_ans", "sib_char", "lfreq", "gen_overlap", "l_echo", "cand_mean"]  # + logrank_*
ECHO = 0.8
FEATS = "v1"
PER_SUBSET = False
SUBSETS_ALL = ["Aka_Gha", "Amh_Eth", "Eng_Eth", "Eng_Gha", "Eng_Ken", "Eng_Uga", "Lug_Uga", "Swa_Ken"]


def load_frames():
    root = X.ROOT / "data"
    train = pd.read_csv(root / "raw" / "Train.csv", dtype=str).fillna("")
    val = pd.read_csv(root / "raw" / "Val.csv", dtype=str).fillna("")
    test = pd.read_csv(root / "raw" / "Test.csv", dtype=str).fillna("")
    held = pd.read_csv(X.REFS["held_out"], dtype=str).fillna("")
    work = pd.read_csv(X.ROOT / "autoresearch_nlp" / "data" / "processed" / "work_train.csv", dtype=str).fillna("")
    return {"held_out": (held, work), "val": (val, train), "test": (test, pd.concat([train, val], ignore_index=True))}


def build_features(queries: pd.DataFrame, pool: pd.DataFrame, picker_preds: list[pd.DataFrame], gen_preds: pd.DataFrame | None,
                   subsets: list[str]) -> pd.DataFrame:
    """One row per (question ID, candidate answer)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = []
    gen_of = dict(zip(gen_preds["ID"], gen_preds["pred"].astype(str))) if gen_preds is not None else {}
    gold_of = dict(zip(queries["ID"], queries["output"].astype(str).str.strip())) if "output" in queries else {}
    for sub in subsets:
        P = pool[pool["subset"] == sub].reset_index(drop=True)
        Q = queries[queries["subset"] == sub].reset_index(drop=True)
        if P.empty or Q.empty:
            continue
        P_ans = P["output"].astype(str).str.strip()
        ans_of = dict(zip(P["ID"], P_ans))
        freq = P_ans.value_counts().to_dict()
        median_len = float(P_ans.str.split().str.len().median() or 1)
        sib: dict[str, list[int]] = {}
        for i, a in enumerate(P_ans):
            sib.setdefault(a, []).append(i)
        vw = TfidfVectorizer(analyzer="word", token_pattern=C.TOKEN, sublinear_tf=True).fit(pd.concat([P["input"], P_ans]))
        if FEATS == "v2":
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.preprocessing import normalize

            uniq = sorted(freq)
            row_of = {a: k for k, a in enumerate(uniq)}
            bow = normalize(CountVectorizer(token_pattern=C.TOKEN, lowercase=False).fit_transform(uniq))
            near = (bow @ bow.T).toarray() >= ECHO  # distinct answer x distinct answer
            mass = near @ np.array([freq[a] for a in uniq], dtype=float)  # pool rows with a near-duplicate answer
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(pd.concat([P["input"], Q["input"]]))
        PC = vc.transform(P["input"])
        lists = [dict(zip(pp["ID"], pp["cand_ids"])) for pp in picker_preds]
        ret_list = dict(zip(picker_preds[0]["ID"], picker_preds[0]["ret_ids"]))
        for _, q in Q.iterrows():
            qid = q["ID"]
            ranks_by_picker = []
            cands: set[str] = set()
            for lst in lists:
                ids = [c for c in str(lst.get(qid, "")).split("|") if c in ans_of]
                r: dict[str, int] = {}
                for k, c in enumerate(ids):
                    r.setdefault(ans_of[c], k)
                ranks_by_picker.append(r)
                cands |= {ans_of[c] for c in ids[:TOP]}
            if not cands:
                continue
            ret: dict[str, int] = {}
            for k, c in enumerate(str(ret_list.get(qid, "")).split("|")):
                if c in ans_of:
                    ret.setdefault(ans_of[c], k)
            cands = sorted(cands)
            qa = (vw.transform([q["input"]]) @ vw.transform(cands).T).toarray()[0]
            qc = (vc.transform([q["input"]]) @ PC.T).toarray()[0]
            if FEATS == "v2":
                ix = [row_of[a] for a in cands]
                cs = (bow[ix] @ bow[ix].T).toarray()
                np.fill_diagonal(cs, np.nan)
            for n, a in enumerate(cands):
                feat = {"ID": qid, "subset": sub, "answer": a, "subset_code": SUBSETS_ALL.index(sub)}
                tops = 0
                for p, r in enumerate(ranks_by_picker):
                    rk = r.get(a, 50)
                    feat[f"logrank_{p}"] = -math.log1p(rk)
                    tops += int(rk == 0)
                feat["n_top1"] = tops
                feat["logret"] = -math.log1p(ret.get(a, 50))
                feat["q_ans"] = qa[n]
                feat["q_ans_d"] = qa[n] - qa.max()
                feat["sib_char"] = float(qc[sib[a]].max())
                feat["lfreq"] = math.log(freq.get(a, 1))
                feat["len_ratio"] = len(a.split()) / median_len
                feat["gen_overlap"] = answer_overlap(a, gen_of[qid]) if qid in gen_of else np.nan
                if FEATS == "v2":
                    feat["l_echo"] = math.log(mass[ix[n]])
                    other = cs[n][~np.isnan(cs[n])]
                    feat["cand_mean"] = float(other.mean()) if other.size else 0.0
                    feat["cand_max"] = float(other.max()) if other.size else 0.0
                    feat["cand_echo"] = int((other >= ECHO).sum())
                if gold_of:
                    feat["label"] = answer_overlap(a, gold_of[qid])
                rows.append(feat)
    df = pd.DataFrame(rows)
    return add_relative(df) if FEATS == "v2" and not df.empty else df


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question z-score and gap to the row's best, for the main signals."""
    cols = sorted(c for c in df.columns if c.startswith("logrank_")) + [c for c in RELATIVE if c in df]
    g = df.groupby("ID")[cols]
    mean, std, mx = g.transform("mean"), g.transform("std").fillna(0), g.transform("max")
    for c in cols:
        df[f"{c}_z"] = (df[c] - mean[c]) / (std[c] + 1e-6)
        df[f"{c}_gap"] = df[c] - mx[c]
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    base = sorted(c for c in df.columns if c.startswith("logrank_") and not c.endswith(("_z", "_gap"))) + FEATURES_FIXED
    if FEATS == "v2":
        base += FEATURES_V2 + sorted(c for c in df.columns if c.endswith(("_z", "_gap")))
    return base


KIND = "hgb"  # hgb = sklearn regressor on the overlap label; lambdarank = LightGBM ranker within each question


def fit(df: pd.DataFrame):
    if PER_SUBSET:
        return {sub: fit_one(g) for sub, g in df.groupby("subset")}
    return fit_one(df)


def predict(model, d: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        out = np.full(len(d), -np.inf)
        for sub, m in model.items():
            mask = (d["subset"] == sub).to_numpy()
            if mask.any():
                out[mask] = m.predict(d.loc[mask, feature_cols(d)])
        return out
    return model.predict(d[feature_cols(d)])


def fit_one(df: pd.DataFrame):
    if KIND == "lambdarank":
        from lightgbm import LGBMRanker

        d = df.sort_values("ID", kind="stable")
        rel = (d["label"] * 10).round().astype(int)  # graded relevance 0..10 from the ROUGE overlap
        m = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       reg_lambda=1.0, random_state=0, verbose=-1)
        return m.fit(d[feature_cols(d)], rel, group=d.groupby("ID", sort=True).size().to_numpy(),
                     categorical_feature=["subset_code"])
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                      categorical_features=[feature_cols(df).index("subset_code")], random_state=0)
    return m.fit(df[feature_cols(df)], df["label"])


def pick(df: pd.DataFrame, model) -> pd.DataFrame:
    d = df.copy()
    d["score"] = predict(model, d)
    top = d.sort_values("score", ascending=False).drop_duplicates("ID")
    return top[["ID", "subset", "answer"]].rename(columns={"answer": "pred"})


def oof_val(held_f: pd.DataFrame, val_f: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    out = []
    for tr, te in GroupKFold(n_splits=folds).split(val_f, groups=val_f["ID"]):
        model = fit(pd.concat([held_f, val_f.iloc[tr]], ignore_index=True))
        out.append(pick(val_f.iloc[te], model))
    return pd.concat(out, ignore_index=True)


def preds_for(exp_id: str, set_name: str) -> pd.DataFrame:
    return pd.read_csv(C.run_dir(exp_id) / f"{set_name}_preds.csv", dtype=str).fillna("")


def test_preds_for(spec: str) -> pd.DataFrame:
    """'EXP-044+EXP-042' -> concatenated test predictions of those runs."""
    return pd.concat([preds_for(e, "test") for e in spec.split("+")], ignore_index=True)


def evaluate(a):
    subsets = a.subsets.split(",")
    frames = load_frames()
    pickers = a.pickers.split(",")
    feats = {}
    for s in ("held_out", "val"):
        q, pool = frames[s]
        feats[s] = build_features(q, pool, [preds_for(p, s) for p in pickers], preds_for(a.gen, s) if a.gen else None, subsets)
        print(f"{s}: {len(feats[s]):,} candidate rows for {feats[s]['ID'].nunique():,} questions")
    val_pred = oof_val(feats["held_out"], feats["val"])
    held_pred = pick(feats["held_out"], fit(feats["val"]))
    return feats, {"held_out": held_pred, "val": val_pred}, frames


def report(preds: dict, subsets: list[str]) -> None:
    best = X.load_best()
    for s in ("held_out", "val"):
        sc = X.score_set(s, preds[s])["per_subset"]
        print(f"{s}: " + " | ".join(f"{sub} {sc[sub]:.4f} (best {best[sub]['scores'][s]:.4f})" for sub in subsets if sub in sc))


def cmd_eval(a) -> None:
    _, preds, _ = evaluate(a)
    report(preds, a.subsets.split(","))


def cmd_record(a) -> None:
    feats, preds, frames = evaluate(a)
    report(preds, a.subsets.split(","))
    subsets = a.subsets.split(",")
    test = None
    if a.test_map:
        mapping = dict(x.split("=") for x in a.test_map.split(";"))
        q, pool = frames["test"]
        tp = [test_preds_for(mapping[p]) for p in a.pickers.split(",")]
        tg = test_preds_for(mapping[a.gen]) if a.gen and a.gen in mapping else None
        tf = build_features(q, pool, tp, tg, subsets)
        test = pick(tf, fit(pd.concat([feats["held_out"], feats["val"]], ignore_index=True)))
        missing = set(q[q["subset"].isin(subsets)]["ID"]) - set(test["ID"])
        if missing:
            sys.exit(f"test predictions missing for {len(missing)} questions; check --test-map")
    model = {"hgb": "HistGradientBoostingRegressor", "lambdarank": "LGBMRanker(lambdarank)"}[KIND]
    cfg = {"combine": {"rule": "ltr", "pickers": a.pickers, "gen": a.gen, "top": TOP, "model": model, "feats": FEATS,
                       "per_subset": PER_SUBSET}}
    C.record_virtual(cfg, "ltr", subsets, preds, test, C.run_dir(a.pickers.split(",")[0]).name, a.hyp, a.desc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "record"):
        p = sub.add_parser(name)
        p.add_argument("--pickers", required=True, help="EXP ids of selector runs with cand_ids/ret_ids (first one supplies retriever ranks)")
        p.add_argument("--gen", help="EXP id of a generation run on the same subsets")
        p.add_argument("--subsets", required=True)
        p.add_argument("--hyp", default="H-011")
        p.add_argument("--model", choices=["hgb", "lambdarank"], default="hgb")
        p.add_argument("--feats", choices=["v1", "v2"], default="v1")
        p.add_argument("--per-subset", action="store_true")
        if name == "record":
            p.add_argument("--test-map", help="'EXP-055=EXP-0aa;EXP-051=EXP-052;EXP-038=EXP-044+EXP-042'")
            p.add_argument("--desc", required=True)
    a = ap.parse_args()
    global KIND, FEATS, PER_SUBSET
    KIND, FEATS, PER_SUBSET = a.model, a.feats, a.per_subset
    {"eval": cmd_eval, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    main()
