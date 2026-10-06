"""exp.py — local side of the autoresearch loop (see autoresearch_nlp/program.md).

    python scripts/exp.py new --from best --set embedder=intfloat/multilingual-e5-large --desc "..."
    python scripts/exp.py submit            # stamp git SHA, upload pending specs to the HF queue
    python scripts/exp.py pull [--wait]     # download results, score, decide, update ledger + vault
    python scripts/exp.py status
    python scripts/exp.py submission        # assemble the per-subset composite test submission

Files:
    experiments/specs/EXP-NNN.json   specs (committed)
    experiments/results.jsonl        one scored record per finished run (committed; scores only)
    experiments/BEST.json            best run per subset (committed)
    experiments/RESULTS.md           human ledger, regenerated from the two files above (committed)
    experiments/runs/<run_id>/       downloaded predictions (gitignored: contains competition text)

Scores are computed HERE with the frozen harness (autoresearch_nlp/prepare.py), on every eval set
the run used, and summarised as a test-mix-weighted average over subsets (weights = subset share
of Test.csv). A run is adopted per subset: it becomes that subset's best only if it beats the
current best by >= KEEP_DELTA on every eval set, and only full-set runs (rows = 0 on both
held_out and val) are eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch_nlp"))
import experiment  # noqa: E402
import prepare  # noqa: E402

EXP_DIR = ROOT / "experiments"
SPEC_DIR = EXP_DIR / "specs"
RUNS_DIR = EXP_DIR / "runs"
RESULTS_JSONL = EXP_DIR / "results.jsonl"
BEST_JSON = EXP_DIR / "BEST.json"
RESULTS_MD = EXP_DIR / "RESULTS.md"
LEADERBOARD_JSONL = EXP_DIR / "leaderboard.jsonl"
VAULT = ROOT / "vault"
RUNS_REPO = "nyakaruosborn/afro-health-qa-runs"
KEEP_DELTA = 0.003
ELIGIBLE_EVAL = {"held_out": 0, "val": 0}
DEFAULT_EVAL = {"held_out": 0, "val": 0}

REFS = {
    "held_out": ROOT / "autoresearch_nlp" / "data" / "processed" / "held_out.csv",
    "val": ROOT / "data" / "raw" / "Val.csv",
}
TEST_CSV = ROOT / "data" / "raw" / "Test.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_specs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(SPEC_DIR.glob("EXP-*.json"))]


def save_spec(spec: dict) -> None:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    (SPEC_DIR / f"{spec['exp']}.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")


def load_records() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    return [json.loads(line) for line in RESULTS_JSONL.read_text().splitlines() if line.strip()]


def load_best() -> dict:
    return json.loads(BEST_JSON.read_text()) if BEST_JSON.exists() else {}


def next_exp_number() -> int:
    nums = [int(m.group(1)) for p in (VAULT / "experiments").glob("EXP-*.md") if (m := re.match(r"EXP-(\d+)", p.name))]
    nums += [int(m.group(1)) for p in SPEC_DIR.glob("EXP-*.json") if (m := re.match(r"EXP-(\d+)", p.name))]
    return max(nums, default=-1) + 1


def slug(cfg: dict, eval_: dict, subsets) -> str:
    mode = cfg["mode"]
    short = {"retrieval": "ret", "zero_shot": "zs", "few_shot": "fs", "rag_few_shot": "rag", "router": "rtr"}.get(mode, mode)
    parts = [short]
    if mode in {"retrieval", "rag_few_shot", "router"}:
        parts.append(re.sub(r"[^a-z0-9]", "", cfg["embedder"].split("/")[-1].lower())[:10])
    if cfg.get("hybrid_with"):
        parts.append("hyb")
    if cfg.get("rerank_model"):
        parts.append("rr" + cfg.get("rerank_on", "question")[0])
    if cfg.get("select") == "vote" and mode in {"retrieval", "router"}:
        parts.append("vote")
    if mode != "retrieval":
        parts.append(re.sub(r"[^a-z0-9]", "", cfg["model_id"].split("/")[-1].lower())[:10])
    if mode in {"few_shot", "rag_few_shot"} or (mode == "router" and cfg.get("router_generate_mode") != "zero_shot"):
        parts.append(f"k{cfg['few_shot_k']}")
    if subsets:
        parts.append("sub" + str(len(subsets)))
    parts.append("n" + "-".join(f"{k[0]}{v or 'all'}" for k, v in sorted(eval_.items())))
    return "_".join(parts)


# ── new ──────────────────────────────────────────────────────────────────────


def parse_value(v: str):
    for cast in (json.loads,):
        try:
            return cast(v)
        except Exception:
            pass
    return v


def cmd_new(a) -> None:
    parent = None
    if a.from_ == "best":
        best = load_best()
        if best:
            # the run that holds the most subsets is the natural parent
            counts: dict[str, int] = {}
            for b in best.values():
                counts[b["run_id"]] = counts.get(b["run_id"], 0) + 1
            parent = max(counts, key=counts.get)
    elif a.from_:
        parent = a.from_
    base_cfg = dict(experiment.DEFAULT_CONFIG)
    base_eval, base_subsets = dict(DEFAULT_EVAL), None
    if parent:
        src = next((s for s in load_specs() if parent in (s.get("run_id"), s["exp"])), None)
        if src is None:
            sys.exit(f"parent {parent!r} not found in experiments/specs")
        base_cfg.update(src["config"])
        base_eval, base_subsets = dict(src["eval"]), src.get("subsets")
        parent = src.get("run_id") or src["exp"]
    cfg = dict(base_cfg)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        if k not in experiment.DEFAULT_CONFIG:
            sys.exit(f"unknown config key {k!r}; valid: {sorted(experiment.DEFAULT_CONFIG)}")
        cfg[k] = parse_value(v)
    eval_ = dict(base_eval)
    if a.eval:
        eval_ = {k: int(v) for k, _, v in (p.partition("=") for p in a.eval.split(","))}
    subsets = a.subsets.split(",") if a.subsets else base_subsets
    changed = sorted(k for k in cfg if cfg[k] != base_cfg.get(k))
    if parent and len(changed) > 1 and not a.allow_multi:
        sys.exit(f"more than one config change vs parent ({changed}); one lever per experiment, or pass --allow-multi")
    n = next_exp_number()
    spec = {
        "exp": f"EXP-{n:03d}",
        "hypothesis": a.hyp,
        "description": a.desc,
        "parent": parent,
        "changed": changed,
        "config": cfg,
        "eval": eval_,
        "subsets": subsets,
        "max_minutes": a.max_minutes,
        "drafted": now(),
    }
    save_spec(spec)
    print(f"wrote experiments/specs/{spec['exp']}.json  (changed vs parent: {changed or 'n/a'})")
    print("next: git add + commit + push, then `python scripts/exp.py submit`")


# ── submit ───────────────────────────────────────────────────────────────────


def ensure_runs_repo(api) -> None:
    api.create_repo(RUNS_REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(RUNS_REPO, repo_type="dataset").private:
        sys.exit(f"{RUNS_REPO} is PUBLIC; refusing to use it (predictions contain competition text)")


def cmd_submit(a) -> None:
    from huggingface_hub import HfApi

    if git("status", "--porcelain", "--", "autoresearch_nlp/experiment.py", "autoresearch_nlp/prepare.py"):
        sys.exit("experiment.py / prepare.py have uncommitted changes; commit and push first")
    head = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
    if head != remote:
        sys.exit(f"HEAD {head[:7]} is not what GitHub main has ({remote[:7]}); push first (molab fetches code by SHA)")
    api = HfApi()
    ensure_runs_repo(api)
    todo = [s for s in load_specs() if not s.get("run_id") and not s.get("withdrawn") and (not a.exp or s["exp"] in a.exp)]
    if not todo:
        print("nothing to submit")
        return
    for spec in todo:
        spec["git_sha"] = head
        spec["created"] = now()
        key = json.dumps({k: spec[k] for k in ("config", "eval", "subsets", "git_sha")}, sort_keys=True)
        num = spec["exp"].split("-")[1]
        spec["run_id"] = f"exp{num}_{slug(spec['config'], spec['eval'], spec['subsets'])}_{hashlib.sha256(key.encode()).hexdigest()[:6]}"
        api.upload_file(
            path_or_fileobj=json.dumps(spec, indent=2).encode(), path_in_repo=f"{'queue' if a.legacy_queue else 'queue_v2'}/{spec['run_id']}.json",
            repo_id=RUNS_REPO, repo_type="dataset", commit_message=f"queue {spec['run_id']}",
        )
        save_spec(spec)
        print(f"queued {spec['exp']} as {spec['run_id']}")
    print("commit experiments/specs/ when convenient; press Run queue on molab if it isn't watching")


# ── scoring + decisions ──────────────────────────────────────────────────────


def test_mix() -> dict[str, float]:
    t = pd.read_csv(TEST_CSV, dtype=str)
    return (t["subset"].value_counts(normalize=True)).to_dict()


def score_set(set_name: str, preds: pd.DataFrame) -> dict:
    ref = pd.read_csv(REFS[set_name], dtype=str).fillna("")[["ID", "subset", "output"]]
    df = preds[["ID", "pred"]].merge(ref, on="ID", how="left")
    if df["output"].isna().any():
        raise ValueError(f"{set_name}: {df['output'].isna().sum()} predictions have no reference")
    df["pred"] = df["pred"].fillna("")
    per = prepare.score_per_subset(df, "pred")
    weights = test_mix()
    subsets = {s: float(per.loc[s, "combined_no_judge"]) for s in per.index if s != "ALL"}
    wsum = sum(weights.get(s, 0) for s in subsets)
    weighted = sum(weights.get(s, 0) * v for s, v in subsets.items()) / wsum if wsum else float("nan")
    return {
        "n": len(df),
        "combined": float(per.loc["ALL", "combined_no_judge"]),
        "rouge1_f1": float(per.loc["ALL", "rouge1_f1"]),
        "rougeL_f1": float(per.loc["ALL", "rougeL_f1"]),
        "test_mix": round(weighted, 4),
        "per_subset": subsets,
    }


def decide(spec: dict, rec: dict, best: dict) -> list[str]:
    """Update best in place; return subsets this run won."""
    if rec["status"] != "ok" or spec.get("eval") != ELIGIBLE_EVAL:
        return []
    won = []
    subsets = set(rec["sets"]["held_out"]["per_subset"]) & set(rec["sets"]["val"]["per_subset"])
    for s in sorted(subsets):
        mine = {k: rec["sets"][k]["per_subset"][s] for k in ELIGIBLE_EVAL}
        cur = best.get(s)
        if cur is None or all(mine[k] - cur["scores"][k] >= KEEP_DELTA for k in ELIGIBLE_EVAL):
            best[s] = {"run_id": rec["run_id"], "exp": rec["exp"], "scores": mine, "since": rec["finished"]}
            won.append(s)
    return won


def composite(best: dict) -> dict:
    w = test_mix()
    out = {}
    for k in ELIGIBLE_EVAL:
        covered = [s for s in w if s in best]
        tot = sum(w[s] for s in covered)
        out[k] = round(sum(w[s] * best[s]["scores"][k] for s in covered) / tot, 4) if tot else None
    out["subsets_covered"] = len([s for s in w if s in best])
    return out


# ── ledger + vault ───────────────────────────────────────────────────────────


def write_results_md(records: list[dict], best: dict) -> None:
    w = test_mix()
    comp = composite(best)
    lines = [
        "# experiment results",
        "",
        "generated by `scripts/exp.py pull`; do not edit by hand. scores are 0.37·rouge-1 + 0.37·rouge-l",
        "(judge not measured, max 0.74). `test-mix` = average over subsets weighted by each subset's share of Test.csv.",
        f"a run is adopted per subset only if it wins by ≥ {KEEP_DELTA} on both held-out and val (full sets).",
        "",
        "## current best (per subset)",
        "",
        f"composite test-mix: **held-out {comp['held_out']}**, **val {comp['val']}** ({comp['subsets_covered']}/8 subsets covered)",
        "",
        "| subset | test share | best run | held-out | val |",
        "|---|--:|---|--:|--:|",
    ]
    for s in sorted(w, key=w.get, reverse=True):
        b = best.get(s)
        lines.append(
            f"| {s} | {w[s]:.1%} | {b['exp'] + ' `' + b['run_id'] + '`' if b else '—'} | "
            f"{b['scores']['held_out']:.4f} | {b['scores']['val']:.4f} |" if b else f"| {s} | {w[s]:.1%} | — | — | — |"
        )
    lines += [
        "",
        "## runs (newest first)",
        "",
        "| exp | run | finished | change | eval | held-out test-mix | val test-mix | won subsets | status | note |",
        "|---|---|---|---|---|--:|--:|---|---|---|",
    ]
    for r in reversed(records):
        sets = r.get("sets", {})
        ho = sets.get("held_out", {}).get("test_mix")
        va = sets.get("val", {}).get("test_mix")
        lines.append(
            f"| {r['exp']} | `{r['run_id']}` | {r['finished'][:16].replace('T', ' ')} | {', '.join(r.get('changed') or []) or '—'} | "
            f"{', '.join(f'{k}={v or 'all'}' for k, v in r['eval'].items())} | "
            f"{'' if ho is None else f'{ho:.4f}'} | {'' if va is None else f'{va:.4f}'} | "
            f"{', '.join(r.get('won') or []) or '—'} | {r['status']} | {r.get('description') or ''} |"
        )
    lb = [json.loads(x) for x in LEADERBOARD_JSONL.read_text().splitlines() if x.strip()] if LEADERBOARD_JSONL.exists() else []
    if lb:
        lines += [
            "",
            "## zindi submissions (newest first)",
            "",
            "total = 0.37·rouge-1 + 0.37·rouge-l + 0.26·judge, scored by zindi on Test.",
            "",
            "| date | file | public | private | rouge-1 | rouge-l | judge | local val test-mix | note |",
            "|---|---|--:|--:|--:|--:|--:|--:|---|",
        ]
        for e in reversed(lb):
            lines.append(
                f"| {e['date']} | `{e['file']}` | {e['public']:.4f} | {e.get('private') or 0:.4f} | {e.get('rouge1') or 0:.4f} | "
                f"{e.get('rougeL') or 0:.4f} | {e.get('judge') or 0:.4f} | {e.get('local_val') or ''} | {e.get('note') or ''} |"
            )
    lines += ["", "## per-subset scores by run", ""]
    subsets = sorted(w, key=w.get, reverse=True)
    lines += ["| exp | set | " + " | ".join(subsets) + " |", "|---|---|" + "--:|" * len(subsets)]
    for r in reversed(records):
        for k, v in r.get("sets", {}).items():
            ps = v.get("per_subset", {})
            lines.append(f"| {r['exp']} | {k} | " + " | ".join(f"{ps[s]:.4f}" if s in ps else "" for s in subsets) + " |")
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def write_vault_note(spec: dict, rec: dict) -> None:
    exp_dir = VAULT / "experiments"
    name = f"{spec['exp']}-{rec['run_id'].split('_', 1)[1].rsplit('_', 1)[0].replace('_', '-')}"
    hyp_links = [p.stem for p in (VAULT / "hypotheses").glob(f"{spec['hypothesis']}-*.md")] if spec.get("hypothesis") else []
    links = [f"[[{h}]]" for h in hyp_links] + ["[[00_INDEX]]"]
    status = "confirmed" if rec.get("won") else "rejected"
    body = [
        "---",
        "type: experiment",
        f"id: {spec['exp']}",
        f"created: {rec['finished'][:10]}",
        f"status: {status}",
        f"links: {json.dumps(links, ensure_ascii=False)}",
        "---",
        f"# {spec['exp']} {spec.get('description') or ''}".rstrip(),
        "",
        f"- run: `{rec['run_id']}` · git `{spec['git_sha'][:7]}` · status **{rec['status']}** · {rec.get('seconds', '?')} s on {rec.get('gpu', {}).get('name', '?')}",
        f"- parent: `{spec.get('parent') or '—'}` · changed: {', '.join(spec.get('changed') or []) or '—'}",
        f"- config: `{json.dumps({k: v for k, v in spec['config'].items() if v != experiment.DEFAULT_CONFIG.get(k)}, ensure_ascii=False)}` (non-default keys)",
        f"- eval: {spec['eval']} · subsets: {spec.get('subsets') or 'all'}",
        f"- adopted for subsets: **{', '.join(rec.get('won') or []) or 'none'}**",
        "",
    ]
    for k, v in rec.get("sets", {}).items():
        body.append(f"**{k}**: combined {v['combined']:.4f}, test-mix {v['test_mix']:.4f}, R1 {v['rouge1_f1']:.4f}, RL {v['rougeL_f1']:.4f}, n={v['n']}")
        body.append("")
        body.append("| subset | combined |")
        body.append("|---|--:|")
        body += [f"| {s} | {x:.4f} |" for s, x in sorted(v["per_subset"].items())]
        body.append("")
    if rec.get("error"):
        body += ["**error**", "", "```", rec["error"][-1500:], "```", ""]
    body += ["## Links", *[f"- {link}" for link in links], ""]
    path = exp_dir / f"{name}.md"
    path.write_text("\n".join(body))
    idx = VAULT / "00_INDEX.md"
    text = idx.read_text()
    bullet = f"- [[{name}]] — {spec.get('description') or rec['run_id']} ({rec['status']}; won: {', '.join(rec.get('won') or []) or 'none'})"
    if f"[[{name}]]" not in text:
        marker = "## Decisions"
        text = text.replace(marker, bullet + "\n\n" + marker, 1) if marker in text else text + "\n" + bullet + "\n"
        idx.write_text(text)


# ── pull ─────────────────────────────────────────────────────────────────────


def cmd_pull(a) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    deadline = time.time() + 60 * a.timeout
    while True:
        done_ids = {r["run_id"] for r in load_records()}
        waiting = [s for s in load_specs() if s.get("run_id") and s["run_id"] not in done_ids]
        if not waiting:
            print("no submitted runs waiting")
            return
        files = set(api.list_repo_files(RUNS_REPO, repo_type="dataset"))
        ready = [s for s in waiting if f"runs/{s['run_id']}/result.json" in files]
        for spec in ready:
            rid = spec["run_id"]
            out = RUNS_DIR / rid
            out.mkdir(parents=True, exist_ok=True)
            for f in [f for f in files if f.startswith(f"runs/{rid}/")]:
                p = hf_hub_download(RUNS_REPO, f, repo_type="dataset", force_download=True)
                (out / Path(f).name).write_bytes(Path(p).read_bytes())
            result = json.loads((out / "result.json").read_text())
            rec = {k: spec.get(k) for k in ("exp", "run_id", "hypothesis", "description", "parent", "changed", "eval", "subsets")}
            rec.update(status=result["status"], finished=result.get("finished", now()), seconds=result.get("seconds"),
                       gpu=result.get("gpu"), error=result.get("error"), sets={})
            if result["status"] == "ok":
                for set_name in spec["eval"]:
                    pf = out / f"{set_name}_preds.csv"
                    if set_name in REFS and pf.exists():
                        rec["sets"][set_name] = score_set(set_name, pd.read_csv(pf, dtype=str).fillna(""))
            best = load_best()
            rec["won"] = decide(spec, rec, best)
            BEST_JSON.write_text(json.dumps(best, indent=2) + "\n")
            with RESULTS_JSONL.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            write_vault_note(spec, rec)
            summary = ", ".join(f"{k} test-mix {v['test_mix']:.4f}" for k, v in rec["sets"].items())
            print(f"{spec['exp']} {rid}: {rec['status']}  {summary}  won: {rec['won'] or 'none'}")
        if ready:
            write_results_md(load_records(), load_best())
            print("updated experiments/RESULTS.md, BEST.json, results.jsonl and the vault")
        still = len(waiting) - len(ready)
        if not a.wait or still == 0 or time.time() > deadline:
            if still:
                print(f"{still} run(s) still pending on molab")
            return
        time.sleep(60)


def cmd_status(a) -> None:
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(RUNS_REPO, repo_type="dataset"))
    done = {r["run_id"]: r for r in load_records()}
    for s in load_specs():
        rid = s.get("run_id")
        if s.get("withdrawn"):
            state = f"withdrawn: {s['withdrawn']}"
        elif not rid:
            state = "draft (not submitted)"
        elif rid in done:
            state = f"pulled: {done[rid]['status']}, won {done[rid].get('won') or 'none'}"
        elif f"runs/{rid}/result.json" in files:
            state = "finished on molab, run `pull`"
        elif any(f.startswith(f"runs/{rid}/") for f in files):
            state = "running / partial"
        else:
            state = "queued"
        print(f"{s['exp']}  {rid or '':48s}  {state}")


def cmd_submission(a) -> None:
    best = load_best()
    if not best:
        sys.exit("no best runs yet")
    test = pd.read_csv(TEST_CSV, dtype=str).fillna("")
    # a best run's test predictions come from a child run: same config, eval {"test": 0}, parent = best run
    test_run = {
        r["parent"]: r["run_id"] for r in load_records()
        if r["status"] == "ok" and "test" in (r.get("eval") or {}) and r.get("parent")
    }
    parts, missing = [], []
    for s in sorted(test["subset"].unique()):
        b = best.get(s)
        tr = test_run.get(b["run_id"]) if b else None
        pf = RUNS_DIR / tr / "test_preds.csv" if tr else None
        if not pf or not pf.exists():
            missing.append(f"{s} (best {b['run_id'] if b else 'none'})")
            continue
        p = pd.read_csv(pf, dtype=str).fillna("")
        parts.append(p[p["subset"] == s])
    if missing:
        sys.exit("no test predictions for: " + "; ".join(missing)
                 + "\nqueue them with: exp.py new --from <run_id> --eval test=0 --desc 'test predictions', then submit + pull")
    preds = pd.concat(parts)
    sub = prepare.build_submission(preds["ID"].tolist(), preds["pred"].tolist())
    prepare.validate_submission(sub, expected_ids=test["ID"].tolist())
    out = RUNS_DIR / "submissions" / f"composite_{datetime.now():%Y%m%d_%H%M}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} ({len(sub)} rows). Upload it to Zindi; it is gitignored on purpose.")


def cmd_lb(a) -> None:
    entry = {"date": a.date or datetime.now().strftime("%Y-%m-%d"), "file": a.file, "public": a.public, "private": a.private,
             "rouge1": a.rouge1, "rougeL": a.rougeL, "judge": a.judge, "local_val": a.local_val, "note": a.note}
    with LEADERBOARD_JSONL.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    write_results_md(load_records(), load_best())
    print(f"recorded {a.file}: public {a.public}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="draft a spec")
    n.add_argument("--from", dest="from_", default="best", help="parent: 'best', a run_id, an EXP id, or '' for defaults")
    n.add_argument("--set", action="append", help="config override key=value (JSON values allowed)")
    n.add_argument("--eval", help="eval sets, e.g. held_out=0,val=0 or test=0 (0 = all rows)")
    n.add_argument("--subsets", help="comma-separated subsets to run on (default all)")
    n.add_argument("--hyp", help="hypothesis id, e.g. H-011")
    n.add_argument("--desc", required=True, help="one line: what changes and why")
    n.add_argument("--max-minutes", type=float, default=120)
    n.add_argument("--allow-multi", action="store_true", help="allow more than one changed key")
    s = sub.add_parser("submit", help="stamp git SHA and queue specs on HF")
    s.add_argument("exp", nargs="*", help="EXP ids (default: all unsubmitted)")
    s.add_argument("--legacy-queue", action="store_true", help="post to queue/ so runners older than 2026-10-06-v2 pick it up")
    p = sub.add_parser("pull", help="fetch, score and record finished runs")
    p.add_argument("--wait", action="store_true", help="poll every 60 s until all submitted runs finish")
    p.add_argument("--timeout", type=float, default=240, help="minutes to wait with --wait")
    sub.add_parser("status")
    sub.add_parser("submission")
    lb = sub.add_parser("lb", help="record a Zindi leaderboard result")
    lb.add_argument("--file", required=True)
    lb.add_argument("--public", type=float, required=True)
    lb.add_argument("--private", type=float)
    lb.add_argument("--rouge1", type=float)
    lb.add_argument("--rougeL", type=float)
    lb.add_argument("--judge", type=float)
    lb.add_argument("--local-val", dest="local_val", type=float, help="local val test-mix of the same composite")
    lb.add_argument("--date")
    lb.add_argument("--note", default="")
    a = ap.parse_args()
    {"new": cmd_new, "submit": cmd_submit, "pull": cmd_pull, "status": cmd_status, "submission": cmd_submission, "lb": cmd_lb}[a.cmd](a)


if __name__ == "__main__":
    main()
