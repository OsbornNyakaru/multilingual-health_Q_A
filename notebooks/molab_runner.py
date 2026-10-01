# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24",
#     "pandas>=2.2",
#     "numpy",
#     "rouge-score==0.1.2",
#     "huggingface_hub>=0.26",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "peft>=0.13",
#     "sentence-transformers>=3.0",
#     "scikit-learn",
#     "sentencepiece",
#     "bitsandbytes>=0.43; platform_system == 'Linux'",
# ]
# ///
"""molab_runner.py — the frozen experiment runner for the autoresearch loop.

You never edit experiments here. The loop works like this:
  1. Locally, `python scripts/exp.py new ...` writes an experiment spec, `git push` publishes the
     code, and `python scripts/exp.py submit` drops the spec into the private HF dataset
     `<user>/afro-health-qa-runs` under queue/.
  2. Here, press **Run queue**. For each pending spec the runner downloads the repo at the spec's
     git SHA, checks the frozen harness (prepare.py) by hash, runs autoresearch_nlp/experiment.py,
     scores with prepare.py, and uploads predictions + result.json to runs/<run_id>/.
  3. Locally, `python scripts/exp.py pull` downloads results, decides keep/drop per subset and
     updates experiments/RESULTS.md and the vault.

This notebook never writes to GitHub and never decides keep/drop.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Afro Health QA — runner")


@app.cell
def _():
    import gc
    import hashlib
    import importlib.util
    import io
    import json
    import os
    import sys
    import tarfile
    import time
    import traceback
    import urllib.request
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return (
        Path, datetime, gc, hashlib, importlib, io, json, mo, os, pd, sys, tarfile, time,
        timezone, traceback, urllib,
    )


@app.cell
def _(mo):
    mo.md("""
    # Afro Health QA — experiment runner

    1. Turn on the GPU. 2. Paste a Hugging Face token (read on the data repo, **write on the runs
    repo**, and accepted licences for any gated model). 3. Press **Run queue**.
    """)
    return


@app.cell
def _(mo, os):
    hf_token_input = mo.ui.text(kind="password", value="", label="Hugging Face token (session only)", full_width=True)
    data_repo_input = mo.ui.text(value=os.environ.get("HF_DATA_REPO", "nyakaruosborn/afro-health-qa-data"), label="data repo", full_width=True)
    runs_repo_input = mo.ui.text(value=os.environ.get("HF_RUNS_REPO", "nyakaruosborn/afro-health-qa-runs"), label="runs repo", full_width=True)
    mo.vstack([hf_token_input, data_repo_input, runs_repo_input])
    return data_repo_input, hf_token_input, runs_repo_input


@app.cell
def _(data_repo_input, hf_token_input, mo, os, runs_repo_input):
    if hf_token_input.value.strip():
        os.environ["HF_TOKEN"] = hf_token_input.value.strip()
    DATA_REPO = data_repo_input.value.strip()
    RUNS_REPO = runs_repo_input.value.strip()
    GITHUB_REPO = "OsbornNyakaru/multilingual-health_Q_A"
    # Frozen harness: autoresearch_nlp/prepare.py must hash to this, or the runner refuses to run.
    PREPARE_SHA256 = "532b2c174894b7107129c3d7056613c62c4066e530761193b1518487e784f5dd"
    HELD_OUT_FINGERPRINT = "a8026f24ea3d"

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    try:
        _who = api.whoami()["name"]
        auth_msg = mo.md(f"Hugging Face: signed in as **{_who}**")
    except Exception as _e:
        _who = None
        auth_msg = mo.callout(mo.md(f"Hugging Face: **not signed in** ({type(_e).__name__}). Paste a token above."), kind="danger")
    auth_msg
    return (
        CommitOperationAdd, DATA_REPO, GITHUB_REPO, HELD_OUT_FINGERPRINT, PREPARE_SHA256, RUNS_REPO,
        api, hf_hub_download,
    )


@app.cell
def _(mo):
    def _gpu():
        try:
            import torch

            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                return {"cuda": True, "name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)}
        except Exception:
            pass
        return {"cuda": False, "name": "none", "vram_gb": 0.0}

    gpu_info = _gpu()
    mo.md(
        f"GPU: **{gpu_info['name']}**, {gpu_info['vram_gb']} GB"
        if gpu_info["cuda"]
        else "GPU: **none**. Only CPU experiments (e.g. `tfidf-char` retrieval) will run."
    )
    return (gpu_info,)


@app.cell
def _(DATA_REPO, Path, mo):
    WORK_DIR = Path.cwd() / "runner_work"
    DATA_DIR = WORK_DIR / "data" / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "Train.csv").exists():
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=str(DATA_DIR), allow_patterns=["*.csv"])
        except Exception as _e:
            print(f"data download failed: {type(_e).__name__}: {_e}")
    data_ok = (DATA_DIR / "Train.csv").exists() and (DATA_DIR / "Test.csv").exists()
    mo.md(f"Data: **{'ready' if data_ok else 'missing'}** in `{DATA_DIR}`")
    return DATA_DIR, WORK_DIR, data_ok


@app.cell
def _(
    CommitOperationAdd, GITHUB_REPO, PREPARE_SHA256, RUNS_REPO, WORK_DIR, api, gc, gpu_info, hashlib,
    hf_hub_download, importlib, io, json, pd, sys, tarfile, time, urllib,
):
    # Kernel-lifetime state: survives across queued runs so models and embeddings load once.
    KERNEL = {"model_key": None, "tok": None, "model": None, "embedders": {}, "cache": {}, "splits": {}}

    def fetch_code(sha: str):
        """Download the repo at `sha` (public codeload tarball, immutable) and return its root."""
        dest = WORK_DIR / "code" / sha
        if not (dest / ".ok").exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f"https://codeload.github.com/{GITHUB_REPO}/tar.gz/{sha}", timeout=120) as r:
                        data = r.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(dest, filter="data")
            (dest / ".ok").write_text("ok")
        roots = [p for p in dest.iterdir() if p.is_dir()]
        return roots[0]

    def load_module(path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def check_harness(root):
        p = root / "autoresearch_nlp" / "prepare.py"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != PREPARE_SHA256:
            raise RuntimeError(f"prepare.py hash {h[:12]} != pinned {PREPARE_SHA256[:12]}: harness changed, refusing to run")
        return p

    def free_model():
        KERNEL["tok"] = KERNEL["model"] = KERNEL["model_key"] = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def get_model(model_id, precision="auto", adapter=None):
        key = (model_id, precision, adapter)
        if KERNEL["model_key"] == key:
            return KERNEL["tok"], KERNEL["model"]
        free_model()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        prec = precision if precision != "auto" else ("bf16" if gpu_info["vram_gb"] >= 40 else "4bit")
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if prec == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.generation_config = GenerationConfig(
            do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        model.eval()
        KERNEL.update(model_key=key, tok=tok, model=model)
        return tok, model

    def get_embedder(name: str):
        if name not in KERNEL["embedders"]:
            from sentence_transformers import SentenceTransformer

            KERNEL["embedders"][name] = SentenceTransformer(name, device="cuda" if gpu_info["cuda"] else "cpu")
        return KERNEL["embedders"][name]

    def hf_files() -> list[str]:
        return api.list_repo_files(RUNS_REPO, repo_type="dataset")

    def hf_download(path_in_repo: str) -> str:
        return hf_hub_download(RUNS_REPO, path_in_repo, repo_type="dataset", force_download=True)

    def hf_json(path_in_repo: str) -> dict:
        return json.loads(open(hf_download(path_in_repo), encoding="utf-8").read())

    def hf_put(files: dict, message: str):
        """files: {path_in_repo: str | bytes | DataFrame}"""
        ops = []
        for path, content in files.items():
            if isinstance(content, pd.DataFrame):
                content = content.to_csv(index=False).encode("utf-8")
            elif isinstance(content, str):
                content = content.encode("utf-8")
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))
        api.create_commit(RUNS_REPO, repo_type="dataset", operations=ops, commit_message=message)

    return (
        KERNEL, check_harness, fetch_code, free_model, get_embedder, get_model, hf_download, hf_files,
        hf_json, hf_put, load_module,
    )


@app.cell
def _(DATA_DIR, HELD_OUT_FINGERPRINT, KERNEL, WORK_DIR, pd):
    def build_sets(prepare, spec: dict) -> dict:
        """{set_name: (eval_df, pool_df)} for the spec's eval block. Pools never contain eval rows."""
        prepare.DATA_DIR = DATA_DIR
        prepare.CACHE_DIR = WORK_DIR / "processed"
        if "held_out" not in KERNEL["splits"]:
            s = prepare.make_splits()
            fp = prepare._fingerprint(s["held_out"].astype(str))
            if fp != HELD_OUT_FINGERPRINT:
                raise RuntimeError(f"held-out fingerprint {fp} != {HELD_OUT_FINGERPRINT}: split drifted")
            KERNEL["splits"] = {**s, **prepare.load_raw()}
        sp = KERNEL["splits"]
        sets = {}
        subsets = spec.get("subsets")
        for name, rows in spec.get("eval", {"held_out": 0}).items():
            if name == "held_out":
                ev, pool = sp["held_out"], sp["work_train"]
            elif name == "val":
                ev, pool = sp["val"], sp["train"]
            elif name == "test":
                ev, pool = sp["test"], pd.concat([sp["train"], sp["val"]], ignore_index=True)
            else:
                raise ValueError(f"unknown eval set {name!r}")
            if subsets:
                ev = ev[ev["subset"].isin(subsets)]
            rows = int(rows or 0)
            if rows and rows < len(ev):
                frac = rows / len(ev)
                ev = pd.concat(
                    [g.sample(max(1, int(round(len(g) * frac))), random_state=7) for _, g in ev.groupby("subset")]
                )
            sets[name] = (ev.reset_index(drop=True), pool.reset_index(drop=True))
        return sets

    return (build_sets,)


@app.cell
def _(
    KERNEL, build_sets, check_harness, datetime, fetch_code, free_model, get_embedder, get_model,
    gpu_info, hf_download, hf_files, hf_json, hf_put, json, load_module, mo, pd, time, timezone, traceback,
):
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _versions():
        out = {}
        for m in ("torch", "transformers", "sentence_transformers", "peft", "pandas", "marimo"):
            try:
                out[m] = __import__(m).__version__
            except Exception:
                pass
        return out

    def pending_runs() -> list[dict]:
        files = hf_files()
        done = {f.split("/")[1] for f in files if f.startswith("runs/") and f.endswith("/result.json")}
        specs = [hf_json(f) for f in files if f.startswith("queue/") and f.endswith(".json")]
        todo = [s for s in specs if s["run_id"] not in done]
        return sorted(todo, key=lambda s: (s["config"].get("model_id", ""), s.get("created", "")))

    def run_one(spec: dict, log) -> dict:
        rid = spec["run_id"]
        base = f"runs/{rid}"
        started, t0 = _now(), time.time()
        deadline = t0 + 60 * float(spec.get("max_minutes", 120))
        result = {"run_id": rid, "git_sha": spec["git_sha"], "started": started, "gpu": gpu_info, "sets": {}}
        log_lines: list[str] = []

        def _log(msg, progress=None):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            log_lines.append(line)
            log(line)

        try:
            root = fetch_code(spec["git_sha"])
            prep_path = check_harness(root)
            tag = spec["git_sha"][:10]
            prepare = load_module(prep_path, f"prepare_{tag}")
            experiment = load_module(root / "autoresearch_nlp" / "experiment.py", f"experiment_{tag}")
            files = hf_files()
            for set_name, (eval_df, pool_df) in build_sets(prepare, spec).items():
                partial = f"{base}/partial_{set_name}.csv"
                resume = {}
                if partial in files:
                    _p = pd.read_csv(hf_download(partial), dtype=str).fillna("")
                    resume = dict(zip(_p["ID"], _p["answer"]))
                    _log(f"{set_name}: resuming {len(resume):,} answers")
                last_save = {"t": 0.0}

                class Ctx:
                    cache = KERNEL["cache"]

                    def __init__(self):
                        self.resume = resume

                    @staticmethod
                    def get_model(model_id, precision="auto", adapter=None):
                        return get_model(model_id, precision, adapter)

                    @staticmethod
                    def get_embedder(name):
                        return get_embedder(name)

                    @staticmethod
                    def save(answers):
                        if time.time() - last_save["t"] > 300:
                            hf_put({partial: pd.DataFrame({"ID": list(answers), "answer": list(answers.values())})},
                                   f"{rid}: checkpoint {set_name} ({len(answers)})")
                            last_save["t"] = time.time()

                    @staticmethod
                    def should_stop():
                        return time.time() > deadline

                    @staticmethod
                    def log(msg, progress=None):
                        _log(f"{set_name}: {msg}")

                _log(f"{set_name}: {len(eval_df):,} rows, pool {len(pool_df):,}")
                ts = time.time()
                answers, meta = experiment.run(spec["config"], eval_df.drop(columns=["output"], errors="ignore"), pool_df, Ctx())
                missing = [i for i in eval_df["ID"] if i not in answers]
                preds = eval_df[["ID", "subset"]].copy()
                preds["pred"] = [answers.get(i, "") for i in preds["ID"]]
                for col in sorted({k for m in meta.values() for k in m}):
                    preds[col] = [meta.get(i, {}).get(col) for i in preds["ID"]]
                entry = {"n": len(eval_df), "missing": len(missing), "seconds": round(time.time() - ts, 1)}
                if "output" in eval_df.columns and not missing:
                    sc = prepare.score(preds["pred"].tolist(), eval_df["output"].tolist())
                    ps = prepare.score_per_subset(preds.assign(output=eval_df["output"].values), "pred")
                    entry.update(rouge1_f1=sc.rouge1_f1, rougeL_f1=sc.rougeL_f1, combined=sc.combined,
                                 per_subset=ps.reset_index().to_dict(orient="records"))
                    _log(f"{set_name}: combined {sc.combined:.4f} (R1 {sc.rouge1_f1:.4f}, RL {sc.rougeL_f1:.4f})")
                result["sets"][set_name] = entry
                hf_put({f"{base}/{set_name}_preds.csv": preds}, f"{rid}: {set_name} predictions")
                if missing:
                    result["status"] = "timeout" if time.time() > deadline else "incomplete"
                    break
            result.setdefault("status", "ok")
        except Exception as e:
            result["status"] = "crash"
            result["error"] = "".join(traceback.format_exception(e))[-4000:]
            _log(f"CRASH: {type(e).__name__}: {e}")
            if "out of memory" in str(e).lower():
                free_model()
        result.update(finished=_now(), seconds=round(time.time() - t0, 1), versions=_versions())
        hf_put({f"{base}/result.json": json.dumps(result, indent=2, default=str), f"{base}/log.txt": "\n".join(log_lines[-300:])},
               f"{rid}: {result['status']}")
        return result

    def run_queue(watch: bool = False, idle_minutes: float = 30, max_runs: int = 100, log=print) -> list[dict]:
        hf_put({"_runner/ping.json": json.dumps({"at": _now(), "gpu": gpu_info})}, "runner ping")
        results, idle_since = [], time.time()
        while len(results) < max_runs:
            todo = pending_runs()
            if not todo:
                if not watch or time.time() - idle_since > 60 * idle_minutes:
                    break
                time.sleep(60)
                continue
            spec = todo[0]
            log(f"▶ {spec['run_id']}  ({len(todo)} pending)")
            results.append(run_one(spec, log))
            log(f"■ {spec['run_id']}: {results[-1]['status']}")
            idle_since = time.time()
        return results

    mo.md("Runner ready.")
    return pending_runs, run_queue


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run queue")
    watch_switch = mo.ui.switch(value=True, label="keep watching for new specs (poll every 60 s)")
    idle_input = mo.ui.number(start=5, stop=600, step=5, value=30, label="stop after N idle minutes")
    mo.hstack([run_btn, watch_switch, idle_input], justify="start", wrap=True)
    return idle_input, run_btn, watch_switch


@app.cell
def _(data_ok, idle_input, mo, run_btn, run_queue, watch_switch):
    mo.stop(not run_btn.value, mo.md("_Press **Run queue** to start._"))
    mo.stop(not data_ok, mo.callout(mo.md("Data is missing. Check the token and data repo above."), kind="danger"))
    _lines: list[str] = []

    def _show(line):
        _lines.append(line)
        mo.output.replace(mo.md("```\n" + "\n".join(_lines[-40:]) + "\n```"))

    _results = run_queue(watch=watch_switch.value, idle_minutes=idle_input.value, log=_show)
    mo.md(
        f"Done: {len(_results)} run(s). "
        + ", ".join(f"`{r['run_id']}` {r['status']}" for r in _results)
        + ". Now run `python scripts/exp.py pull` locally."
    )
    return


if __name__ == "__main__":
    app.run()
