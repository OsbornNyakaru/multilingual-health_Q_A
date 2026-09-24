# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24",
#     "pandas>=2.2",
#     "pyarrow",
#     "rouge-score==0.1.2",
#     "huggingface_hub>=0.26",
#     "tqdm",
#     "transformers>=4.46",
#     "accelerate>=1.0",
#     "datasets>=3.0",
#     "peft>=0.13",
#     "sentencepiece",
#     "bitsandbytes>=0.43; platform_system == 'Linux'",
# ]
# ///
"""molab_afro_health_qa.py — one marimo notebook for the whole autoresearch loop.

Runs in three places with the same file:
  * Mac (no GPU):  `.venv-marimo/bin/marimo edit notebooks/molab_afro_health_qa.py --mcp --no-token`
                   -> mode "dry_run" exercises data, splits, scoring, submission writing on CPU.
  * molab (GPU):   import this file by GitHub URL, upload data/raw/*.csv in the sidebar (or set
                   HF_DATA_REPO + HF_TOKEN), pick a GPU mode, run.
  * script:        `python notebooks/molab_afro_health_qa.py` runs every cell top to bottom.

Scoring, splitting and submission logic mirror autoresearch_nlp/prepare.py (the frozen harness):
same held-out seed (1234), same 7% stratified slice, same whitespace-token ROUGE, same
0.37/0.37/0.26 weights. The "Harness parity" cell verifies this whenever the repo is present.

GPU-only libraries (torch, transformers, peft) are imported lazily inside the cells that need them,
so the notebook opens and dry-runs on a machine without them. molab auto-installs them on import.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Afro Health QA — molab")


@app.cell
def _():
    import importlib
    import marimo as mo

    return importlib, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Afro Health QA — autoresearch loop on molab

    Zindi *Multilingual Health QA in Low-Resource African Languages*, re-run as practice on a
    96 GB GPU. One experiment = one change in **Config** below, then run top to bottom.

    Order of operations: **Config → Environment → Data → Splits → Held-out eval → Test submission**.
    The LoRA cell is optional and gated behind a button. Everything writes next to this file, so
    on molab the folder layout is `data/raw/`, `data/processed/`, `submissions/`, `models/`, `results.tsv`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Config — change ONE lever per experiment
    """)
    return


@app.cell
def _(mo):
    run_name = mo.ui.text(value="exp010_afriquellama_bf16_fewshot", label="Run name")
    mode = mo.ui.dropdown(
        options=["dry_run", "zero_shot", "few_shot"],
        value="dry_run",
        label="Mode",
    )
    model_id = mo.ui.dropdown(
        options=[
            "McGill-NLP/AfriqueLlama-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2.5-7B-Instruct",
            "CohereLabs/aya-expanse-8b",
        ],
        value="McGill-NLP/AfriqueLlama-8B",
        allow_select_none=False,
        label="Base model (HF id)",
    )
    custom_model_id = mo.ui.text(value="", label="…or type any HF id (overrides dropdown)")
    adapter_dir = mo.ui.text(value="", label="LoRA adapter dir (blank = none)")
    precision = mo.ui.dropdown(
        options=["auto", "bf16", "4bit"], value="auto", label="Precision (auto = bf16 if VRAM ≥ 40 GB)"
    )
    num_beams = mo.ui.number(start=1, stop=8, step=1, value=1, label="num_beams")
    infer_batch = mo.ui.number(start=1, stop=128, step=1, value=32, label="Inference batch size")
    few_shot_k = mo.ui.number(start=0, stop=4, step=1, value=2, label="Few-shot examples per subset")
    few_shot_max_chars = mo.ui.number(start=40, stop=600, step=10, value=150, label="Few-shot answer cap (chars)")
    heldout_limit = mo.ui.number(
        start=0, stop=2088, step=1, value=0, label="Held-out rows to score (0 = all 2,088)"
    )
    checkpoint_every = mo.ui.number(start=1, stop=200, step=1, value=10, label="Checkpoint every N batches")
    hf_push = mo.ui.switch(value=False, label="Push submission + adapter to HF Hub (needs HF_TOKEN, HF_RUNS_REPO)")

    config_ui = mo.vstack(
        [
            mo.hstack([run_name, mode, precision], wrap=True),
            mo.hstack([model_id, custom_model_id], wrap=True),
            mo.hstack([adapter_dir, num_beams, infer_batch], wrap=True),
            mo.hstack([few_shot_k, few_shot_max_chars, heldout_limit, checkpoint_every], wrap=True),
            hf_push,
        ]
    )
    config_ui
    return (
        adapter_dir,
        checkpoint_every,
        custom_model_id,
        few_shot_k,
        few_shot_max_chars,
        heldout_limit,
        hf_push,
        infer_batch,
        mode,
        model_id,
        num_beams,
        precision,
        run_name,
    )


@app.cell
def _(
    adapter_dir,
    custom_model_id,
    few_shot_k,
    mode,
    model_id,
    num_beams,
    precision,
    run_name,
):
    CFG = {
        "run_name": run_name.value.strip() or "unnamed_run",
        "mode": mode.value,
        "model_id": (custom_model_id.value.strip() or model_id.value),
        "adapter_dir": adapter_dir.value.strip() or None,
        "precision": precision.value,
        "num_beams": int(num_beams.value),
        "do_sample": False,
        "no_repeat_ngram": 3,
        "length_penalty": 1.0,
        "max_input_tokens": 512,
        "few_shot_k": int(few_shot_k.value) if mode.value == "few_shot" else 0,
    }
    DRY_RUN = CFG["mode"] == "dry_run"
    return CFG, DRY_RUN


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Environment — where am I, what hardware do I have
    """)
    return


@app.cell
def _(importlib, mo):
    import os
    import platform
    import sys
    from pathlib import Path

    def _find_project_root() -> Path:
        """Repo root if we are inside the repo (Mac), else the notebook's own folder (molab)."""
        here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        for cand in [here, *here.parents, Path.cwd(), *Path.cwd().parents]:
            if (cand / "autoresearch_nlp" / "prepare.py").exists() or (cand / "pyproject.toml").exists():
                return cand
        return here

    PROJECT_ROOT = _find_project_root()
    NB_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    IN_REPO = (PROJECT_ROOT / "autoresearch_nlp" / "prepare.py").exists()
    # On molab there is no repo: everything lives beside the notebook.
    WORK_DIR = PROJECT_ROOT if IN_REPO else NB_DIR

    gpu_info = {"cuda": False, "name": None, "vram_gb": 0.0, "torch": None}
    try:
        _torch = importlib.import_module("torch")

        gpu_info["torch"] = _torch.__version__
        if _torch.cuda.is_available():
            props = _torch.cuda.get_device_properties(0)
            gpu_info.update(cuda=True, name=props.name, vram_gb=round(props.total_memory / 1e9, 1))
    except Exception:  # torch absent on the Mac
        pass

    IS_MOLAB = bool(os.environ.get("MOLAB") or "molab" in os.environ.get("HOSTNAME", "").lower()) or (
        not IN_REPO and gpu_info["cuda"]
    )

    env_md = mo.md(
        f"""
        | | |
        |---|---|
        | Python | `{platform.python_version()}` on `{platform.system()} {platform.machine()}` |
        | Interpreter | `{sys.executable}` |
        | Notebook dir | `{NB_DIR}` |
        | Project root | `{PROJECT_ROOT}` {'(repo detected)' if IN_REPO else '(standalone — molab layout)'} |
        | Work dir | `{WORK_DIR}` |
        | torch | `{gpu_info['torch'] or 'not installed'}` |
        | GPU | {'**' + str(gpu_info['name']) + '**, ' + str(gpu_info['vram_gb']) + ' GB VRAM' if gpu_info['cuda'] else 'none — only `dry_run` will work here'} |
        | Guess | {'**molab**' if IS_MOLAB else 'local machine'} |
        """
    )
    env_md
    return IN_REPO, PROJECT_ROOT, Path, WORK_DIR, gpu_info, os


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Data — find the four Zindi CSVs (or fetch them from a private HF dataset)
    """)
    return


@app.cell
def _(Path, WORK_DIR, mo, os):
    REQUIRED_FILES = ("Train.csv", "Val.csv", "Test.csv", "SampleSubmission.csv")

    def _locate_data_dir() -> Path | None:
        candidates = [
            WORK_DIR / "data" / "raw",
            Path.cwd() / "data" / "raw",
            WORK_DIR / "data",
            Path.cwd() / "data",
            WORK_DIR,
            Path.cwd(),
            WORK_DIR / "raw",
            Path.cwd() / "raw",
        ]
        for c in candidates:
            if (c / "Train.csv").exists() and (c / "Test.csv").exists():
                return c
        return None

    DATA_DIR = _locate_data_dir()
    data_source_note = ""

    if DATA_DIR is None:
        hf_repo = os.environ.get("HF_DATA_REPO", "").strip()
        if hf_repo:
            from huggingface_hub import snapshot_download

            target = WORK_DIR / "data" / "raw"
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=hf_repo,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=["*.csv"],
                token=os.environ.get("HF_TOKEN") or None,
            )
            DATA_DIR = target if (target / "Train.csv").exists() else None
            data_source_note = f"downloaded from HF dataset `{hf_repo}`"
    else:
        data_source_note = "found on disk"

    if DATA_DIR is None:
        data_status = mo.callout(
            mo.md(
                """
                **No data found.** Do one of:

                1. molab sidebar → upload `Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv`
                   into a `data/raw/` folder next to this notebook (a flat upload also works), then re-run this cell.
                2. Set the environment variables `HF_DATA_REPO=<user>/afro-health-qa-data` and `HF_TOKEN`
                   (molab secrets panel) and re-run this cell.
                """
            ),
            kind="danger",
        )
    else:
        missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
        data_status = mo.callout(
            mo.md(
                f"Data dir: `{DATA_DIR}` ({data_source_note})"
                + (f"  \n⚠️ missing optional files: {missing}" if missing else "  \nAll four CSVs present.")
            ),
            kind="success" if not missing else "warn",
        )
    data_status
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, mo):
    import pandas as pd

    mo.stop(DATA_DIR is None, mo.md("_Waiting for data (see above)._"))

    ID_COL, INPUT_COL, OUTPUT_COL, SUBSET_COL = "ID", "input", "output", "subset"
    SUBMISSION_COLUMNS = ("ID", "TargetRLF1", "TargetR1F1", "TargetLLM")

    raw = {}
    for _name in ("Train", "Val", "Test", "SampleSubmission"):
        _p = DATA_DIR / f"{_name}.csv"
        if _p.exists():
            raw[_name.lower()] = pd.read_csv(_p, dtype=str).fillna("")

    train_df, test_df = raw["train"], raw["test"]
    val_df = raw.get("val")

    _counts = (
        train_df[SUBSET_COL]
        .value_counts()
        .rename("train_rows")
        .to_frame()
        .join(test_df[SUBSET_COL].value_counts().rename("test_rows"))
        .join(val_df[SUBSET_COL].value_counts().rename("val_rows") if val_df is not None else None)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "subset"})
    )
    mo.vstack(
        [
            mo.md(
                f"Loaded: train **{len(train_df):,}**, val **{0 if val_df is None else len(val_df):,}**, "
                f"test **{len(test_df):,}** rows. Columns: `{list(train_df.columns)}`"
            ),
            mo.ui.table(_counts, selection=None),
        ]
    )
    return (
        ID_COL,
        INPUT_COL,
        OUTPUT_COL,
        SUBMISSION_COLUMNS,
        SUBSET_COL,
        pd,
        test_df,
        train_df,
        val_df,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Harness — frozen scoring, split and submission rules

    Mirror of `autoresearch_nlp/prepare.py`. Do **not** tune anything here; it is what makes runs comparable.
    """)
    return


@app.cell
def _(ID_COL, OUTPUT_COL, SUBMISSION_COLUMNS, SUBSET_COL, pd):
    import hashlib
    from dataclasses import dataclass

    METRIC_WEIGHTS = {"rouge1_f1": 0.37, "rougeL_f1": 0.37, "judge": 0.26}
    HELDOUT_FRACTION = 0.07
    HELDOUT_SEED = 1234

    class _WhitespaceTokenizer:
        """Language-agnostic tokenizer; the default one strips Amharic Ge'ez to nothing."""

        def tokenize(self, text):
            return str(text).strip().split() if text else []

    def _rouge_scorer():
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=_WhitespaceTokenizer(), use_stemmer=False)

    @dataclass
    class Score:
        rouge1_f1: float
        rougeL_f1: float
        judge: float
        combined: float
        n: int

        def greppable(self) -> str:
            return (
                f"combined: {self.combined:.6f}\nrouge1_f1: {self.rouge1_f1:.6f}\n"
                f"rougeL_f1: {self.rougeL_f1:.6f}\njudge: {self.judge:.6f}\nn: {self.n}"
            )

    def score(predictions, references, judge_scores=None) -> Score:
        if len(predictions) != len(references):
            raise ValueError(f"len(pred)={len(predictions)} != len(ref)={len(references)}")
        scorer = _rouge_scorer()
        r1 = rl = 0.0
        for p, r in zip(predictions, references):
            s = scorer.score(str(r), str(p))
            r1 += s["rouge1"].fmeasure
            rl += s["rougeL"].fmeasure
        n = max(1, len(predictions))
        r1, rl = r1 / n, rl / n
        judge = float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0
        combined = METRIC_WEIGHTS["rouge1_f1"] * r1 + METRIC_WEIGHTS["rougeL_f1"] * rl + METRIC_WEIGHTS["judge"] * judge
        return Score(r1, rl, judge, combined, len(predictions))

    def score_per_subset(df, pred_col, ref_col=OUTPUT_COL, subset_col=SUBSET_COL):
        rows = []
        groups = [("ALL", df)] + (list(df.groupby(subset_col)) if subset_col in df.columns else [])
        for name, g in groups:
            s = score(g[pred_col].tolist(), g[ref_col].tolist())
            rows.append(
                {
                    "subset": name,
                    "n": s.n,
                    "rouge1_f1": round(s.rouge1_f1, 4),
                    "rougeL_f1": round(s.rougeL_f1, 4),
                    "combined_no_judge": round(s.combined, 4),
                }
            )
        return pd.DataFrame(rows).set_index("subset")

    def make_splits(train: pd.DataFrame):
        held_parts, work_parts = [], []
        for _, grp in train.groupby(SUBSET_COL):
            g = grp.sample(frac=1.0, random_state=HELDOUT_SEED)
            n_held = max(1, int(round(len(g) * HELDOUT_FRACTION)))
            held_parts.append(g.iloc[:n_held])
            work_parts.append(g.iloc[n_held:])
        return pd.concat(work_parts).sort_index(), pd.concat(held_parts).sort_index()

    def build_submission(ids, answers) -> pd.DataFrame:
        if len(ids) != len(answers):
            raise ValueError("ids/answers length mismatch")
        answers = ["" if a is None else str(a) for a in answers]
        df = pd.DataFrame({SUBMISSION_COLUMNS[0]: [str(i) for i in ids]})
        for col in SUBMISSION_COLUMNS[1:]:
            df[col] = answers
        return df[list(SUBMISSION_COLUMNS)]

    def validate_submission(df: pd.DataFrame, expected_ids=None) -> None:
        if tuple(df.columns) != tuple(SUBMISSION_COLUMNS):
            raise ValueError(f"columns {tuple(df.columns)} != {tuple(SUBMISSION_COLUMNS)}")
        base = df[SUBMISSION_COLUMNS[1]].astype(str)
        for col in SUBMISSION_COLUMNS[2:]:
            if not df[col].astype(str).equals(base):
                raise ValueError(f"target column {col} differs from {SUBMISSION_COLUMNS[1]}")
        if df[ID_COL].duplicated().any():
            raise ValueError("duplicate IDs")
        if (df[ID_COL].astype(str) == "").any():
            raise ValueError("empty IDs")
        if expected_ids is not None and set(df[ID_COL]) != set(map(str, expected_ids)):
            raise ValueError("submission IDs do not match the expected test IDs")

    def fingerprint(df: pd.DataFrame) -> str:
        return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]

    return (
        build_submission,
        fingerprint,
        make_splits,
        score,
        score_per_subset,
        validate_submission,
    )


@app.cell
def _(SUBSET_COL, WORK_DIR, make_splits, mo, train_df):
    PROCESSED_DIR = WORK_DIR / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    work_train, held_out = make_splits(train_df)
    work_train.to_csv(PROCESSED_DIR / "work_train.csv", index=False, encoding="utf-8")
    held_out.to_csv(PROCESSED_DIR / "held_out.csv", index=False, encoding="utf-8")

    mo.md(
        f"Splits written to `{PROCESSED_DIR}`: work_train **{len(work_train):,}**, held_out **{len(held_out):,}** "
        f"(7 % stratified by `{SUBSET_COL}`, seed 1234). Never train or draw few-shot from held_out."
    )
    return held_out, work_train


@app.cell
def _(IN_REPO, PROJECT_ROOT, fingerprint, held_out, mo):
    # Harness parity: when the repo is present, the held-out slice must be byte-identical to
    # what autoresearch_nlp/prepare.py produced. If this ever fails, one of the two drifted.
    _ref = PROJECT_ROOT / "autoresearch_nlp" / "data" / "processed" / "held_out.csv"
    if IN_REPO and _ref.exists():
        import pandas as _pd

        _ref_df = _pd.read_csv(_ref, dtype=str).fillna("")
        _ok = fingerprint(_ref_df) == fingerprint(held_out.astype(str))
        parity_msg = mo.callout(
            mo.md(
                f"Harness parity with `autoresearch_nlp/prepare.py`: **{'OK' if _ok else 'MISMATCH'}** "
                f"(fingerprint `{fingerprint(held_out.astype(str))}` vs `{fingerprint(_ref_df)}`)"
            ),
            kind="success" if _ok else "danger",
        )
    else:
        parity_msg = mo.md("_Harness parity check skipped (repo not present — expected on molab)._")
    parity_msg
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · Prompting — per-language instructions, few-shot bank, per-subset length bounds
    """)
    return


@app.cell
def _(CFG, INPUT_COL, OUTPUT_COL, SUBSET_COL, few_shot_max_chars, work_train):
    SUBSET_TO_LANG = {
        "Aka_Gha": "aka",
        "Amh_Eth": "amh",
        "Eng_Eth": "eng",
        "Eng_Gha": "eng",
        "Eng_Ken": "eng",
        "Eng_Uga": "eng",
        "Lug_Uga": "lug",
        "Swa_Ken": "swa",
    }
    SYSTEM_INSTRUCTIONS = {
        "eng": "You are an expert health worker. Answer the health question accurately and concisely in English, in the same style as a clinical reference answer. Never refuse to answer.",
        "swa": "Wewe ni mtaalamu wa afya. Jibu swali la afya kwa usahihi na kwa ufupi kwa Kiswahili. Usiseme huwezi kujibu.",
        "lug": "Oli omukugu mu by'obulamu. Ddamu ekibuuzo ky'obulamu mu bujjuvu era mu Luganda. Togamba nti toyinza kuddamu.",
        "aka": "Woyɛ apɔmuden ho nimdefoɔ. Bua apɔmuden asɛmmisa no yiye wɔ Akan kasa mu. Nnka sɛ wontumi mmua so.",
        "amh": "እርስዎ የጤና ባለሙያ ነዎት። ይህን የጤና ጥያቄ በትክክል እና በአጭሩ በአማርኛ ይመልሱ። መመለስ አልችልም አይበሉ።",
    }
    PROMPT_TEMPLATES = {
        "eng": "Question: {question}\nAnswer:",
        "swa": "Swali: {question}\nJibu:",
        "lug": "Ekibuuzo: {question}\nEky'okuddamu:",
        "aka": "Asɛmmisa: {question}\nMmuaeɛ:",
        "amh": "ጥያቄ: {question}\nመልስ:",
    }
    ANSWER_MARKERS = {"eng": "Answer:", "swa": "Jibu:", "lug": "Eky'okuddamu:", "aka": "Mmuaeɛ:", "amh": "መልስ:"}

    # Per-subset new-token bounds from autoresearch_nlp/tools/length_calibrate.py (June 2026).
    LENGTH_BOUNDS = {
        "Aka_Gha": {"min_new_tokens": 35, "max_new_tokens": 266},
        "Amh_Eth": {"min_new_tokens": 11, "max_new_tokens": 58},
        "Eng_Eth": {"min_new_tokens": 15, "max_new_tokens": 65},
        "Eng_Gha": {"min_new_tokens": 42, "max_new_tokens": 184},
        "Eng_Ken": {"min_new_tokens": 30, "max_new_tokens": 226},
        "Eng_Uga": {"min_new_tokens": 26, "max_new_tokens": 273},
        "Lug_Uga": {"min_new_tokens": 22, "max_new_tokens": 226},
        "Swa_Ken": {"min_new_tokens": 29, "max_new_tokens": 246},
    }
    DEFAULT_BOUNDS = {"min_new_tokens": 8, "max_new_tokens": 160}

    REFUSAL_PATTERNS = (
        "i apologize", "i'm sorry", "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i'm unable to", "i am unable to", "as an ai", "as a language model", "i don't understand",
    )

    def lang_of(subset: str) -> str:
        return SUBSET_TO_LANG.get(subset, "eng")

    # Few-shot bank: k examples per subset drawn from work_train ONLY, deterministic, refusal-free,
    # answers capped so prompts stay short (the T4 lesson: prompt length is the throughput lever).
    _k = CFG["few_shot_k"]
    _cap = int(few_shot_max_chars.value)
    FEW_SHOT_BANK: dict[str, list[tuple[str, str]]] = {}
    if _k > 0:
        for _subset, _grp in work_train.groupby(SUBSET_COL):
            _pool = _grp[
                (_grp[OUTPUT_COL].str.len().between(50, 400))
                & (~_grp[OUTPUT_COL].str.lower().str.contains("cannot|sorry|apologize", regex=True))
            ]
            _pool = _pool if len(_pool) >= _k else _grp
            _sample = _pool.sample(min(_k, len(_pool)), random_state=42)
            FEW_SHOT_BANK[_subset] = [
                (str(r[INPUT_COL]).strip()[:300], str(r[OUTPUT_COL]).strip()[:_cap]) for _, r in _sample.iterrows()
            ]

    def build_messages(question: str, subset: str) -> list[dict]:
        """Chat-format messages; rendered with the model's chat template when it has one."""
        lang = lang_of(subset)
        msgs = [{"role": "system", "content": SYSTEM_INSTRUCTIONS[lang]}]
        for ex_q, ex_a in FEW_SHOT_BANK.get(subset, []):
            msgs.append({"role": "user", "content": PROMPT_TEMPLATES[lang].format(question=ex_q)})
            msgs.append({"role": "assistant", "content": ex_a})
        msgs.append({"role": "user", "content": PROMPT_TEMPLATES[lang].format(question=question.strip())})
        return msgs

    def render_prompt(tok, question: str, subset: str) -> str:
        msgs = build_messages(question, subset)
        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        # Plain fallback for base models without a chat template.
        return "\n\n".join(m["content"] for m in msgs)

    def postprocess(text: str, subset: str) -> str:
        text = str(text).strip()
        marker = ANSWER_MARKERS.get(lang_of(subset), "")
        if marker and marker in text:
            text = text.split(marker, 1)[-1].strip()
        if "\n\n" in text:
            text = text.split("\n\n", 1)[0].strip()
        return text

    def is_refusal(text: str) -> bool:
        low = str(text).lower().strip()
        return len(low) < 5 or any(p in low for p in REFUSAL_PATTERNS)

    return (
        DEFAULT_BOUNDS,
        FEW_SHOT_BANK,
        LENGTH_BOUNDS,
        SYSTEM_INSTRUCTIONS,
        build_messages,
        is_refusal,
        postprocess,
        render_prompt,
    )


@app.cell
def _(FEW_SHOT_BANK: dict[str, list[tuple[str, str]]], mo):
    _rows = [
        {"subset": s, "example_q": q[:80] + ("…" if len(q) > 80 else ""), "example_a": a[:80] + ("…" if len(a) > 80 else "")}
        for s, exs in FEW_SHOT_BANK.items()
        for q, a in exs
    ]
    mo.ui.table(_rows, selection=None) if _rows else mo.md("_Few-shot bank empty (mode is not `few_shot`)._")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 · Model — loaded once per session, bf16 on molab, 4-bit only if VRAM is tight
    """)
    return


@app.cell
def _(CFG, DRY_RUN, gpu_info, mo):
    _model_cache: dict = {}

    def _resolve_precision() -> str:
        if CFG["precision"] != "auto":
            return CFG["precision"]
        return "bf16" if gpu_info["vram_gb"] >= 40 else "4bit"

    def load_model():
        """Cached across cell re-runs so a config tweak does not reload 16 GB of weights."""
        key = (CFG["model_id"], CFG["adapter_dir"], _resolve_precision())
        if key in _model_cache:
            return _model_cache[key]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        tok = AutoTokenizer.from_pretrained(CFG["model_id"], padding_side="left", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if key[2] == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], **kwargs)
        if CFG["adapter_dir"]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, CFG["adapter_dir"])
        # Kill sampling params that leak from generation_config.json (the Colab warning spam).
        model.generation_config = GenerationConfig(
            do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        model.eval()
        _model_cache.clear()
        _model_cache[key] = (tok, model)
        return tok, model

    if DRY_RUN:
        model_status = mo.md("_dry_run: no model loaded; a stub generator is used._")
    elif not gpu_info["cuda"]:
        model_status = mo.callout(mo.md("No CUDA GPU here. Switch mode to `dry_run` or run this on molab."), kind="danger")
    else:
        model_status = mo.md(
            f"Ready to load `{CFG['model_id']}` in **{_resolve_precision()}** on {gpu_info['name']} "
            f"({gpu_info['vram_gb']} GB). Loading happens lazily in the first generation cell."
        )
    model_status
    return (load_model,)


@app.cell
def _(mo):
    mo.md("""
    ## 7 · Generation engine — batched by subset, sorted by length, checkpointed
    """)
    return


@app.cell
def _(
    CFG,
    DEFAULT_BOUNDS,
    DRY_RUN,
    ID_COL,
    INPUT_COL,
    LENGTH_BOUNDS,
    SUBSET_COL,
    checkpoint_every,
    infer_batch,
    load_model,
    mo,
    pd,
    postprocess,
    render_prompt,
):
    def generate(df: pd.DataFrame, checkpoint_path=None, label: str = "generate") -> list[str]:
        """Return one answer per row of df (original order). Resumable via checkpoint_path."""
        df = df.reset_index(drop=True)
        if DRY_RUN:
            return ["information"] * len(df)

        import torch

        tok, model = load_model()
        out: dict[int, str] = {}

        if checkpoint_path is not None and checkpoint_path.exists():
            _ck = pd.read_csv(checkpoint_path, dtype=str).fillna("")
            _done = dict(zip(_ck[ID_COL], _ck["answer"]))
            for i, rid in enumerate(df[ID_COL]):
                if rid in _done:
                    out[i] = _done[rid]

        batch_size = int(infer_batch.value)
        every = int(checkpoint_every.value)
        todo = [i for i in range(len(df)) if i not in out]
        prompts = {i: render_prompt(tok, str(df.loc[i, INPUT_COL]), str(df.loc[i, SUBSET_COL])) for i in todo}

        # Group by subset (bounds differ), then sort by prompt length inside each group (tight padding).
        batches: list[tuple[str, list[int]]] = []
        for subset, grp in df.loc[todo].groupby(SUBSET_COL):
            idxs = sorted(grp.index.tolist(), key=lambda i: len(prompts[i]))
            for s in range(0, len(idxs), batch_size):
                batches.append((str(subset), idxs[s : s + batch_size]))

        def _save():
            if checkpoint_path is None:
                return
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {ID_COL: [df.loc[i, ID_COL] for i in sorted(out)], "answer": [out[i] for i in sorted(out)]}
            ).to_csv(checkpoint_path, index=False, encoding="utf-8")

        with mo.status.progress_bar(total=len(batches), title=f"{label}: {len(todo)} rows, {len(batches)} batches") as bar:
            for b, (subset, chunk) in enumerate(batches, 1):
                bounds = LENGTH_BOUNDS.get(subset, DEFAULT_BOUNDS)
                enc = tok(
                    [prompts[i] for i in chunk], return_tensors="pt", padding=True, truncation=True,
                    max_length=CFG["max_input_tokens"],
                ).to(model.device)
                with torch.inference_mode():
                    gen = model.generate(
                        **enc,
                        num_beams=CFG["num_beams"],
                        do_sample=CFG["do_sample"],
                        no_repeat_ngram_size=CFG["no_repeat_ngram"],
                        length_penalty=CFG["length_penalty"],
                        min_new_tokens=bounds["min_new_tokens"],
                        max_new_tokens=bounds["max_new_tokens"],
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                new = gen[:, enc["input_ids"].shape[1] :]
                for i, text in zip(chunk, tok.batch_decode(new, skip_special_tokens=True)):
                    out[i] = postprocess(text, subset)
                if b % every == 0:
                    _save()
                bar.update()
        _save()
        return [out[i] for i in range(len(df))]

    return (generate,)


@app.cell
def _(mo):
    mo.md("""
    ## 8 · Held-out evaluation — the number you optimise (per-subset block is the real signal)
    """)
    return


@app.cell
def _(mo):
    run_evaluation = mo.ui.run_button(label="Evaluate held-out")
    run_evaluation
    return (run_evaluation,)


@app.cell
def _(
    CFG,
    OUTPUT_COL,
    SUBSET_COL,
    WORK_DIR,
    generate,
    held_out,
    heldout_limit,
    is_refusal,
    mo,
    pd,
    postprocess,
    run_evaluation,
    score,
    score_per_subset,
):
    import json
    import time

    mo.stop(not run_evaluation.value, mo.md("_Click **Evaluate held-out** to start an experiment._"))

    _lim = int(heldout_limit.value)
    eval_df = held_out.copy()
    if _lim and _lim < len(eval_df):
        # Stratified subsample so the per-subset block stays meaningful on quick loops.
        eval_df = (
            eval_df.groupby(SUBSET_COL, group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * _lim / len(held_out)))), random_state=7))
            .reset_index(drop=True)
        )
    eval_df = eval_df.reset_index(drop=True)

    _t0 = time.time()
    CKPT_DIR = WORK_DIR / "submissions" / "checkpoints"
    _preds = generate(eval_df, checkpoint_path=CKPT_DIR / f"{CFG['run_name']}_heldout.csv", label="held-out")
    eval_df["pred"] = [postprocess(p, s) for p, s in zip(_preds, eval_df[SUBSET_COL])]
    eval_seconds = round(time.time() - _t0, 1)

    heldout_score = score(eval_df["pred"].tolist(), eval_df[OUTPUT_COL].tolist())
    per_subset = score_per_subset(eval_df, "pred")
    refusal_rate = float(pd.Series([is_refusal(p) for p in eval_df["pred"]]).mean())

    report_block = "\n".join(
        ["---", f"name: {CFG['run_name']}", heldout_score.greppable(), "per_subset:", per_subset.to_string()]
    )
    print(report_block)

    mo.vstack(
        [
            mo.md(
                f"**{CFG['run_name']}** · mode `{CFG['mode']}` · model `{CFG['model_id']}` · "
                f"n={heldout_score.n} · {eval_seconds}s · refusal rate {refusal_rate:.1%}"
            ),
            mo.hstack(
                [
                    mo.stat(f"{heldout_score.combined:.4f}", label="combined (no judge)"),
                    mo.stat(f"{heldout_score.rouge1_f1:.4f}", label="ROUGE-1 F1"),
                    mo.stat(f"{heldout_score.rougeL_f1:.4f}", label="ROUGE-L F1"),
                ]
            ),
            mo.ui.table(per_subset.reset_index(), selection=None),
            mo.accordion({"Greppable report": mo.md(f"```\n{report_block}\n```")}),
            mo.accordion(
                {
                    "Sample predictions": mo.ui.table(
                        eval_df[[SUBSET_COL, "input", OUTPUT_COL, "pred"]].head(20), selection=None
                    )
                }
            ),
        ]
    )
    return heldout_score, json, per_subset, refusal_rate


@app.cell
def _(CFG, WORK_DIR, heldout_score, mo, refusal_rate):
    # results.tsv — the append-only ledger the autoresearch loop reads. Same header as autoresearch_nlp.
    import subprocess
    from datetime import datetime, timezone

    RESULTS_TSV = WORK_DIR / "results.tsv"
    try:
        _commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=WORK_DIR, timeout=5
        ).stdout.strip() or "nogit"
    except Exception:
        _commit = "nogit"
    _row = "\t".join(
        [
            _commit,
            f"{heldout_score.combined:.6f}",
            f"{heldout_score.rouge1_f1:.6f}",
            f"{heldout_score.rougeL_f1:.6f}",
            f"{heldout_score.judge:.6f}",
            "dry_run" if CFG["mode"] == "dry_run" else "ok",
            f"{CFG['run_name']} | {CFG['mode']} | {CFG['model_id']} | beams={CFG['num_beams']} | "
            f"k={CFG['few_shot_k']} | refusal={refusal_rate:.3f} | {datetime.now(timezone.utc).isoformat(timespec='minutes')}",
        ]
    )
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tcombined\trouge1_f1\trougeL_f1\tjudge\tstatus\tdescription\n", encoding="utf-8")
    with RESULTS_TSV.open("a", encoding="utf-8") as _f:
        _f.write(_row + "\n")

    _tail = RESULTS_TSV.read_text(encoding="utf-8").strip().splitlines()[-8:]
    mo.md(f"Appended to `{RESULTS_TSV}`. Last rows:\n\n```\n" + "\n".join(_tail) + "\n```")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9 · Test inference → validated Zindi submission (+ JSON sidecar, optional HF push)
    """)
    return


@app.cell
def _(mo):
    make_submission = mo.ui.run_button(label="Generate test submission")
    make_submission
    return (make_submission,)


@app.cell
def _(
    CFG,
    ID_COL,
    SUBSET_COL,
    WORK_DIR,
    build_submission,
    fingerprint,
    generate,
    heldout_score,
    hf_push,
    importlib,
    json,
    make_submission,
    mo,
    os,
    per_subset,
    postprocess,
    test_df,
    validate_submission,
):
    mo.stop(not make_submission.value, mo.md("_Click **Generate test submission** when held-out looks good._"))

    SUB_DIR = WORK_DIR / "submissions"
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    _raw = generate(test_df, checkpoint_path=SUB_DIR / "checkpoints" / f"{CFG['run_name']}_test.csv", label="test")
    _answers = [postprocess(p, s) for p, s in zip(_raw, test_df[SUBSET_COL])]
    submission = build_submission(test_df[ID_COL].tolist(), _answers)
    validate_submission(submission, expected_ids=test_df[ID_COL].tolist())

    sub_path = SUB_DIR / f"{CFG['run_name']}.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")
    sidecar = {
        "run_name": CFG["run_name"],
        "config": CFG,
        "heldout": {
            "combined": heldout_score.combined,
            "rouge1_f1": heldout_score.rouge1_f1,
            "rougeL_f1": heldout_score.rougeL_f1,
            "n": heldout_score.n,
            "per_subset": per_subset.reset_index().to_dict(orient="records"),
        },
        "fingerprint": fingerprint(submission),
        "rows": len(submission),
    }
    sub_path.with_suffix(".csv.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

    push_note = ""
    if hf_push.value and os.environ.get("HF_RUNS_REPO"):
        _HfApi = importlib.import_module("huggingface_hub").HfApi

        _api = _HfApi(token=os.environ.get("HF_TOKEN") or None)
        _api.create_repo(os.environ["HF_RUNS_REPO"], repo_type="dataset", private=True, exist_ok=True)
        for _p in (sub_path, sub_path.with_suffix(".csv.json")):
            _api.upload_file(
                path_or_fileobj=str(_p), path_in_repo=f"submissions/{_p.name}",
                repo_id=os.environ["HF_RUNS_REPO"], repo_type="dataset",
            )
        push_note = f"  \nPushed to HF dataset `{os.environ['HF_RUNS_REPO']}` under `submissions/`."

    mo.callout(
        mo.md(
            f"Submission written: `{sub_path}` ({len(submission):,} rows, fingerprint `{sidecar['fingerprint']}`). "
            f"Sidecar: `{sub_path.with_suffix('.csv.json').name}`.{push_note}"
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10 · Optional — LoRA supervised fine-tune (96 GB GPU: bf16 base, no quantisation)

    Trains on `work_train` (+ Val). Saves the adapter to `models/<run_name>/`. Then set **LoRA adapter dir**
    in Config to that path and re-run held-out eval. Gated behind the button; it takes ~1–3 h for 1 epoch on 8B.
    """)
    return


@app.cell
def _(mo):
    lora_epochs = mo.ui.number(start=0.1, stop=5, step=0.1, value=1.0, label="epochs")
    lora_rank = mo.ui.number(start=4, stop=128, step=4, value=16, label="LoRA rank")
    lora_lr = mo.ui.text(value="2e-4", label="learning rate")
    lora_bs = mo.ui.number(start=1, stop=64, step=1, value=8, label="per-device batch")
    lora_grad_acc = mo.ui.number(start=1, stop=64, step=1, value=4, label="grad accumulation")
    lora_max_len = mo.ui.number(start=256, stop=4096, step=64, value=1024, label="max seq len")
    lora_use_val = mo.ui.switch(value=True, label="also train on Val.csv")
    train_lora = mo.ui.run_button(label="Train LoRA adapter")
    mo.vstack([mo.hstack([lora_epochs, lora_rank, lora_lr, lora_bs, lora_grad_acc, lora_max_len], wrap=True), lora_use_val, train_lora])
    return (
        lora_bs,
        lora_epochs,
        lora_grad_acc,
        lora_lr,
        lora_max_len,
        lora_rank,
        lora_use_val,
        train_lora,
    )


@app.cell
def _(
    CFG,
    INPUT_COL,
    OUTPUT_COL,
    SUBSET_COL,
    SYSTEM_INSTRUCTIONS,
    WORK_DIR,
    build_messages,
    gpu_info,
    hf_push,
    importlib,
    lora_bs,
    lora_epochs,
    lora_grad_acc,
    lora_lr,
    lora_max_len,
    lora_rank,
    lora_use_val,
    mo,
    os,
    pd,
    train_lora,
    val_df,
    work_train,
):
    mo.stop(not train_lora.value, mo.md("_Press **Train LoRA adapter** to start (GPU only)._"))
    mo.stop(not gpu_info["cuda"], mo.callout(mo.md("LoRA needs a CUDA GPU — run this on molab."), kind="danger"))

    _torch = importlib.import_module("torch")
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    _train = work_train
    if lora_use_val.value and val_df is not None:
        _train = pd.concat([work_train, val_df], ignore_index=True)

    _tok = AutoTokenizer.from_pretrained(CFG["model_id"], trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _tok.padding_side = "right"

    def _to_text(row) -> tuple[str, str]:
        msgs = build_messages(str(row[INPUT_COL]), str(row[SUBSET_COL]))
        if getattr(_tok, "chat_template", None):
            prompt = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(m["content"] for m in msgs)
        return prompt, str(row[OUTPUT_COL]).strip() + (_tok.eos_token or "")

    def _tokenize(batch):
        ids, labels = [], []
        for prompt, answer in zip(batch["prompt"], batch["answer"]):
            p = _tok(prompt, add_special_tokens=False)["input_ids"]
            a = _tok(answer, add_special_tokens=False)["input_ids"]
            x = (p + a)[: int(lora_max_len.value)]
            y = ([-100] * len(p) + a)[: int(lora_max_len.value)]
            ids.append(x)
            labels.append(y)
        return {"input_ids": ids, "labels": labels}

    _pairs = [_to_text(r) for _, r in _train.iterrows()]
    _ds = Dataset.from_dict({"prompt": [p for p, _ in _pairs], "answer": [a for _, a in _pairs]})
    _ds = _ds.map(_tokenize, batched=True, remove_columns=["prompt", "answer"])

    _model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"], dtype=_torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    _model.gradient_checkpointing_enable()
    _model = get_peft_model(
        _model,
        LoraConfig(
            r=int(lora_rank.value), lora_alpha=2 * int(lora_rank.value), lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    _model.print_trainable_parameters()

    ADAPTER_OUT = WORK_DIR / "models" / CFG["run_name"]
    _args = TrainingArguments(
        output_dir=str(ADAPTER_OUT / "trainer"),
        num_train_epochs=float(lora_epochs.value),
        per_device_train_batch_size=int(lora_bs.value),
        gradient_accumulation_steps=int(lora_grad_acc.value),
        learning_rate=float(lora_lr.value),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        seed=42,
        group_by_length=True,
    )
    _trainer = Trainer(
        model=_model, args=_args, train_dataset=_ds,
        data_collator=DataCollatorForSeq2Seq(_tok, padding=True, label_pad_token_id=-100),
    )
    _trainer.train()
    _model.save_pretrained(str(ADAPTER_OUT))
    _tok.save_pretrained(str(ADAPTER_OUT))

    _push = ""
    if hf_push.value and os.environ.get("HF_RUNS_REPO"):
        _HfApi = importlib.import_module("huggingface_hub").HfApi

        _HfApi(token=os.environ.get("HF_TOKEN") or None).upload_folder(
            folder_path=str(ADAPTER_OUT), path_in_repo=f"models/{CFG['run_name']}",
            repo_id=os.environ["HF_RUNS_REPO"], repo_type="dataset",
            ignore_patterns=["trainer/*"],
        )
        _push = f"  \nAdapter pushed to `{os.environ['HF_RUNS_REPO']}/models/{CFG['run_name']}`."

    mo.callout(
        mo.md(
            f"Adapter saved to `{ADAPTER_OUT}` ({len(_ds):,} training rows, {len(SYSTEM_INSTRUCTIONS)} languages).{_push}  \n"
            f"Now paste that path into **LoRA adapter dir** in Config and re-run held-out eval."
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes for the next session

    - Molab kills idle sessions after 90 min and every session at 12 h. Checkpoints live in
      `submissions/checkpoints/` and resume automatically; set `HF_RUNS_REPO` + `HF_TOKEN` and flip the
      HF push switch so nothing is lost.
    - Noise band on held-out is about ±0.005 combined. Only keep a lever if it wins by ≥ 0.003.
    - The knowledge graph for this project is in `vault/` (Obsidian). Log every run in `vault/Experiment History.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
