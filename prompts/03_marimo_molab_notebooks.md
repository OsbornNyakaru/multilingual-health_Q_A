# 03 — marimo Notebooks for molab (GitHub → molab workflow)

Run after `00` and `01`. Can run in parallel with `02` Step 1 — the
notebooks call the same package code that `02` writes. Read
`vault/00_INDEX.md` and `vault/facts/repo-state.md` first.

## How our workflow actually runs (design around this)

- We edit code **locally** in this repo, push to GitHub
  (`OsbornNyakaru/multilingual-health_Q_A`, branch `main`).
- We run on **molab** (molab.marimo.io), which opens notebooks straight
  from GitHub and **re-syncs when we push**:
  `https://molab.marimo.io/github/OsbornNyakaru/multilingual-health_Q_A/blob/main/notebooks/<file>.py`
- molab compute: 4 CPU, 32 GiB RAM, optional **RTX Pro 6000 Blackwell,
  96 GB VRAM**. All GPU work (embedding, fine-tuning, vLLM) happens here,
  not locally.
- **molab storage is not durable.** Only files uploaded through the
  sidebar, or cached with `mo.persistent_cache`, persist. Anything else
  (checkpoints, RAG CSVs, predictions) must be pushed somewhere durable —
  use **Hugging Face Hub** (private dataset repo for data/artifacts,
  private model repo for LoRA adapters). `HF_TOKEN` comes from an env var
  or a `mo.ui.text(kind="password")` field — never hard-coded, never
  committed.

## Deliverables

Create `notebooks/` with **one notebook per pipeline stage** (not one giant
notebook — marimo is reactive, and we never want a cell edit to trigger a
re-train):

| notebook | does | GPU |
|---|---|---|
| `00_setup_data.py` | pull Zindi CSVs (sidebar upload or private HF dataset), validate schema, show per-subset counts | no |
| `01_retrieval.py` | BGE-M3 embeddings (cached), retrieval-only baseline, RAG dataset build for k=1..3, push RAG CSVs to HF | yes (small) |
| `02_train.py` | LoRA fine-tune with Unsloth, checkpoints pushed to HF Hub every N steps | yes |
| `03_infer_eval.py` | vLLM inference with an adapter from HF, local metric (overall + per subset), submission CSV | yes |
| `04_dashboard.py` | reads `vault/experiments/*.md` frontmatter + scores, charts progress over experiments | no |

## Rules for every notebook

1. **Pure marimo `.py` format** — no `.ipynb`. Generate with
   `marimo edit` conventions (`app = marimo.App()`, `@app.cell` functions),
   and check each file with `marimo check` before pushing.
2. **Inline dependencies (PEP 723)** at the top of each file so molab
   installs them in a sandbox. Include our own package from GitHub:
   ```python
   # /// script
   # requires-python = ">=3.11"
   # dependencies = [
   #   "marimo",
   #   "afro-health-qa @ git+https://github.com/OsbornNyakaru/multilingual-health_Q_A@main",
   #   ...
   # ]
   # ///
   ```
   Verify this installs from `pyproject.toml` correctly. Because molab may
   cache the `@main` install, add a small cell with a text field for a
   branch or commit SHA and a button that force-reinstalls the package
   from it, then shows `afro_health_qa.__version__` + the git SHA. Always
   log the SHA into the experiment note.
3. **Thin notebooks, fat package.** Notebooks only wire config + UI +
   display. All logic (retrieval, prompt building, training, inference,
   scoring) lives in `src/afro_health_qa/` so it's testable with pytest
   and shared across notebooks.
4. **Gate expensive cells** behind `mo.ui.run_button()` and `mo.stop()` so
   nothing trains or generates just because the notebook opened or an
   upstream cell changed.
5. **Config via UI → Hydra.** Expose the knobs we sweep (model name, k,
   LoRA r/alpha/dropout, lr, epochs, decoding) as `mo.ui` elements whose
   values build a Hydra override list, e.g. `["model=sunflower32b", "rag.k=3"]`.
   The same overrides must also work from the CLI.
6. **Seeds everywhere** — set and display the seed in every notebook.
7. **Survive session loss.** Training pushes the adapter to HF Hub at
   every checkpoint and supports resume from the latest checkpoint.
   Embedding caches use `mo.persistent_cache`.
8. **Write back to the vault.** `03_infer_eval.py` ends with a cell that
   renders a ready-to-paste EXP note (frontmatter + config + git SHA +
   overall and per-subset scores) and offers it as a download. The local
   agent commits it into `vault/experiments/` after the run. The
   notebooks never write to GitHub directly.

## Version and environment risks to check (don't assume)

- Blackwell (sm_120) needs recent CUDA wheels. Confirm the torch, vLLM,
  Unsloth and flash-attn versions all support it **together**. Record the
  working pin set in `vault/decisions/D-00X-molab-env-pins.md`.
- vLLM and Unsloth training in the same process on one GPU compete for
  memory. Keep them in separate notebooks (as above) and restart the
  kernel between stages.
- Check whether molab allows long-running training cells without an idle
  timeout. If it doesn't, document a workaround (e.g. shorter runs with
  resume) in a decision note.

## Done when

- [ ] All five notebooks pass `marimo check` and open in molab from their
      GitHub URLs.
- [ ] `00_setup_data.py` and `01_retrieval.py` run end to end on molab and
      reproduce EXP-001 (retrieval-only) scores within noise.
- [ ] A README section in `notebooks/README.md` lists the molab URL for
      each notebook and the run order.
