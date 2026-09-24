---
tags: [code, experiments]
---
# Autoresearch Harness (`autoresearch_nlp/`)

A Karpathy-style autoresearch loop adapted for generative NLP. Dated 2026-06-07. This is the cleanest, most recent codified pipeline and the thing we port to [[Molab Platform]].

## Files
- `prepare.py` — **frozen** harness. Already configured for this competition: `ID/input/output/subset`, 4-col submission, weights 0.37/0.37/0.26 ([[Scoring Metric]]). Provides `load_raw()`, `make_splits()` (7% stratified held-out, seed 1234), `score()`, `score_per_subset()`, `report()`, `build_submission()`, `validate_submission()`.
  - `DATA_DIR = Path("../data/raw")` and `CACHE_DIR = Path("data/processed")` — relative to `autoresearch_nlp/`. Loads `Train, Test, SampleSubmission, Val` if present.
- `train.py` — the one editable file. CONFIG block: `RUN_NAME`, `DRY_RUN`, `MODEL_ID`, `ADAPTER_DIR`, `LOAD_IN_4BIT`, beams=4, no_repeat_ngram=3, per-language system prompts and templates, **per-subset length bounds** (from `tools/length_calibrate.py`). `generate()` batches by subset so bounds apply. `main()` scores held-out, then writes `submissions/<RUN_NAME>.csv`.
- `tools/length_calibrate.py`, `tools/synth_sanity.py` (CPU smoke test).
- `program.md` (the loop), `RULES.md`, `PREFLIGHT.md`, `AGENT_PLAYBOOK.md`, `LESSONS.md`, `MODEL_DECISION.md`, `COMPETITION_INTEL.md`.
- `results.tsv` — header only; no GPU run was ever logged through this harness.

## The loop (program.md)
One hypothesis, one lever, commit, run `python train.py > run.log`, grep `combined:`, read per-subset block, append results.tsv, keep if > +0.003 else revert.

## Status
`DRY_RUN = True` and `MODEL_ID = CohereLabs/aya-expanse-8b` (flagged as a bad fit, see [[Model Decisions]]). Never ran for real because the T4 could not hold an 8B model comfortably. Molab's 96 GB GPU removes that blocker.

Related: [[Molab Notebook Plan]], [[Submission Format]].
