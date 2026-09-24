---
tags: [molab, marimo, handoff, experiments]
updated: 2026-09-18
---
# Molab Notebook Plan

`notebooks/molab_afro_health_qa.py` is the single interactive notebook intended for Molab. It is a **practice-research workspace**, not a finished competition submission. It mirrors the frozen local harness: the 7% stratified Train holdout (seed 1234), whitespace-token ROUGE, and the 4-column submission schema.

## Safe operating sequence

1. Push the notebook and its supporting code to GitHub, then import the notebook into Molab by URL.
2. Upload `Train.csv`, `Val.csv`, `Test.csv`, and `SampleSubmission.csv` to `data/raw/`, or configure the private HF dataset variables described in [[File Organisation]].
3. Open the notebook in `dry_run` mode and inspect the data and harness-parity cells. No evaluation or submission happens until its explicit buttons are pressed.
4. Set one hypothesis in Config: model, few-shot setting, decoding setting, or adapter. Do not combine levers.
5. Click **Evaluate held-out**. Read the overall and per-subset ROUGE results before considering test inference.
6. Record the measured run in `results.tsv` and [[Experiment History]]. Only then click **Generate test submission**.
7. Before the 12-hour Molab limit, persist the submission and any adapter to the configured private HF dataset or download them.

## What the notebook does

- Locates the four competition CSVs or downloads them from a private Hugging Face dataset.
- Reproduces the `autoresearch_nlp/prepare.py` split and checks it against the repo copy when available.
- Builds per-language prompts and a deterministic few-shot bank drawn only from `work_train`.
- Runs batched, subset-specific generation with resumable checkpoints.
- Writes a schema-validated submission and JSON sidecar only after the user explicitly asks it to.
- Offers optional LoRA training on `work_train` plus `Val.csv`, never on the held-out slice.

## Handoff status

The notebook passed Python compilation and Marimo structural validation after the 2026-09-18 handoff repair. It has **not** completed a GPU-backed generation run, selected a license-cleared model, or recorded a public leaderboard score. Treat those as decisions/experiments still to make, not as completed work.

Related: [[Marimo MCP and Pairing]], [[Molab Platform]], [[Autoresearch Harness]], [[Open Questions]].
