# archive/

This folder holds files that are no longer part of the working project. They are kept for history, and `git log --follow <path>` still works on them. Nothing here is maintained, so don't import from it or follow its instructions. The one exception: `scripts/build_rag_dataset.py` still reads the prompt constants from `archive/autoresearch_nlp/train.py`.

Each entry gives what the item was, why it was archived, and what replaced it. Moved on 2026-10-01.

## docs/
- `docs/STEP_BY_STEP_GUIDE.md`: guide to running exp001 on Kaggle. Archived because Kaggle is no longer used. Replaced by `docs/molab_runbook.md`.
- `docs/PLAYBOOK.md`: the original 11-week plan (April to July 2026). Archived because the competition is closed. Replaced by `vault/00_INDEX.md` and `autoresearch_nlp/program.md`.
- `docs/SPRINT_PLAN.md`: the final-push sprint to the 2026-06-21 deadline. Archived because the deadline has passed. Replaced by `vault/00_INDEX.md` (current goal).
- `docs/ROADMAP_TOP5.md`: the practice-submission roadmap to top 5. Archived because its contents were migrated into vault hypotheses and decisions (H-011, H-012, D-003). Replaced by `vault/00_INDEX.md`.
- `docs/competition_report/` (the PDF and `figures/`): the June 2026 competition report and its charts. Archived because it is a one-off snapshot whose figures predate the whitespace-tokenizer re-score. Replaced by the vault experiment notes. Its `data/` subfolder stayed at `docs/competition_report/data/`.

## notebooks/
- `notebooks/exp001_full_pipeline.py`, `exp001_kaggle_part1.py`, `exp001_kaggle_part2.py`: the May 2026 Aya-Expanse Kaggle pipeline (EXP-001, which ran out of memory). Archived because the model was rejected (D-002) and the platform changed. Replaced by `notebooks/molab_afro_health_qa.py`.
- `notebooks/exp002_few_shot_colab.py`, `exp002_kaggle_or_colab.ipynb`: AfriqueLlama few-shot runs on Colab (EXP-002). Archived as historical runs that were never scored. Replaced by `notebooks/molab_afro_health_qa.py`.
- `notebooks/Copy_of_notebook26836f548f.ipynb`: an uploaded copy of a Colab notebook. Archived as historical; its hardcoded HF token was scrubbed (D-006). Replaced by `notebooks/molab_afro_health_qa.py`. Its `.py` duplicate, `notebooks/uploaded_nb_code.py`, was deleted.

## autoresearch_nlp/
- `autoresearch_nlp/PREFLIGHT.md`: the day-1 hard-stop checklist. Archived because every check is done and recorded in vault facts F-001 to F-004. Replaced by `vault/00_INDEX.md`.
- `autoresearch_nlp/MODEL_DECISION.md`: the base-model selection guide. Archived because it was folded into D-002 and D-004 (still open). Replaced by `vault/decisions/D-004-base-model-for-molab.md`.
- `autoresearch_nlp/train.py`: the original single-file experiment script (DRY_RUN stub, Aya default). Archived because the new loop uses experiment specs. Replaced by `autoresearch_nlp/experiment.py` and `notebooks/molab_runner.py`. Its prompts and length bounds are also in `notebooks/molab_afro_health_qa.py`.

## scripts/
- `scripts/select_final.py`: picked the final two submissions. Archived because there are no final picks post-close and practice submissions are unlimited. No replacement.
- `scripts/run_baseline.sh`: the `make baseline` runner. Archived because it was broken (it pointed at a missing notebook). Replaced by molab runs.
- `scripts/verify_reproducibility.py`: the `make verify` hash check. Archived because it was broken (a fixture was missing). Replaced by `tests/`.
- `scripts/run_exp001.py`, `scripts/local_multilingual_qa.py`: May-era local and Kaggle generation scripts. Archived as historical. Replaced by `notebooks/molab_afro_health_qa.py`.
- `scripts/download_data.sh`: the Zindi download script. Archived because the data now comes from the private HF dataset (D-006). Replaced by `HF_DATA_REPO` and `scripts/push_data_to_hf.py`.
- `scripts/setup_env.sh`: environment bootstrap. Archived because the environment is set up by `make setup` locally and by molab remotely. Replaced by `make setup`.
- `scripts/build_report_charts.py`, `scripts/build_report_pdf.py`: built the competition-report PDF and figures. Archived together with the report.

## configs/
- `configs/` (base, data, models, training, decoding YAML): the Hydra-style config tree for the broken `src/.../training` path. Archived because the weights and columns were wrong and every config targeted Aya. Replaced by the experiment specs and notebook settings. `src/afro_health_qa/training/run.py` now points at `archive/configs/`.

## mac_local_execution/
- `mac_local_execution/README.md`: a plan for MLX/AfriqueLlama on a Mac. Archived because no code was ever written. Replaced by molab.

## prompts/
- `prompts/00_orchestrator.md`: the one-time orchestrator prompt that set up the vault, the repo audit and the metric replica. Archived because it was completed (see `vault/facts/repo-state.md` and `metric-replica.md`). Replaced by `vault/00_INDEX.md`.

## experiments/
- `experiments/LOG.md`, `HYPOTHESES.md`, `ABLATIONS.md`: the first repo-side trackers, which hold only one exp000 row. Archived because they were migrated into the vault (EXP-000, H-001 to H-010). Replaced by the vault and `experiments/RESULTS.md`.
- `experiments/data/length_calibration_full_report.csv`, `length_optimization_results.csv`, `length_optimization_summary.csv`: EXP-005 length-truncation results. They hold only aggregate per-subset scores and lengths, no competition text. Archived because they were cluttering the repo root. Summarised in `vault/experiments/EXP-005-length-truncation-ablation.md`.
