# 11-Week Playbook

The condensed plan. Dates assume 2026-04-24 start, 2026-07-14 end.

## Phase 0 — Bootstrap (2026-04-24, day 0)

Already done by this scaffold. On data-drop day, run:

```bash
cp .env.example .env                   # fill in HF_TOKEN + WANDB_API_KEY
bash scripts/setup_env.sh              # install pinned deps, editable package
bash scripts/download_data.sh          # downloads lid.176.bin; reminds re: Zindi CSVs
jupyter lab notebooks/00_data_exploration.ipynb   # run top-to-bottom, produces data_report.md
```

Exit gate: `make test` green; `data/raw/Train.csv`, `Test.csv`, `SampleSubmission.csv` present; language ID + prompt tests pass.

## Phase 1 — Baseline (Week 1, 2026-04-24 → 2026-05-01)

- Run `notebooks/01_tokeniser_audit.ipynb` — confirms per-language token budgets.
- Run `notebooks/02_baseline_zeroshot.ipynb` — zero-shot Aya, greedy decoding, first submission.
- Log as `exp001`. Expected: combined ~0.30–0.40. If < 0.25, prompt or decoding is broken.
- Start `docs/DATA_REPORT.md` from notebook 00 outputs.

## Phase 2 — First fine-tune (Week 2, 2026-05-01 → 2026-05-08)

- `notebooks/10_qlora_training.ipynb` with `configs/training/qlora_default.yaml`.
- Submit once (greedy decoding) — `exp002`. Expected lift: +0.05 combined.
- Run per-language breakdown. If one language is off, note in `KNOWN_ISSUES.md`.

## Phase 3 — Decoding sweep (Week 3, 2026-05-08 → 2026-05-15)

- `notebooks/20_decoding_experiments.ipynb`. Sweep length_penalty ∈ {0.8, 1.0, 1.2}, num_beams ∈ {3, 5, 8}.
- At most 2 submissions for decoding (budget-aware). Log all sweeps locally.
- Upgrade to Colab Pro (or Kaggle T4×2 fallback — see `docs/KNOWN_ISSUES.md`).

## Phase 4 — Data augmentation (Weeks 4–5)

- Build `sunbird_salt` and `ghananlp_khaya` loaders. Enable in `configs/data.yaml`.
- Run `medmcqa_translated` pipeline once — ~5K rows per language.
- Ablate: comp-only vs. comp+each-source vs. all. Rule: keep sources that don't hurt per-language min.

## Phase 5 — Reranker + ensemble (Weeks 6–7)

- Turn on `beam_rerank.yaml` config. Verify reranker flips answer on ≥ 60% of rows.
- Train a second adapter on Gemma-2-9B. Ensemble via AfroLM-reranker.
- AWS credits? → switch training to g5.2xlarge.

## Phase 6 — Polish (Weeks 8–10)

- Finalise prompt templates (native-language labels confirmed best? if so lock them).
- Full judge run on top 3 candidates.
- Rehearse `99_final_submission.ipynb` on a fresh machine; fix any env issues.

## Phase 7 — Final selection (Week 11, 2026-07-08 → 2026-07-14)

- `python scripts/select_final.py` — review the decision matrix daily.
- Submit the two chosen entries at least 24h before close (2026-07-13 latest).
- Confirm selection on Zindi UI — **do not let auto-pick decide for you**.

## Submission budget pacing

50 total submissions over 11 weeks ≈ 4.5/week. Suggested profile:

| Weeks | Target submissions |
|-------|---------------------|
| 1     | 1 (baseline)        |
| 2     | 2 (first fine-tune, sanity) |
| 3     | 3 (decoding sweep)  |
| 4–5   | 8 (aug ablations)   |
| 6–7   | 10 (reranker, ensemble) |
| 8–10  | 18 (polish, per-language targeted) |
| 11    | 8 (held in reserve + final two) |

Reserve ≥ 8 for the last week. Do not front-load.
