# afro-health-qa

Chiromo Forge's entry for the [Zindi Multilingual Health Question Answering in Low-Resource African Languages Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge). Build a seq2seq model that answers maternal, sexual, and reproductive health questions in Luganda, Kiswahili, Akan/Twi, and Amharic.

**Window:** 2026-04-24 → 2026-07-14. **Goal:** top 3 on private leaderboard.

## Setup (clone → baseline in five commands)

```bash
git clone <this-repo> afro-health-qa && cd afro-health-qa
cp .env.example .env          # then fill in HF_TOKEN + WANDB_API_KEY
make setup                    # venv + pinned deps + editable install
bash scripts/download_data.sh # after 2026-04-24 data drop
make baseline                 # zero-shot Aya-Expanse-8B → first submission
```

## Directory guide

| Path | Purpose |
|------|---------|
| `configs/` | YAML configs: models, training regimes, decoding strategies. Hydra-composable. |
| `src/afro_health_qa/` | All production code. Importable as `afro_health_qa`. |
| `notebooks/` | Numbered notebooks; `99_final_submission.ipynb` is the Zindi reviewer entry point. |
| `scripts/` | Shell scripts + Python CLIs (baseline run, final selection, repro check). |
| `experiments/` | `LOG.md`, `HYPOTHESES.md`, `ABLATIONS.md` — the competition paper trail. |
| `data/` | Gitignored. Raw/processed/augmented competition data lives here. |
| `models/` | Gitignored. Fine-tuned QLoRA adapters + cached base weights. |
| `submissions/` | Tracked. Every CSV has a sidecar JSON with run metadata. |
| `docs/` | Playbook, scoring notes, ADRs, known issues. |
| `tests/` | Pytest suite — seeding, submission format, data split, prompt round-trip, end-to-end repro. |

## Workflow

Follow [docs/PLAYBOOK.md](docs/PLAYBOOK.md) — an 11-week plan broken into phases (bootstrap, baseline, augmentation, fine-tuning, ensembling, finalisation). Each experiment follows:

1. Write the hypothesis in `experiments/HYPOTHESES.md`.
2. Run training / inference with a config pinned in `configs/`.
3. Score locally with `make evaluate RUN=<predictions>`.
4. Append a row to `experiments/LOG.md` before you submit.
5. `make submit RUN=<predictions>` — writes validated CSV to `submissions/`.

## Submission strategy

50 total submissions, 5/day. Never submit without a logged hypothesis. Final two submissions are selected manually via:

```bash
python scripts/select_final.py
```

Prints a decision matrix ranking every submission by public leaderboard, held-out local score, and per-language balance. Picks one "best public" and one "best held-out" to hedge public→private shakeup. See [docs/DECISIONS.md](docs/DECISIONS.md) for the rationale.

## Reproducibility

Zindi audits the top-10 finishers' code. This repo enforces:

- `src/afro_health_qa/seeding.py` — five-place seed (random, numpy, torch, torch.cuda, PYTHONHASHSEED).
- All deps pinned with `==` in `requirements.txt` and `pyproject.toml`.
- `scripts/verify_reproducibility.py` — runs the full pipeline from a fresh checkpoint and hashes the output.
- `tests/test_reproducibility.py` — end-to-end CPU-only path on a tiny model must produce bit-identical submissions across two runs.
- `requirements-submission.txt` — the minimal set that the final notebook installs at the top of `99_final_submission.ipynb` (no custom local packages).

## Team

- **Osborn Nyakaru** — lead (Chiromo Forge founder, CS@Chuka).
- **Vera Nyagaka** — CTO (TBC: joining from week _?_).
- **Grace Ngari** — ML Engineer (TBC: joining from week _?_).

Fill in roles by day 3.

## Licence

- **Code:** Apache-2.0.
- **Any data shipped in the repo:** CC-BY-SA-4.0, per Zindi competition rules.
- No model weights ship with this repo; download via `scripts/download_data.sh` and Hugging Face Hub.

## Compute plan

| Weeks | Machine | Notes |
|-------|---------|-------|
| 1–3 | Colab free (T4, 12h) | Baseline, tokeniser audit, small QLoRA runs with grad accumulation. |
| 3–6 | Colab Pro (A100) | Full QLoRA runs, decoding sweeps. Fallback: Kaggle T4 x2, Lightning Studios. |
| 6–end | AWS g5.2xlarge (A10G) if Activate credits land | Ensemble + long training. |

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for memory caveats per machine.
