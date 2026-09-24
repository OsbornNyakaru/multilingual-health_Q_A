---
tags: [repo]
---
# Repo Layout

Workspace: `afro-health-qa/` (git remote `OsbornNyakaru/multilingual-health_Q_A`, branch `main`, single commit `00d444b` as of 2026-09-18).

| Path | Tracked? | What it is |
|------|----------|------------|
| `src/afro_health_qa/` | yes | The "production" package: data loaders, evaluation, inference, submission, training. Written pre-data-drop against the wrong column names (see [[Data Schema]]). Was lost locally by the corruption and restored, see [[Repo Recovery 2026-09-18]]. |
| `configs/` | yes | Hydra-style YAML: models (aya_expanse_8b, gemma2_9b), training (qlora_*), decoding (greedy, beam5, beam_rerank). |
| `scripts/` | partly | download_data.sh, run_baseline.sh, select_final.py, verify_reproducibility.py (tracked) + retrieval baselines and report builders (untracked). |
| `autoresearch_nlp/` | no | The [[Autoresearch Harness]]. Most recent structured work (June 2026). |
| `notebooks/` | no | Kaggle/Colab notebooks exp001/exp002 (May 2026). |
| `resources/` | no | Zindi zip contents you dropped in on 2026-09-18 + `multilingual_qa.ipynb` ([[Latest Colab Notebook]]). |
| `data/` | gitignored | See [[File Organisation]]. |
| `submissions/` | no (untracked) | Real submission CSVs. |
| `docs/` | partly | PLAYBOOK (tracked), ROADMAP_TOP5, SPRINT_PLAN, STEP_BY_STEP_GUIDE, competition_report PDF + figures (untracked). |
| `experiments/` | yes | LOG.md / HYPOTHESES.md / ABLATIONS.md — only the bootstrap row was ever logged ([[Experiment History]]). |
| `vault/` | new | This Obsidian vault. |
| `.cache/afrolm_embeddings/` | gitignored | Cached AfroLM embeddings from a BERTScore experiment. |

Dependency pins (`requirements.txt`): torch 2.4.1, transformers 4.46.2, peft 0.13.2, bitsandbytes 0.44.1, Python 3.11. `pyproject.toml` locally loosened `requires-python` to `>=3.11` (uncommitted diff vs GitHub's `>=3.11,<3.12`). The local Mac runs Python 3.14, so these pins cannot install locally; they are for the GPU box.

Related: [[Molab Notebook Plan]], [[File Organisation]].
