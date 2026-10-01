# afro-health-qa

This is a post-close practice run of the Zindi [Multilingual Health Question Answering in Low-Resource African Languages Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge). The task is to answer maternal, sexual and reproductive health questions in Akan, Amharic, Luganda, Swahili and English, across 8 locale subsets. The score is 0.37 ROUGE-1 F1 + 0.37 ROUGE-L F1 + 0.26 LLM judge, with whitespace ROUGE. The challenge is closed, so practice submissions are unlimited. Only open-weight models are allowed, and no paid APIs.

## Current state

- **Current best:** see `experiments/RESULTS.md` (per-subset best, test-mix weighted, on held-out and Val). As of 2026-10-01 the CPU tf-idf retrieval baseline (EXP-007) scores 0.2984 held-out / 0.3048 Val, ROUGE-only. Dense retrieval runs are queued next.
- No generative run has been scored yet, and no public leaderboard score is recorded.
- Goal: beat the 11th-place reference approach (BGE-M3 retrieval, then a RAG-enriched LoRA fine-tune, plus a closed-pool vs generative router) on our own held-out set, measured per subset.

## How work happens

1. Edit code locally and push it to GitHub. Then run on **molab** (GPU), which pulls the repo.
2. Experiments follow one loop: hypothesis, one change, run, score per subset, log, then keep or revert. The keep threshold is +0.003. Each experiment is a spec that `notebooks/molab_runner.py` runs. Results go to the ledger `experiments/RESULTS.md`. Run `python scripts/exp.py --help`; the procedure is in `autoresearch_nlp/program.md`.
3. The local metric replica is `src/afro_health_qa/evaluation/scorer.py`. Run `make evaluate RUN=<preds.csv>`, `make test` or `make sanity`.

## Reading order

1. `README.md` (this file)
2. `vault/00_INDEX.md`: the project's memory (facts, decisions, hypotheses, experiments)
3. `autoresearch_nlp/program.md`: the experiment loop
4. `docs/molab_runbook.md`: how to run on molab
5. `autoresearch_nlp/LESSONS.md` and `autoresearch_nlp/COMPETITION_INTEL.md`
6. `docs/notion_tracker_prompt.md`: the Notion mirror of the record

## Data policy

Zindi competition data is **never** committed to this public repo. It lives in a private Hugging Face dataset, and molab pulls it through `HF_DATA_REPO` (see `vault/decisions/D-006-data-hosting-private-hf.md`). Locally it sits in the gitignored `data/`. Never hardcode tokens.

## Archive

Old plans, Kaggle/Colab notebooks, unused scripts, the old config tree and the first experiment trackers are in `archive/`. `archive/README.md` lists what each item was and what replaced it.

## Licence

Code: Apache-2.0. No data or model weights ship with this repo.
