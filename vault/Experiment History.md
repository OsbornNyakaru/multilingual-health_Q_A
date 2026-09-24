---
tags: [experiments]
---
> Migrated to vault/ graph, see [[00_INDEX]]

# Experiment History

`experiments/LOG.md` contains only the exp000 bootstrap row. `autoresearch_nlp/results.tsv` has only a header. The real history is scattered:

- `notebooks/exp001_*` (Kaggle T4 / Colab, May 9–10): data audit, length bounds, held-out split, evaluation calibration.
- `notebooks/exp002_few_shot_colab.py`, `exp002_kaggle_or_colab.ipynb` (May 12–13): few-shot prompting.
- `data/processed/data_audit_report.md`, `suspect_amharic.csv` (May 9).
- `length_optimization_*.csv`, `length_calibration_full_report.csv` (June 24): per-subset length tuning.
- `data/processed/val_rag_enriched.jsonl` (June 24): RAG-enriched validation set.
- `submissions/`: `submission_20260512_0241.xlsx`, `submission_ready.csv`, `final_checkpoint.csv` (AfriqueLlama, see [[Latest Colab Notebook]]), `20260624_095953_e5-small_retrieval_test.csv` (+ JSON sidecar).
- `docs/competition_report/afro_health_qa_report.pdf` with figures (score ladder, roadmap, language mix).

Public leaderboard scores were never written down in the repo. **Open question** ([[Open Questions]]): what did the final submissions score?

Going forward every molab run appends to `autoresearch_nlp/results.tsv` and gets a note here.
