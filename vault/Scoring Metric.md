---
tags: [metric]
---
> Migrated to vault/ graph, see [[00_INDEX]]

# Scoring Metric

Zindi scored ROUGE-1 F1, ROUGE-L F1 and an LLM judge (the three target columns in [[Submission Format]]). The exact host weights were never pinned down in the repo:

- `configs/base.yaml` says 0.25 / 0.25 / 0.30 (AfroLM BERTScore) / 0.20 (judge) — this is the pre-drop guess.
- `autoresearch_nlp/prepare.py` (frozen harness) says **rouge1 0.37, rougeL 0.37, judge 0.26** — this is the later, deliberate choice. Use this one.

## Local scoring rules (from prepare.py)
- Google `rouge-score` library, **whitespace tokenizer, no stemming**. The default tokenizer strips Amharic Ge'ez script to nothing, so whitespace tokenisation matches the official starter notebook.
- Judge is 0 locally; ROUGE alone still ranks experiments correctly.
- Noise band: sigma ~0.005–0.010 on a few-hundred-row slice. Only trust deltas >= +0.003 combined.
- Held-out slice: 7% of Train, stratified by subset, seed 1234. Never train or draw few-shot from it.

Related: [[Autoresearch Harness]], [[Data Schema]].
