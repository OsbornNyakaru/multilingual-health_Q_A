> Migrated to vault/, see [[00_INDEX]]

# Hypothesis Backlog

Anything we want to test. Each item is a short, falsifiable statement. Move items into `LOG.md` when they run; remove them when superseded.

## Pending

- **H-01**: Zero-shot Aya-Expanse-8B with greedy decoding will score combined ≥ 0.35 (baseline floor).
- **H-02**: Beam search (num_beams=5) lifts combined over greedy by ≥ 0.01 at no additional training cost.
- **H-03**: Training-answer-quantile length bounds per language reduce judge-metric truncation penalty.
- **H-04**: QLoRA r=32 / alpha=64, 3 epochs, LR 2e-4 on comp-only data beats zero-shot by ≥ 0.05 combined.
- **H-05**: Adding `medmcqa_translated` augmentation for Swahili + Luganda lifts ROUGE-L but not AfroLM-BS (translation fluency ceiling).
- **H-06**: Synthetic QA from WHO fact sheets lifts judge score more than lexical scores.
- **H-07**: AfroLM-BertScore-based reranker over 4 diverse beams picks higher-scoring candidates on ≥ 60% of rows.
- **H-08**: UlizaLlama-7B fine-tuned on Swahili rows only beats Aya-Expanse on Swahili-only AfroLM-BS.
- **H-09**: Gemma-2-9B adapter ensemble with Aya adapter improves the judge sub-metric (diverse fluency).
- **H-10**: Per-language prompt templates beat a single `Question:/Answer:` English-labelled prompt.

## Resolved
_(move items here with a one-line verdict: confirmed / refuted / inconclusive, plus linked exp ID)._
