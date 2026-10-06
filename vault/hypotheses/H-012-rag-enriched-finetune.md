---
type: hypothesis
id: H-012
created: 2026-09-24
status: testing
links: ["[[F-003-reference-approach]]", "[[H-004-qlora-comp-only-beats-zero-shot]]", "[[00_INDEX]]"]
---
# H-012 rag enriched finetune

**Statement:** RAG-enriched LoRA fine-tuning (k=3 retrieved labelled examples in context, excluding self; target = original answer) beats plain SFT on the same base model.

## Notes
The 11th-place winning move ([[F-003-reference-approach]]). Val retrieves from train only; test from train+val. A RAG-enriched validation file already exists locally (`data/processed/val_rag_enriched.jsonl`, 2026-06-24), not yet used for training.

Source: new, from the reference approach ([[F-003-reference-approach]]) and `autoresearch_nlp/COMPETITION_INTEL.md`.

## 2026-10-06: first evidence (EXP-030 smoke test)
- RAG-enriched LoRA on Qwen2.5-7B, reference hyperparameters (r64, α64, dropout 0.5, lr 2e-4, batch 4, k=3 excluding self), **10% of work_train, 1 epoch**: 2,772 sequences, 693 steps in 13 min on the RTX Pro 6000 (~0.9 steps/s), loss 1.91 → 1.22.
- Held-out (623 rows): **Eng_Gha 0.253** vs prompting 0.198 (EXP-026) vs retrieval 0.169 (EXP-009/012), with answer length now matching the reference (67 vs 71 words). **Aka_Gha 0.129**, below retrieval's 0.184: Qwen is weak in Akan and saw only ~400 Akan examples.
- Next: full data (1 epoch ≈ 2 h, 3 epochs ≈ 6.5 h), held-out + Val; an Akan-capable base (AfriqueLlama-8B) for Aka_Gha.

## Links
- [[F-003-reference-approach]]
- [[H-004-qlora-comp-only-beats-zero-shot]]
- [[00_INDEX]]
