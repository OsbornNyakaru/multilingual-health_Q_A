---
type: hypothesis
id: H-012
created: 2026-09-24
status: open
links: ["[[F-003-reference-approach]]", "[[H-004-qlora-comp-only-beats-zero-shot]]", "[[00_INDEX]]"]
---
# H-012 rag enriched finetune

**Statement:** RAG-enriched LoRA fine-tuning (k=3 retrieved labelled examples in context, excluding self; target = original answer) beats plain SFT on the same base model.

## Notes
The 11th-place winning move ([[F-003-reference-approach]]). Val retrieves from train only; test from train+val. A RAG-enriched validation file already exists locally (`data/processed/val_rag_enriched.jsonl`, 2026-06-24), not yet used for training.

Source: new, from the reference approach ([[F-003-reference-approach]]) and `autoresearch_nlp/COMPETITION_INTEL.md`.

## Links
- [[F-003-reference-approach]]
- [[H-004-qlora-comp-only-beats-zero-shot]]
- [[00_INDEX]]
