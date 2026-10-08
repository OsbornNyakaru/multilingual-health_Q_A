---
type: hypothesis
id: H-014
created: 2026-10-08
status: open
links: ["[[F-007-winning-solutions]]", "[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# H-014 bigger generator on the generated subsets

**Statement:** On the subsets whose answers must be generated (Aka_Gha, Eng_Gha, Amh_Eth), a 27–31B LoRA beats our 7B LoRA and retrieval copy. The main evidence is 1st place's single gemma-4-31B LoRA on Val with our scorer: Aka 0.216, Eng_Gha 0.254, Amh 0.154, against our 0.174 / 0.261 / 0.121 ([[F-007-winning-solutions]]). More epochs on the 7B gave nothing (EXP-056), so size is the remaining lever.

## Tests
- EXP-061: Gemma-4-31B QLoRA, our RAG k=3 prompt, trained on the three subsets only, 1 epoch.
- Next: MBR over sampled answers (12 samples × T 0.7/1.0/1.3), then a second large model (Qwen3.6/3.8-27B, k=5 same-subset demos) for cross-model MBR.

## Links
- [[F-007-winning-solutions]]
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
