---
type: fact
id: F-003
created: 2026-09-24
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[H-012-rag-enriched-finetune]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[00_INDEX]]"]
---
# F-003 Reference approach (11th place, koleshjr)

Source: `koleshjr/multilingual_qa_training` (11th place), as summarised in `prompts/00_orchestrator.md`.

## Retrieval-only baseline (their numbers, verbatim)
**BGE-M3 retrieval-only** (return the nearest labelled example's answer):

| ROUGE-L F1 | ROUGE-1 F1 | LLM-Judge |
|-----------:|-----------:|----------:|
| **0.4823** | **0.5548** | **0.7379** |

Implication: many test questions are near-duplicates / paraphrases of train/val questions. Derived (not in their writeup): the ROUGE-only part is 0.37·(0.5548+0.4823) = 0.3837; with the judge term, 0.3837 + 0.26·0.7379 = 0.5756.

## Their winning move: RAG-enriched fine-tuning
- For each training row, retrieve **k=3** similar labelled examples (**excluding itself**), put them in the prompt as context, and train the model to still produce the **original** ground-truth answer (teaches using context rather than copying it).
- Base model: `Sunbird/Sunflower-32B`, Unsloth + LoRA.
- LoRA config: **rank 64, alpha 64, dropout 0.5, 3 epochs, lr 2e-4, batch 4, AdamW-8bit**.
- Honesty rule: validation retrieves from **train only**; test retrieves from **train+val** (both have known answers).

## The untried lever: sweetlhare's router
A competitor (`sweetlhare`) split subsets into:
- **closed-pool** (paraphrase-heavy) → retrieve the answer directly;
- **generative** → fine-tuned LLM.

This per-subset router is the clearest untried lever to beat 11th place. See [[H-011-closed-pool-vs-generative-router]]. Our own measurements ([[EXP-004-bge-m3-retrieval-heldout]], [[FND-001-retrieval-strength-by-subset]]) point the same way: Eng_Ken / Swa_Ken / Eng_Uga / Eng_Eth are retrieval-friendly, Amh_Eth / Aka_Gha / Eng_Gha are not.

## Links
- [[00_INDEX]]
- [[H-011-closed-pool-vs-generative-router]]
- [[H-012-rag-enriched-finetune]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[FND-001-retrieval-strength-by-subset]]
- [[F-001-competition-metric]]
