---
type: decision
id: D-004
created: 2026-09-24
status: open
links: ["[[D-002-reject-aya-expanse]]", "[[F-003-reference-approach]]", "[[F-004-rules]]", "[[00_INDEX]]"]
---
# D-004 Which licence-clean base model to standardise on for molab (OPEN)

**Status:** not decided. Carried over from legacy [[Open Questions]] ("Which licence-clean base model to standardise on").

**Constraints:** open-weight, licence-clean ([[F-004-rules]]); molab RTX Pro 6000, 96 GB VRAM — an 8B model fits in bf16 without bitsandbytes, and 32B LoRA is feasible ([[Molab Platform]]).

**Candidates:**

| Model | Evidence so far |
|-------|-----------------|
| `McGill-NLP/AfriqueLlama-8B` | Actually run in 4-bit on T4 ([[EXP-002-afriquellama-fewshot-colab]]); never scored locally. Fastest path to reproduce the old baseline in bf16. |
| `Sunbird/Sunflower-32B` | The 11th-place base for RAG-enriched LoRA ([[F-003-reference-approach]]). |
| Gemma-2-9B / 27B | `configs/models/gemma2_9b.yaml` exists; config only. |
| Llama-3.1-8B, Qwen2.5 | Suggested in legacy notes, untested. |
| `CohereLabs/aya-expanse-8b` | Rejected ([[D-002-reject-aya-expanse]]). |
| AfroLM | Encoder for BertScore only, not a generator. |

**Prior plan (legacy [[Model Decisions]]):** (1) AfriqueLlama-8B bf16 zero/few-shot to reproduce the old baseline fast, (2) LoRA SFT on work_train + Val, (3) larger permissive models for per-subset selection.

**How to close:** licence check + one held-out run per candidate under [[D-005-held-out-protocol]]; pick by per-subset weighted score.

## Links
- [[00_INDEX]]
- [[D-002-reject-aya-expanse]]
- [[EXP-002-afriquellama-fewshot-colab]]
- [[F-003-reference-approach]]
- [[F-004-rules]]
- [[H-004-qlora-comp-only-beats-zero-shot]]
- [[H-008-ulizallama-swahili]]
