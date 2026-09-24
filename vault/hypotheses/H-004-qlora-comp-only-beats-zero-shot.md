---
type: hypothesis
id: H-004
created: 2026-09-24
status: open
links: ["[[D-004-base-model-for-molab]]", "[[F-003-reference-approach]]", "[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# H-004 qlora comp only beats zero shot

**Statement:** QLoRA r=32 / alpha=64, 3 epochs, LR 2e-4 on comp-only data beats zero-shot by ≥ 0.05 combined.

## Notes
Originally framed on Aya-Expanse-8B; now model-agnostic (base model pending, [[D-004-base-model-for-molab]]). Related planned ablation 'LoRA rank' r ∈ {16,32,64}, alpha=2r. The reference approach used r=64/alpha=64 on a 32B model ([[F-003-reference-approach]]).

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[D-004-base-model-for-molab]]
- [[F-003-reference-approach]]
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
