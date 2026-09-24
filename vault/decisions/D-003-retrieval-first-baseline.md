---
type: decision
id: D-003
created: 2026-09-24
status: confirmed
links: ["[[EXP-004-bge-m3-retrieval-heldout]]", "[[FND-001-retrieval-strength-by-subset]]", "[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# D-003 Retrieval is the floor; generation must beat it per subset

**Decision:** Nearest-neighbour retrieval (BGE-M3, within subset) is the baseline every generative run must beat, **per subset**, on held-out. The target architecture is a retrieval + generation router ([[H-011-closed-pool-vs-generative-router]]) rather than pure fine-tuning.

**Why:** Measured retrieval already reaches R1 0.560 / RL 0.491 on held-out ([[EXP-004-bge-m3-retrieval-heldout]]), comparable to the 11th-place retrieval baseline ([[F-003-reference-approach]]). Retrieval dominates the English and Swahili subsets but fails on Amharic/Akan/Eng_Gha ([[FND-001-retrieval-strength-by-subset]]). Small zero-shot generators (mT5) score near 0 on ROUGE (`autoresearch_nlp/COMPETITION_INTEL.md` §3).

**Alternatives:**
- Pure fine-tuned generation for all rows: leaves easy retrieval points on the table.
- Pure retrieval: caps out on Amharic (~0.01) and Akan.

Source: legacy [[Model Decisions]] (e5/BGE retrieval rows), `autoresearch_nlp/COMPETITION_INTEL.md` §5, `docs/ROADMAP_TOP5.md`.

## Links
- [[00_INDEX]]
- [[EXP-003-token-overlap-retrieval]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[EXP-006-e5-small-retrieval-test-submission]]
- [[FND-001-retrieval-strength-by-subset]]
- [[H-011-closed-pool-vs-generative-router]]
- [[F-003-reference-approach]]
