---
type: hypothesis
id: H-001
created: 2026-09-24
status: superseded
links: ["[[D-002-reject-aya-expanse]]", "[[EXP-001-aya-expanse-oom]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[00_INDEX]]"]
---
# H-001 zero shot aya floor

**Statement:** Zero-shot Aya-Expanse-8B with greedy decoding will score combined ≥ 0.35 (baseline floor).

## Notes
Superseded: Aya-Expanse-8B was rejected as a base model ([[D-002-reject-aya-expanse]]) and never produced a measured score (4-bit load OOM'd on a T4, [[EXP-001-aya-expanse-oom]]). The 0.35 floor was also stated under the old 0.25/0.25/0.30/0.20 weights. Retrieval already sets a higher floor ([[EXP-004-bge-m3-retrieval-heldout]]).

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[D-002-reject-aya-expanse]]
- [[EXP-001-aya-expanse-oom]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[00_INDEX]]
