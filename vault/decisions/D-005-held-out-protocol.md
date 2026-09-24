---
type: decision
id: D-005
created: 2026-09-24
status: confirmed
links: ["[[F-002-data-shape]]", "[[D-001-metric-weights-and-tokenizer]]", "[[00_INDEX]]"]
---
# D-005 Held-out protocol and keep/revert threshold

**Decision:**
- Local evaluation set = `autoresearch_nlp/prepare.py make_splits()`: 7% of Train, stratified by `subset`, seed 1234 → `held_out.csv` (2,088 rows); `work_train.csv` (27,727) for training/few-shot/retrieval pool.
- Never train on, draw few-shot from, or retrieve from held-out. (Val may be used for training once held-out is reserved; RAG retrieval for held-out rows uses the training pool only.)
- Report R1, RL, judge stand-in and weighted total, **per subset** and overall.
- Keep a change only if weighted combined improves by **≥ +0.003** over current best (noise σ ≈ 0.005–0.010 on a few-hundred-row slice); otherwise revert.
- One lever per experiment. Never print an estimate where a measurement belongs.

**Why:** an earlier notebook folded Val into training and printed fake scores ([[EXP-001-aya-expanse-oom]]; `autoresearch_nlp/LESSONS.md` #4).

Source: legacy [[Scoring Metric]], [[Autoresearch Harness]], `autoresearch_nlp/program.md`, `docs/SPRINT_PLAN.md`.

## Links
- [[00_INDEX]]
- [[F-002-data-shape]]
- [[D-001-metric-weights-and-tokenizer]]
- [[EXP-001-aya-expanse-oom]]
- [[Autoresearch Harness]]
