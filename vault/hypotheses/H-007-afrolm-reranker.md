---
type: hypothesis
id: H-007
created: 2026-09-24
status: superseded
links: ["[[D-001-metric-weights-and-tokenizer]]", "[[H-002-beam-search-lift]]", "[[00_INDEX]]"]
---
# H-007 afrolm reranker

**Statement:** An AfroLM-BertScore-based reranker over 4 diverse beams picks higher-scoring candidates on ≥ 60% of rows.

## Notes
Superseded: AfroLM-BS is not on the leaderboard; ranking candidates by an unscored metric was a recorded mistake (`autoresearch_nlp/LESSONS.md` #5). Any reranker must select by the scored metric (ROUGE proxy / judge). See [[D-001-metric-weights-and-tokenizer]].

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[D-001-metric-weights-and-tokenizer]]
- [[H-002-beam-search-lift]]
- [[00_INDEX]]
