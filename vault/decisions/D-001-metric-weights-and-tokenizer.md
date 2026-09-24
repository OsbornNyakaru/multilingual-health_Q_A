---
type: decision
id: D-001
created: 2026-09-24
status: confirmed
links: ["[[F-001-competition-metric]]", "[[H-013-whitespace-tokenizer-matches-grader]]", "[[00_INDEX]]"]
---
# D-001 Optimise 0.37 R1 / 0.37 RL / 0.26 judge, whitespace-tokenized ROUGE

**Decision:** All local scoring uses the leaderboard weights (0.37 ROUGE-1 F1, 0.37 ROUGE-L F1, 0.26 LLM-judge normalized) and `rouge-score` with a whitespace tokenizer, `use_stemmer=False`. AfroLM BertScore is reported at most as a diagnostic, with weight 0, and is never used for model/candidate selection.

**Why:** The repo spent weeks optimising the pre-drop guess (0.25/0.25/0.30 AfroLM/0.20), a metric that scored zero on the board (`autoresearch_nlp/LESSONS.md` #1, #5). The whitespace tokenizer matches the starter notebook and keeps Ge'ez tokens (default tokenizer drops them).

**Alternatives considered:**
- `configs/base.yaml` weights: rejected, wrong.
- Default `rouge-score` tokenizer: rejected, it zeroes Amharic; keep only as a comparison in the metric replica.

**Open:** whether the host grader really uses whitespace tokenization ([[H-013-whitespace-tokenizer-matches-grader]]). Resolves the metric-weights item of legacy [[Open Questions]].

## Links
- [[00_INDEX]]
- [[F-001-competition-metric]]
- [[H-013-whitespace-tokenizer-matches-grader]]
- [[H-007-afrolm-reranker]]
