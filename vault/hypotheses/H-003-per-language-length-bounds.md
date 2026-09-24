---
type: hypothesis
id: H-003
created: 2026-09-24
status: testing
links: ["[[EXP-005-length-truncation-ablation]]", "[[FND-002-length-truncation-on-retrieval]]", "[[00_INDEX]]"]
---
# H-003 per language length bounds

**Statement:** Training-answer-quantile length bounds per language reduce the truncation penalty and improve ROUGE F1.

## Notes
Planned ablation 'Per-language length policy' (fixed max=128 vs per-language quantile; expected to halve Amharic truncation). Partial evidence from [[EXP-005-length-truncation-ablation]]: truncating *retrieval* outputs to an 'optimal' length did **not** help overall (large losses on Eng_Ken/Eng_Uga/Swa_Ken, tiny gains on Amh_Eth/Aka_Gha). Still untested for *generated* outputs, where length is the free variable. See [[FND-002-length-truncation-on-retrieval]].

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[EXP-005-length-truncation-ablation]]
- [[FND-002-length-truncation-on-retrieval]]
- [[00_INDEX]]
