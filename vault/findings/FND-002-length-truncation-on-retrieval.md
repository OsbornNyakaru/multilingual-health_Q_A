---
type: finding
id: FND-002
created: 2026-09-24
status: confirmed
links: ["[[EXP-005-length-truncation-ablation]]", "[[H-003-per-language-length-bounds]]", "[[00_INDEX]]"]
---
# FND-002 Truncating retrieved answers does not help

From [[EXP-005-length-truncation-ablation]] on top of [[EXP-004-bge-m3-retrieval-heldout]]: cutting retrieved answers to a per-subset "optimal" length lost 0.05 ROUGE-only on each of Eng_Ken, Eng_Uga and Swa_Ken, lost 0.001–0.018 on Eng_Eth, Eng_Gha and Lug_Uga, and gained < +0.002 only on Aka_Gha and Amh_Eth (within noise). Retrieved answers are real reference text: truncating them removes matching n-grams faster than it improves precision.

Scope: this says nothing about **generated** answers. There, per-subset length bounds (gold medians range from ~20 words for Amh_Eth to ~101 for Aka_Gha) remain an open lever ([[H-003-per-language-length-bounds]]).

## Links
- [[00_INDEX]]
- [[EXP-005-length-truncation-ablation]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[H-003-per-language-length-bounds]]
