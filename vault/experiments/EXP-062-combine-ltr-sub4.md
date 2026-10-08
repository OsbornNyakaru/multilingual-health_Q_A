---
type: experiment
id: EXP-062
created: 2026-10-08
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-062 EXP-060 with LightGBM LambdaRank (graded relevance 0-10 from ROUGE overlap, ranked within each question) instead of the sklearn regressor; 1st place's ranker objective

- run: `exp062_combine_ltr_sub4_5f85f5` · git `b594801` · status **ok** · 0 s on none (offline combine)
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-055,EXP-051", "gen": "EXP-038", "top": 20, "model": "LGBMRanker(lambdarank)"}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **Eng_Ken, Eng_Uga, Swa_Ken**

**held_out**: combined 0.5980, test-mix 0.5948, R1 0.8157, RL 0.8006, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.6270 |
| Eng_Uga | 0.6299 |
| Lug_Uga | 0.4877 |
| Swa_Ken | 0.6320 |

**val**: combined 0.5918, test-mix 0.5917, R1 0.8072, RL 0.7923, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.6152 |
| Eng_Uga | 0.6268 |
| Lug_Uga | 0.4922 |
| Swa_Ken | 0.6232 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
