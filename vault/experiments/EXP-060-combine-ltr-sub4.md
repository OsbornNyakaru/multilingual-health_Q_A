---
type: experiment
id: EXP-060
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-060 learned final ranker (gradient-boosted) over EXP-055 + EXP-051 candidates with retriever rank, question-answer overlap, sibling similarity, frequency, length and EXP-038 generator overlap; Val out-of-fold

- run: `exp060_combine_ltr_sub4_eed26c` · git `c69a80c` · status **ok** · 0 s on none (offline combine)
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-055,EXP-051", "gen": "EXP-038", "top": 20, "model": "HistGradientBoostingRegressor"}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **Eng_Ken, Lug_Uga, Swa_Ken**

**held_out**: combined 0.6018, test-mix 0.5991, R1 0.8208, RL 0.8057, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.6232 |
| Eng_Uga | 0.6292 |
| Lug_Uga | 0.5104 |
| Swa_Ken | 0.6286 |

**val**: combined 0.5897, test-mix 0.5896, R1 0.8043, RL 0.7895, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.6070 |
| Eng_Uga | 0.6260 |
| Lug_Uga | 0.4932 |
| Swa_Ken | 0.6161 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
