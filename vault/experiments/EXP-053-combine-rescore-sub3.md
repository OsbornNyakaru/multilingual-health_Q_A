---
type: experiment
id: EXP-053
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-053 re-scoring (selector rank + retriever rank + question-answer overlap) on top of the answer-aware selector EXP-051

- run: `exp053_combine_rescore_sub3_6aaa58` · git `9b0e49c` · status **ok** · 0 s on none (offline combine)
- parent: `exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264` · changed: combine
- config: `{"combine": {"rule": "rescore", "base": "exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264", "weights": {"Eng_Uga": [0.5, 8.0], "Swa_Ken": [0.25, 0.0], "Eng_Ken": [0.5, 4.0]}, "top": 20}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **Eng_Ken, Eng_Uga, Swa_Ken**

**held_out**: combined 0.6244, test-mix 0.6244, R1 0.8508, RL 0.8368, n=825

| subset | combined |
|---|--:|
| Eng_Ken | 0.6167 |
| Eng_Uga | 0.6299 |
| Swa_Ken | 0.6122 |

**val**: combined 0.6136, test-mix 0.6137, R1 0.8365, RL 0.8219, n=2596

| subset | combined |
|---|--:|
| Eng_Ken | 0.5942 |
| Eng_Uga | 0.6200 |
| Swa_Ken | 0.6074 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
