---
type: experiment
id: EXP-095
created: 2026-10-09
status: rejected
links: ["[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# EXP-095 LambdaRank final ranker on Swa_Ken alone plus the NLLB-translated view's top-50 (EXP-092 tr_ids) as a third candidate list

- run: `exp095_combine_ltr_sub1_7776e5` · git `20dcaa3` · status **ok** · 0 s on none (offline combine)
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-055,EXP-051,EXP-092:tr_ids", "gen": "EXP-038", "top": 20, "model": "LGBMRanker(lambdarank)", "feats": "v1", "per_subset": false}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Swa_Ken']
- adopted for subsets: **none**

**held_out**: combined 0.6342, test-mix 0.6342, R1 0.8636, RL 0.8506, n=145

| subset | combined |
|---|--:|
| Swa_Ken | 0.6342 |

**val**: combined 0.6278, test-mix 0.6278, R1 0.8559, RL 0.8409, n=518

| subset | combined |
|---|--:|
| Swa_Ken | 0.6278 |

## Links
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
