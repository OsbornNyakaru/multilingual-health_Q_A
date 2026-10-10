---
type: experiment
id: EXP-105
created: 2026-10-10
status: rejected
links: ["[[H-018-cross-fitted-ranker-training]]", "[[00_INDEX]]"]
---
# EXP-105 Eng_Ken ranker (LambdaRank) over EXP-097's fold-averaged selector lists (ret_ids + cand_ids + cand1_ids), no generator; the cross-fitted training lists themselves did not help

- run: `exp105_combine_ltr_sub1_1cb7a1` · git `8d11133` · status **ok** · 0 s on none (offline combine)
- parent: `exp097_ret_bgem3_rrb_sub5_nhall-tall-vall_36387c` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-097,EXP-097:cand1_ids", "gen": null, "top": 20, "model": "LGBMRanker(lambdarank)", "feats": "v1", "per_subset": false}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Ken']
- adopted for subsets: **none**

**held_out**: combined 0.6138, test-mix 0.6138, R1 0.8398, RL 0.8191, n=146

| subset | combined |
|---|--:|
| Eng_Ken | 0.6138 |

**val**: combined 0.6114, test-mix 0.6114, R1 0.8343, RL 0.8181, n=390

| subset | combined |
|---|--:|
| Eng_Ken | 0.6114 |

## Links
- [[H-018-cross-fitted-ranker-training]]
- [[00_INDEX]]
