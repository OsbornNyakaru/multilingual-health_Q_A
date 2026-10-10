---
type: experiment
id: EXP-106
created: 2026-10-10
status: confirmed
links: ["[[H-018-cross-fitted-ranker-training]]", "[[00_INDEX]]"]
---
# EXP-106 Closed-subset ranker (LambdaRank, pooled over 4 subsets) over EXP-097's fold-averaged selector lists, no generator; the cross-fitted training lists themselves did not help

- run: `exp106_combine_ltr_sub4_1cb7a1` · git `8d11133` · status **ok** · 0 s on none (offline combine)
- parent: `exp097_ret_bgem3_rrb_sub5_nhall-tall-vall_36387c` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-097,EXP-097:cand1_ids", "gen": null, "top": 20, "model": "LGBMRanker(lambdarank)", "feats": "v1", "per_subset": false}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **Eng_Ken**

**held_out**: combined 0.5988, test-mix 0.5958, R1 0.8165, RL 0.8018, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.6284 |
| Eng_Uga | 0.6258 |
| Lug_Uga | 0.4910 |
| Swa_Ken | 0.6456 |

**val**: combined 0.5911, test-mix 0.5909, R1 0.8061, RL 0.7914, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.6258 |
| Eng_Uga | 0.6250 |
| Lug_Uga | 0.4873 |
| Swa_Ken | 0.6239 |

## Links
- [[H-018-cross-fitted-ranker-training]]
- [[00_INDEX]]
