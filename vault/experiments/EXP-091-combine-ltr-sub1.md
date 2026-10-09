---
type: experiment
id: EXP-091
created: 2026-10-09
status: confirmed
links: ["[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# EXP-091 EXP-060's learned final ranker on Lug_Uga plus the NLLB-translated view's top-50 (EXP-088 tr_ids) as a third candidate list (FND-004 translate-then-retrieve)

- run: `exp091_combine_ltr_sub1_489d9a` · git `3de731f` · status **ok** · 0 s on none (offline combine)
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-055,EXP-051,EXP-088:tr_ids", "gen": "EXP-038", "top": 20, "model": "HistGradientBoostingRegressor", "feats": "v1", "per_subset": false}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Lug_Uga']
- adopted for subsets: **Lug_Uga**

**held_out**: combined 0.5205, test-mix 0.5205, R1 0.7120, RL 0.6948, n=237

| subset | combined |
|---|--:|
| Lug_Uga | 0.5205 |

**val**: combined 0.5118, test-mix 0.5118, R1 0.6998, RL 0.6835, n=846

| subset | combined |
|---|--:|
| Lug_Uga | 0.5118 |

## Links
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
