---
type: experiment
id: EXP-086
created: 2026-10-09
status: confirmed
links: ["[[00_INDEX]]"]
---
# EXP-086 LambdaRank with 1st-place selector features (pool near-duplicate mass, agreement with other candidates, per-question z-scores and gaps)

- run: `exp086_combine_ltr_sub4_e0ac5e` · git `0a92e70` · status **ok** · 0 s on none (offline combine)
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: combine
- config: `{"combine": {"rule": "ltr", "pickers": "EXP-055,EXP-051", "gen": "EXP-038", "top": 20, "model": "LGBMRanker(lambdarank)", "feats": "v2", "per_subset": false}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **Eng_Uga, Swa_Ken**

**held_out**: combined 0.6004, test-mix 0.5973, R1 0.8188, RL 0.8039, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.6233 |
| Eng_Uga | 0.6332 |
| Lug_Uga | 0.4924 |
| Swa_Ken | 0.6331 |

**val**: combined 0.5954, test-mix 0.5953, R1 0.8120, RL 0.7973, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.6187 |
| Eng_Uga | 0.6298 |
| Lug_Uga | 0.4932 |
| Swa_Ken | 0.6330 |

## Links
- [[00_INDEX]]
