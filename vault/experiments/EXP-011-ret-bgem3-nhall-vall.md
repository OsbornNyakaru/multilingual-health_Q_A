---
type: experiment
id: EXP-011
created: 2026-10-06
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-011 bge-m3 + answer vote over top-50 (sim^4)

- run: `exp011_ret_bgem3_nhall-vall_2f8d92` · git `f5fff0c` · status **ok** · 35.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp009_ret_bgem3_nhall-vall_ad6575` · changed: select
- config: `{"select": "vote"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **none**

**held_out**: combined 0.2922, test-mix 0.2810, R1 0.4215, RL 0.3681, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1750 |
| Amh_Eth | 0.0976 |
| Eng_Eth | 0.4063 |
| Eng_Gha | 0.1739 |
| Eng_Ken | 0.3320 |
| Eng_Uga | 0.4133 |
| Lug_Uga | 0.2565 |
| Swa_Ken | 0.3274 |

**val**: combined 0.2864, test-mix 0.2882, R1 0.4153, RL 0.3589, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1676 |
| Amh_Eth | 0.1178 |
| Eng_Eth | 0.3853 |
| Eng_Gha | 0.1733 |
| Eng_Ken | 0.3371 |
| Eng_Uga | 0.4542 |
| Lug_Uga | 0.2157 |
| Swa_Ken | 0.3567 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
