---
type: experiment
id: EXP-007
created: 2026-10-01
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-007 CPU baseline: char 3-5-gram tfidf retrieval, top-1, within subset

- run: `exp007_ret_tfidfchar_nhall-vall_6d656b` · git `6c0b8e1` · status **ok** · 453.6 s on none
- parent: `—` · changed: embedder
- config: `{"embedder": "tfidf-char"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **Aka_Gha, Amh_Eth, Eng_Eth, Eng_Gha, Eng_Ken, Eng_Uga, Lug_Uga, Swa_Ken**

**held_out**: combined 0.3085, test-mix 0.2984, R1 0.4427, RL 0.3910, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1801 |
| Amh_Eth | 0.0832 |
| Eng_Eth | 0.4265 |
| Eng_Gha | 0.1546 |
| Eng_Ken | 0.4116 |
| Eng_Uga | 0.3899 |
| Lug_Uga | 0.3508 |
| Swa_Ken | 0.4194 |

**val**: combined 0.3003, test-mix 0.3048, R1 0.4329, RL 0.3788, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1694 |
| Amh_Eth | 0.1091 |
| Eng_Eth | 0.3771 |
| Eng_Gha | 0.1591 |
| Eng_Ken | 0.4306 |
| Eng_Uga | 0.4025 |
| Lug_Uga | 0.3700 |
| Swa_Ken | 0.4254 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
