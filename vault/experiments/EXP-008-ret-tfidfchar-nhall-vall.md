---
type: experiment
id: EXP-008
created: 2026-10-06
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-008 answer vote over top-50 neighbours (sim^4) instead of top-1

- run: `exp008_ret_tfidfchar_nhall-vall_402044` · git `134aa90` · status **ok** · 38.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp007_ret_tfidfchar_nhall-vall_6d656b` · changed: select
- config: `{"embedder": "tfidf-char", "select": "vote"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **Eng_Eth, Eng_Uga, Lug_Uga**

**held_out**: combined 0.3177, test-mix 0.3053, R1 0.4546, RL 0.4041, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1800 |
| Amh_Eth | 0.0832 |
| Eng_Eth | 0.4568 |
| Eng_Gha | 0.1568 |
| Eng_Ken | 0.4027 |
| Eng_Uga | 0.4070 |
| Lug_Uga | 0.3760 |
| Swa_Ken | 0.3958 |

**val**: combined 0.3065, test-mix 0.3075, R1 0.4409, RL 0.3874, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1693 |
| Amh_Eth | 0.1091 |
| Eng_Eth | 0.4350 |
| Eng_Gha | 0.1621 |
| Eng_Ken | 0.4236 |
| Eng_Uga | 0.4062 |
| Lug_Uga | 0.3803 |
| Swa_Ken | 0.4113 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
