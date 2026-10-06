---
type: experiment
id: EXP-012
created: 2026-10-06
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-012 hybrid: bge-m3 blended 50/50 with char tf-idf (aimed at Lug_Uga)

- run: `exp012_ret_bgem3_nhall-vall_c5159b` · git `f5fff0c` · status **ok** · 31.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp009_ret_bgem3_nhall-vall_ad6575` · changed: hybrid_with
- config: `{"hybrid_with": "tfidf-char"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **Aka_Gha, Amh_Eth**

**held_out**: combined 0.3507, test-mix 0.3434, R1 0.4990, RL 0.4489, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1838 |
| Amh_Eth | 0.1041 |
| Eng_Eth | 0.4361 |
| Eng_Gha | 0.1624 |
| Eng_Ken | 0.5076 |
| Eng_Uga | 0.4787 |
| Lug_Uga | 0.3726 |
| Swa_Ken | 0.5069 |

**val**: combined 0.3378, test-mix 0.3457, R1 0.4827, RL 0.4303, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1735 |
| Amh_Eth | 0.1213 |
| Eng_Eth | 0.3832 |
| Eng_Gha | 0.1686 |
| Eng_Ken | 0.5065 |
| Eng_Uga | 0.4830 |
| Lug_Uga | 0.3933 |
| Swa_Ken | 0.5048 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
