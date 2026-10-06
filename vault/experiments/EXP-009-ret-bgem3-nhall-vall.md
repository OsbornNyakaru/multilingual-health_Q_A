---
type: experiment
id: EXP-009
created: 2026-10-06
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-009 dense retrieval with bge-m3 instead of char tf-idf

- run: `exp009_ret_bgem3_nhall-vall_ad6575` · git `134aa90` · status **ok** · 81.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp007_ret_tfidfchar_nhall-vall_6d656b` · changed: embedder
- config: `{}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **Amh_Eth, Eng_Gha, Eng_Ken, Eng_Uga, Swa_Ken**

**held_out**: combined 0.3697, test-mix 0.3620, R1 0.5228, RL 0.4762, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1745 |
| Amh_Eth | 0.0981 |
| Eng_Eth | 0.4435 |
| Eng_Gha | 0.1687 |
| Eng_Ken | 0.5895 |
| Eng_Uga | 0.5467 |
| Lug_Uga | 0.3134 |
| Swa_Ken | 0.5413 |

**val**: combined 0.3502, test-mix 0.3587, R1 0.4981, RL 0.4483, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1668 |
| Amh_Eth | 0.1168 |
| Eng_Eth | 0.3973 |
| Eng_Gha | 0.1738 |
| Eng_Ken | 0.5704 |
| Eng_Uga | 0.5420 |
| Lug_Uga | 0.3055 |
| Swa_Ken | 0.5593 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
