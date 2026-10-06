---
type: experiment
id: EXP-013
created: 2026-10-06
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-013 bge-m3 top-20 reranked by bge-reranker-v2-m3, question vs neighbour question

- run: `exp013_ret_bgem3_nhall-vall_179e11` · git `f5fff0c` · status **ok** · 253.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp009_ret_bgem3_nhall-vall_ad6575` · changed: rerank_model
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **Eng_Uga**

**held_out**: combined 0.3775, test-mix 0.3728, R1 0.5330, RL 0.4873, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1784 |
| Amh_Eth | 0.0953 |
| Eng_Eth | 0.4367 |
| Eng_Gha | 0.1638 |
| Eng_Ken | 0.5854 |
| Eng_Uga | 0.5542 |
| Lug_Uga | 0.3762 |
| Swa_Ken | 0.5454 |

**val**: combined 0.3558, test-mix 0.3650, R1 0.5053, RL 0.4565, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1706 |
| Amh_Eth | 0.1225 |
| Eng_Eth | 0.3957 |
| Eng_Gha | 0.1679 |
| Eng_Ken | 0.5603 |
| Eng_Uga | 0.5527 |
| Lug_Uga | 0.3438 |
| Swa_Ken | 0.5437 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
