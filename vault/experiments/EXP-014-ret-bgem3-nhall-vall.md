---
type: experiment
id: EXP-014
created: 2026-10-06
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-014 rerank on question vs neighbour answer instead of neighbour question

- run: `exp014_ret_bgem3_nhall-vall_a1737b` · git `f5fff0c` · status **ok** · 647.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-013` · changed: rerank_on
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_on": "answer"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **none**

**held_out**: combined 0.3127, test-mix 0.3169, R1 0.4524, RL 0.3928, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1850 |
| Amh_Eth | 0.0924 |
| Eng_Eth | 0.3082 |
| Eng_Gha | 0.1666 |
| Eng_Ken | 0.4349 |
| Eng_Uga | 0.4570 |
| Lug_Uga | 0.3137 |
| Swa_Ken | 0.4491 |

**val**: combined 0.2996, test-mix 0.3106, R1 0.4355, RL 0.3741, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1762 |
| Amh_Eth | 0.1039 |
| Eng_Eth | 0.2868 |
| Eng_Gha | 0.1727 |
| Eng_Ken | 0.4627 |
| Eng_Uga | 0.4440 |
| Lug_Uga | 0.3001 |
| Swa_Ken | 0.4292 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
