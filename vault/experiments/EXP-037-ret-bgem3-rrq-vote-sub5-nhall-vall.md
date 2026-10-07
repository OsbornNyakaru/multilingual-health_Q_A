---
type: experiment
id: EXP-037
created: 2026-10-07
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-037 answer-group scoring: sum reranker scores of all top-50 candidates that share an answer, instead of the single top question

- run: `exp037_ret_bgem3_rrq_vote_sub5_nhall-vall_919c98` · git `e53a3d3` · status **ok** · 230.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-036` · changed: select, vote_power
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "select": "vote", "vote_power": 1, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.3818, test-mix 0.3703, R1 0.5330, RL 0.4989, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4197 |
| Eng_Ken | 0.3633 |
| Eng_Uga | 0.4350 |
| Lug_Uga | 0.2700 |
| Swa_Ken | 0.3158 |

**val**: combined 0.3846, test-mix 0.3835, R1 0.5372, RL 0.5023, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.3937 |
| Eng_Ken | 0.3681 |
| Eng_Uga | 0.4585 |
| Lug_Uga | 0.2768 |
| Swa_Ken | 0.3227 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
