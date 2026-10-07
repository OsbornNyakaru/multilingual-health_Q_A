---
type: experiment
id: EXP-036
created: 2026-10-07
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-036 DIAG: bge-m3 + rerank of top-50 (was 20), recording candidates for recall@k on the closed-pool subsets

- run: `exp036_ret_bgem3_rrq_sub5_nhall-vall_fef674` · git `e53a3d3` · status **ok** · 232.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp013_ret_bgem3_nhall-vall_179e11` · changed: diag_k, rerank_k
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.5017, test-mix 0.5102, R1 0.6886, RL 0.6673, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4395 |
| Eng_Ken | 0.5770 |
| Eng_Uga | 0.5516 |
| Lug_Uga | 0.3879 |
| Swa_Ken | 0.5454 |

**val**: combined 0.4874, test-mix 0.4979, R1 0.6702, RL 0.6471, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.3968 |
| Eng_Ken | 0.5607 |
| Eng_Uga | 0.5515 |
| Lug_Uga | 0.3506 |
| Swa_Ken | 0.5453 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
