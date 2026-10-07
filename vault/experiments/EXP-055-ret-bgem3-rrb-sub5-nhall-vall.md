---
type: experiment
id: EXP-055
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-055 ensemble of 3 answer-aware selectors (seeds 0-2: different data order and sampled hard negatives), scores averaged

- run: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · git `e8c4ca4` · status **ok** · 2980.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264` · changed: rerank_ensemble
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "rerank_ensemble": 3, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Eng_Uga**

**held_out**: combined 0.5786, test-mix 0.5931, R1 0.7892, RL 0.7745, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4965 |
| Eng_Ken | 0.6209 |
| Eng_Uga | 0.6288 |
| Lug_Uga | 0.5121 |
| Swa_Ken | 0.6147 |

**val**: combined 0.5708, test-mix 0.5803, R1 0.7790, RL 0.7636, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4902 |
| Eng_Ken | 0.5886 |
| Eng_Uga | 0.6231 |
| Lug_Uga | 0.4878 |
| Swa_Ken | 0.6098 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
