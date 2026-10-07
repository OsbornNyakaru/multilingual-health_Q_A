---
type: experiment
id: EXP-054
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-054 graded labels: the answer-aware selector learns soft targets = answer overlap with the gold (near-duplicate answers no longer taught as wrong)

- run: `exp054_ret_bgem3_rrb_sub5_nhall-vall_984c50` · git `e8c4ca4` · status **ok** · 1110.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264` · changed: rerank_train_graded
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "rerank_train_graded": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Eng_Ken**

**held_out**: combined 0.5699, test-mix 0.5833, R1 0.7780, RL 0.7622, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4920 |
| Eng_Ken | 0.6164 |
| Eng_Uga | 0.6195 |
| Lug_Uga | 0.4936 |
| Swa_Ken | 0.6120 |

**val**: combined 0.5703, test-mix 0.5795, R1 0.7786, RL 0.7627, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4924 |
| Eng_Ken | 0.5999 |
| Eng_Uga | 0.6212 |
| Lug_Uga | 0.4839 |
| Swa_Ken | 0.6079 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
