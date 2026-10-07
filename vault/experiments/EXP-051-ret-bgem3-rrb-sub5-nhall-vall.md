---
type: experiment
id: EXP-051
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-051 answer-aware selector: the cross-encoder sees candidate question || answer (first 400 chars), trained and scored that way; fine-tuned retriever as in EXP-048

- run: `exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264` · git `4f50024` · status **ok** · 1089.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp048_ret_bgem3_rrq_sub5_nhall-vall_9294c3` · changed: rerank_on, rerank_train_max_len
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Eng_Uga, Lug_Uga**

**held_out**: combined 0.5706, test-mix 0.5850, R1 0.7787, RL 0.7636, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4875 |
| Eng_Ken | 0.6133 |
| Eng_Uga | 0.6280 |
| Lug_Uga | 0.4860 |
| Swa_Ken | 0.6120 |

**val**: combined 0.5665, test-mix 0.5760, R1 0.7734, RL 0.7577, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4869 |
| Eng_Ken | 0.5878 |
| Eng_Uga | 0.6163 |
| Lug_Uga | 0.4862 |
| Swa_Ken | 0.6064 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
