---
type: experiment
id: EXP-043
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-043 learned selector trained 3 epochs instead of 1 (1 epoch took 3 minutes)

- run: `exp043_ret_bgem3_rrq_sub5_nhall-vall_a810af` · git `d654060` · status **ok** · 762.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df` · changed: rerank_train_epochs
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "rerank_train_epochs": 3, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Swa_Ken**

**held_out**: combined 0.5236, test-mix 0.5501, R1 0.7164, RL 0.6987, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.3793 |
| Eng_Ken | 0.5871 |
| Eng_Uga | 0.6054 |
| Lug_Uga | 0.4266 |
| Swa_Ken | 0.5897 |

**val**: combined 0.5351, test-mix 0.5440, R1 0.7321, RL 0.7142, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4593 |
| Eng_Ken | 0.5695 |
| Eng_Uga | 0.6001 |
| Lug_Uga | 0.4054 |
| Swa_Ken | 0.5917 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
