---
type: experiment
id: EXP-039
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-039 learned selector: fine-tune bge-reranker-v2-m3 on work_train (positive = question with the same answer, 7 hard negatives), then rerank bge-m3 top-50

- run: `exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df` · git `7ded48f` · status **ok** · 425.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp036_ret_bgem3_rrq_sub5_nhall-vall_fef674` · changed: rerank_train
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Eng_Ken, Eng_Uga, Lug_Uga, Swa_Ken**

**held_out**: combined 0.5386, test-mix 0.5513, R1 0.7366, RL 0.7190, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4574 |
| Eng_Ken | 0.5935 |
| Eng_Uga | 0.6068 |
| Lug_Uga | 0.4295 |
| Swa_Ken | 0.5637 |

**val**: combined 0.5409, test-mix 0.5497, R1 0.7395, RL 0.7225, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4665 |
| Eng_Ken | 0.5835 |
| Eng_Uga | 0.6051 |
| Lug_Uga | 0.4149 |
| Swa_Ken | 0.5867 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
