---
type: experiment
id: EXP-048
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-048 fine-tuned BGE-M3 retriever + learned selector (re-run of EXP-047 after the tokenizer fix); target: Luganda recall@50 73% -> ~85%

- run: `exp048_ret_bgem3_rrq_sub5_nhall-vall_9294c3` · git `53c8e63` · status **ok** · 528.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp047_ret_bgem3_rrq_sub5_nhall-vall_da9740` · changed: —
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Lug_Uga, Swa_Ken**

**held_out**: combined 0.5502, test-mix 0.5613, R1 0.7517, RL 0.7354, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4781 |
| Eng_Ken | 0.6027 |
| Eng_Uga | 0.6115 |
| Lug_Uga | 0.4297 |
| Swa_Ken | 0.6049 |

**val**: combined 0.5548, test-mix 0.5633, R1 0.7580, RL 0.7416, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4831 |
| Eng_Ken | 0.5914 |
| Eng_Uga | 0.6123 |
| Lug_Uga | 0.4440 |
| Swa_Ken | 0.5994 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
