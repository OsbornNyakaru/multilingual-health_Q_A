---
type: experiment
id: EXP-040
created: 2026-10-07
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-040 EXP-039 with hybrid candidates (bge-m3 + char tf-idf) for better Luganda recall (73% at top-50 vs 88% possible)

- run: `exp040_ret_bgem3_hyb_rrq_sub5_nhall-vall_b949e3` · git `7ded48f` · status **ok** · 434.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-039` · changed: hybrid_with
- config: `{"hybrid_with": "tfidf-char", "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.5347, test-mix 0.5410, R1 0.7318, RL 0.7134, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4810 |
| Eng_Ken | 0.5958 |
| Eng_Uga | 0.5978 |
| Lug_Uga | 0.4057 |
| Swa_Ken | 0.5532 |

**val**: combined 0.5360, test-mix 0.5441, R1 0.7331, RL 0.7155, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4665 |
| Eng_Ken | 0.5763 |
| Eng_Uga | 0.5963 |
| Lug_Uga | 0.4201 |
| Swa_Ken | 0.5741 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
