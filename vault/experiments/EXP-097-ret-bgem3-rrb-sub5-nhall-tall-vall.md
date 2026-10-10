---
type: experiment
id: EXP-097
created: 2026-10-10
status: rejected
links: ["[[H-018-cross-fitted-ranker-training]]", "[[00_INDEX]]"]
---
# EXP-097 5-fold cross-fitted EXP-055 pipeline (fine-tuned BGE-M3 + 3 answer-aware selectors per fold): leave-self-out top-30 lists for ~17.7k work_train questions to train the final ranker, and held-out/Val/test lists from the same fold models averaged (1st place's consistency lesson)

- run: `exp097_ret_bgem3_rrb_sub5_nhall-tall-vall_36387c` · git `081fa58` · status **ok** · 5855.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: diag_k, oof_folds, translate_model
- config: `{"embedder_train": true, "translate_model": "facebook/nllb-200-distilled-1.3B", "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "rerank_ensemble": 3, "no_repeat_ngram": 3, "diag_k": 30, "oof_folds": 5}` (non-default keys)
- eval: {'held_out': 0, 'val': 0, 'test': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.5778, test-mix 0.5920, R1 0.7881, RL 0.7734, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4968 |
| Eng_Ken | 0.6219 |
| Eng_Uga | 0.6279 |
| Lug_Uga | 0.5030 |
| Swa_Ken | 0.6240 |

**val**: combined 0.5739, test-mix 0.5839, R1 0.7830, RL 0.7682, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4897 |
| Eng_Ken | 0.6050 |
| Eng_Uga | 0.6247 |
| Lug_Uga | 0.4919 |
| Swa_Ken | 0.6109 |

## Links
- [[H-018-cross-fitted-ranker-training]]
- [[00_INDEX]]
