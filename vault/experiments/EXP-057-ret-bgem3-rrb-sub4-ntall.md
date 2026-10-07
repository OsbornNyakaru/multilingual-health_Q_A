---
type: experiment
id: EXP-057
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-057 test predictions for EXP-055 (3-selector ensemble; input to the learned final ranker)

- run: `exp057_ret_bgem3_rrb_sub4_ntall_506623` · git `7f555d3` · status **ok** · 2084.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57` · changed: —
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "rerank_ensemble": 3, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
