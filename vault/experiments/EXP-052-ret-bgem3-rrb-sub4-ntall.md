---
type: experiment
id: EXP-052
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-052 test predictions for EXP-051 (answer-aware selector; retriever + selector retrained on Train + Val)

- run: `exp052_ret_bgem3_rrb_sub4_ntall_1f1723` · git `81923b4` · status **ok** · 768.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp051_ret_bgem3_rrb_sub5_nhall-vall_86b264` · changed: —
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
