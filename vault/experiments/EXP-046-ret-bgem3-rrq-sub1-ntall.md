---
type: experiment
id: EXP-046
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-046 test predictions for EXP-043 (Swa_Ken, 3-epoch selector retrained on Train + Val)

- run: `exp046_ret_bgem3_rrq_sub1_ntall_0e2fd5` · git `27b2023` · status **ok** · 749.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp043_ret_bgem3_rrq_sub5_nhall-vall_a810af` · changed: —
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "rerank_train_epochs": 3, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Swa_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
