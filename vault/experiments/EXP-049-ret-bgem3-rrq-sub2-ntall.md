---
type: experiment
id: EXP-049
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-049 test predictions for EXP-048 (Lug_Uga, Swa_Ken; retriever + selector retrained on Train + Val)

- run: `exp049_ret_bgem3_rrq_sub2_ntall_7aa797` · git `c99e369` · status **ok** · 390.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp048_ret_bgem3_rrq_sub5_nhall-vall_9294c3` · changed: —
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Lug_Uga', 'Swa_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
