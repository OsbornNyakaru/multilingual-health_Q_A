---
type: experiment
id: EXP-041
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-041 test predictions for EXP-039 (learned selector retrained on Train + Val, the test pool)

- run: `exp041_ret_bgem3_rrq_sub5_ntall_e1a18d` · git `d654060` · status **ok** · 329.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df` · changed: —
- config: `{"rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
