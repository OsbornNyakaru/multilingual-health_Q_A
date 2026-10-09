---
type: experiment
id: EXP-090
created: 2026-10-09
status: rejected
links: ["[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# EXP-090 EXP-088 on the test set (Lug_Uga): translated-view rankings feed the Lug_Uga combiner (ltr.py EXP-088:tr_ids picker)

- run: `exp090_ret_bgem3_sub1_ntall_e42cae` · git `bcb2375` · status **ok** · 124.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp088_ret_bgem3_sub1_nhall-vall_a31b27` · changed: —
- config: `{"translate_model": "facebook/nllb-200-distilled-1.3B", "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Lug_Uga']
- adopted for subsets: **none**

## Links
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
