---
type: experiment
id: EXP-025
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-025 SCREEN: same 40 rows as EXP-020 with no_repeat_ngram off (it pushed Qwen into Chinese and garbled Akan)

- run: `exp025_rag_bgem3_qwen257bin_k2_sub2_nh40_cc19d5` · git `a743b70` · status **ok** · 24.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp020_rag_bgem3_qwen257bin_k2_sub2_nh40_4658cc` · changed: no_repeat_ngram
- config: `{"mode": "rag_few_shot"}` (non-default keys)
- eval: {'held_out': 40} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.1801, test-mix 0.1801, R1 0.2775, RL 0.2093, n=40

| subset | combined |
|---|--:|
| Aka_Gha | 0.1785 |
| Eng_Gha | 0.1818 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
