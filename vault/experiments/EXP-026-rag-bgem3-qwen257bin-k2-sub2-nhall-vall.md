---
type: experiment
id: EXP-026
created: 2026-10-06
status: confirmed
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-026 full run of EXP-025: Qwen2.5-7B rag few-shot k=2, no_repeat_ngram off, Aka_Gha + Eng_Gha

- run: `exp026_rag_bgem3_qwen257bin_k2_sub2_nhall-vall_47e98d` · git `abbac19` · status **ok** · 1215.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp025_rag_bgem3_qwen257bin_k2_sub2_nh40_cc19d5` · changed: —
- config: `{"mode": "rag_few_shot"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **Eng_Gha**

**held_out**: combined 0.1863, test-mix 0.1863, R1 0.2835, RL 0.2201, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.1745 |
| Eng_Gha | 0.1982 |

**val**: combined 0.1801, test-mix 0.1801, R1 0.2746, RL 0.2122, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.1675 |
| Eng_Gha | 0.1928 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
