---
type: experiment
id: EXP-023
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-023 rag few-shot with AfriqueLlama-8B instead of Qwen2.5-7B (started by a second runner session before it was put on hold)

- run: `exp023_rag_bgem3_afriquella_k2_sub2_nhall-vall_6c67a3` · git `c9508b1` · status **ok** · 1910.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-021` · changed: model_id
- config: `{"mode": "rag_few_shot", "model_id": "McGill-NLP/AfriqueLlama-8B", "no_repeat_ngram": 3}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.1015, test-mix 0.1015, R1 0.1663, RL 0.1080, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.0760 |
| Eng_Gha | 0.1270 |

**val**: combined 0.0992, test-mix 0.0993, R1 0.1617, RL 0.1064, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.0763 |
| Eng_Gha | 0.1224 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
