---
type: experiment
id: EXP-032
created: 2026-10-07
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-032 same as EXP-031 with AfriqueLlama-8B as the base (for Akan)

- run: `exp032_lora_rag_afriquella_sub2_nhall-vall_a9b0c3` · git `81dc56d` · status **ok** · 12597.8 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-031` · changed: model_id
- config: `{"mode": "lora_rag", "model_id": "McGill-NLP/AfriqueLlama-8B", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.2086, test-mix 0.2086, R1 0.3242, RL 0.2397, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.1593 |
| Eng_Gha | 0.2581 |

**val**: combined 0.2032, test-mix 0.2034, R1 0.3174, RL 0.2317, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.1514 |
| Eng_Gha | 0.2555 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
