---
type: experiment
id: EXP-033
created: 2026-10-07
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-033 Akan: reuse EXP-032's AfriqueLlama adapter (no retraining) with a minimum answer length (p10 of real answers); it stopped early on 32% of answers

- run: `exp033_rag_bgem3_afriquella_k3_sub1_nhall-vall_944d2c` · git `711415f` · status **ok** · 1699.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp032_lora_rag_afriquella_sub2_nhall-vall_a9b0c3` · changed: adapter, min_len_pct, mode
- config: `{"mode": "rag_few_shot", "model_id": "McGill-NLP/AfriqueLlama-8B", "adapter": "run:exp032_lora_rag_afriquella_sub2_nhall-vall_a9b0c3", "few_shot_k": 3, "min_len_pct": 10, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.1774, test-mix 0.1774, R1 0.2851, RL 0.1943, n=312

| subset | combined |
|---|--:|
| Aka_Gha | 0.1774 |

**val**: combined 0.1740, test-mix 0.1740, R1 0.2795, RL 0.1909, n=1114

| subset | combined |
|---|--:|
| Aka_Gha | 0.1740 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
