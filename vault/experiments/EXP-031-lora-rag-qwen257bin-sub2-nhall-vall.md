---
type: experiment
id: EXP-031
created: 2026-10-06
status: confirmed
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-031 LoRA full data, 1 epoch: RAG-enriched LoRA on Qwen2.5-7B, all of work_train (~27.7k rows), k=3; Aka_Gha + Eng_Gha held-out + Val

- run: `exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e` · git `81dc56d` · status **ok** · 9774.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp030_lora_rag_qwen257bin_sub2_nhall_9a4f45` · changed: lora_data_frac, lora_save_steps
- config: `{"mode": "lora_rag", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **Eng_Gha**

**held_out**: combined 0.2077, test-mix 0.2077, R1 0.3201, RL 0.2411, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.1506 |
| Eng_Gha | 0.2649 |

**val**: combined 0.2047, test-mix 0.2049, R1 0.3172, RL 0.2360, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.1491 |
| Eng_Gha | 0.2608 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
