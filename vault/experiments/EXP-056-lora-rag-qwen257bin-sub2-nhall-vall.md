---
type: experiment
id: EXP-056
created: 2026-10-07
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-056 full reference recipe: RAG-enriched LoRA on Qwen2.5-7B, 3 epochs (EXP-031 was 1 epoch); Aka_Gha + Eng_Gha held-out + Val

- run: `exp056_lora_rag_qwen257bin_sub2_nhall-vall_729c43` · git `fd6aace` · status **ok** · 4804.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e` · changed: lora_epochs
- config: `{"mode": "lora_rag", "few_shot_k": 3, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.2205, test-mix 0.2206, R1 0.3453, RL 0.2507, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.1791 |
| Eng_Gha | 0.2621 |

**val**: combined 0.2187, test-mix 0.2189, R1 0.3442, RL 0.2470, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.1776 |
| Eng_Gha | 0.2603 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
