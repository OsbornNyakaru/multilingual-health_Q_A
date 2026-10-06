---
type: experiment
id: EXP-030
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-030 LoRA SMOKE on the already-running runner (inline training): RAG-enriched LoRA on Qwen2.5-7B, 10% of work_train, 1 epoch, k=3; eval held-out Aka_Gha + Eng_Gha

- run: `exp030_lora_rag_qwen257bin_sub2_nhall_9a4f45` · git `05bf3b7` · status **ok** · 1189.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp027_lora_rag_qwen257bin_sub2_nhall_26988d` · changed: —
- config: `{"mode": "lora_rag", "few_shot_k": 3, "lora_epochs": 1, "lora_data_frac": 0.1}` (non-default keys)
- eval: {'held_out': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.1907, test-mix 0.1907, R1 0.2940, RL 0.2215, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.1287 |
| Eng_Gha | 0.2529 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
