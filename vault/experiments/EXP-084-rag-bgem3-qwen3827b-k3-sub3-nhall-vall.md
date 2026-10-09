---
type: experiment
id: EXP-084
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-084 13-candidate vLLM generation with EXP-083's Qwen3.8-27B adapter (max_num_seqs 256; EXP-083's own generation hits the Mamba-cache limit)

- run: `exp084_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_e80aaa` · git `5f9645c` · status **ok** · 7229.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: adapter, model_id, vllm_max_seqs
- config: `{"mode": "rag_few_shot", "model_id": "Qwen/Qwen3.8-27B", "precision": "bf16", "adapter": "run:exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "vllm_max_seqs": 256, "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.1177, test-mix 0.1198, R1 0.1747, RL 0.1435, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.0783 |
| Amh_Eth | 0.1028 |
| Eng_Gha | 0.1635 |

**val**: combined 0.1196, test-mix 0.1201, R1 0.1773, RL 0.1460, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.0774 |
| Amh_Eth | 0.1173 |
| Eng_Gha | 0.1632 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
