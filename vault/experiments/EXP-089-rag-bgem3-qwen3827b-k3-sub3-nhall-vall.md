---
type: experiment
id: EXP-089
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-089 EXP-084 rerun with paragraphs=join: Qwen3.8 answers in structured lists and postprocess kept only the first paragraph (median 12-21 words, Val 0.08-0.16); also stores raw candidates (raw_cands) for offline re-postprocessing

- run: `exp089_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_6c34df` · git `d1dd6b0` · status **ok** · 7445.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp084_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_e80aaa` · changed: paragraphs
- config: `{"mode": "rag_few_shot", "model_id": "Qwen/Qwen3.8-27B", "precision": "bf16", "adapter": "run:exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "vllm_max_seqs": 256, "few_shot_k": 3, "paragraphs": "join", "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.1367, test-mix 0.1414, R1 0.2146, RL 0.1549, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.1042 |
| Amh_Eth | 0.1025 |
| Eng_Gha | 0.1835 |

**val**: combined 0.1381, test-mix 0.1406, R1 0.2155, RL 0.1578, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.1009 |
| Amh_Eth | 0.1208 |
| Eng_Gha | 0.1829 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
