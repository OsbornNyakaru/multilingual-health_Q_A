---
type: experiment
id: EXP-061
created: 2026-10-08
status: confirmed
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-061 1st-place generator size: Gemma-4-31B QLoRA (4-bit, r64) with our RAG k=3 prompt, trained only on Aka_Gha/Eng_Gha/Amh_Eth, 1 epoch; generation on the bf16 base with the merged adapter

- run: `exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1` · git `f81ad94` · status **ok** · 23828.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e` · changed: infer_batch, lora_batch, lora_bits, lora_grad_acc, lora_save_steps, lora_train_subsets, model_id, precision
- config: `{"mode": "lora_rag", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "infer_batch": 16, "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **Aka_Gha, Amh_Eth, Eng_Gha**

**held_out**: combined 0.2251, test-mix 0.2352, R1 0.3469, RL 0.2615, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.2098 |
| Amh_Eth | 0.1516 |
| Eng_Gha | 0.2710 |

**val**: combined 0.2258, test-mix 0.2321, R1 0.3479, RL 0.2624, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.1988 |
| Amh_Eth | 0.1810 |
| Eng_Gha | 0.2718 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
