---
type: experiment
id: EXP-063
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-063 test predictions for EXP-061 (Gemma-4-31B LoRA adapter reused; pool = Train + Val)

- run: `exp063_rag_bgem3_gemma431bi_k3_sub3_ntall_353827` · git `c1285e8` · status **ok** · 3174.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1` · changed: adapter, mode
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
