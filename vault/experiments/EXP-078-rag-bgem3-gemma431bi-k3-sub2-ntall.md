---
type: experiment
id: EXP-078
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-078 EXP-077 test predictions (Aka_Gha, Eng_Gha): 13-candidate MBR via vLLM on EXP-061's adapter

- run: `exp078_rag_bgem3_gemma431bi_k3_sub2_ntall_9a412a` · git `9aee5d0` · status **ok** · 2470.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: —
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
