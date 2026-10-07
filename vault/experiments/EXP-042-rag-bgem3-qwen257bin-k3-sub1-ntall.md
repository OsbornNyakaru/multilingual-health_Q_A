---
type: experiment
id: EXP-042
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-042 test predictions for EXP-038 (Eng_Eth, fine-tuned Qwen adapter)

- run: `exp042_rag_bgem3_qwen257bin_k3_sub1_ntall_c35f57` · git `d654060` · status **ok** · 5.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e` · changed: —
- config: `{"mode": "rag_few_shot", "adapter": "run:exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Eth']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
