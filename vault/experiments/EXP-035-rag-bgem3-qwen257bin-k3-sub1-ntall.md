---
type: experiment
id: EXP-035
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-035 test predictions for EXP-031 (Eng_Gha), reusing its adapter; pool = Train + Val

- run: `exp035_rag_bgem3_qwen257bin_k3_sub1_ntall_956f60` · git `711415f` · status **ok** · 160.8 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e` · changed: adapter, mode
- config: `{"mode": "rag_few_shot", "adapter": "run:exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Gha']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
