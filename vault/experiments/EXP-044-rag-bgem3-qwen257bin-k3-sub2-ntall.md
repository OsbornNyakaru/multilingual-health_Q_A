---
type: experiment
id: EXP-044
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-044 test predictions for EXP-038 on Eng_Uga + Eng_Ken (input to the agreement combiner)

- run: `exp044_rag_bgem3_qwen257bin_k3_sub2_ntall_35d725` · git `cde0454` · status **ok** · 249.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e` · changed: —
- config: `{"mode": "rag_few_shot", "adapter": "run:exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Uga', 'Eng_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
