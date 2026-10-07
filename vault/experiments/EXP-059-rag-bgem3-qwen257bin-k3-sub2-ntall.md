---
type: experiment
id: EXP-059
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-059 test predictions for EXP-038 on Lug_Uga + Swa_Ken (generator feature for the learned final ranker)

- run: `exp059_rag_bgem3_qwen257bin_k3_sub2_ntall_24605c` · git `7f555d3` · status **ok** · 431.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e` · changed: —
- config: `{"mode": "rag_few_shot", "adapter": "run:exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'test': 0} · subsets: ['Lug_Uga', 'Swa_Ken']
- adopted for subsets: **none**

## Links
- [[00_INDEX]]
