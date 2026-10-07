---
type: experiment
id: EXP-038
created: 2026-10-07
status: confirmed
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-038 fine-tuned Qwen (EXP-031 adapter, trained on all subsets) on the closed-pool subsets: does it reproduce canned answers?

- run: `exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e` · git `e53a3d3` · status **ok** · 2057.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e` · changed: adapter, mode
- config: `{"mode": "rag_few_shot", "adapter": "run:exp031_lora_rag_qwen257bin_sub2_nhall-vall_5c414e", "few_shot_k": 3, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **Eng_Eth, Eng_Uga**

**held_out**: combined 0.5076, test-mix 0.4923, R1 0.6968, RL 0.6750, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.5409 |
| Eng_Ken | 0.5691 |
| Eng_Uga | 0.5867 |
| Lug_Uga | 0.2880 |
| Swa_Ken | 0.4502 |

**val**: combined 0.4866, test-mix 0.4874, R1 0.6688, RL 0.6464, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4777 |
| Eng_Ken | 0.5411 |
| Eng_Uga | 0.5851 |
| Lug_Uga | 0.2886 |
| Swa_Ken | 0.4577 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
