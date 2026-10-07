---
type: experiment
id: EXP-034
created: 2026-10-07
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-034 Akan: EXP-033 + fall back to the retrieved answer when a generation is under half the typical length

- run: `exp034_rag_bgem3_afriquella_k3_sub1_nhall-vall_960555` · git `711415f` · status **ok** · 1592.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `EXP-033` · changed: fallback_below_frac
- config: `{"mode": "rag_few_shot", "model_id": "McGill-NLP/AfriqueLlama-8B", "adapter": "run:exp032_lora_rag_afriquella_sub2_nhall-vall_a9b0c3", "few_shot_k": 3, "min_len_pct": 10, "fallback_below_frac": 0.5, "lora_epochs": 1, "lora_save_steps": 1500}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.1796, test-mix 0.1796, R1 0.2925, RL 0.1930, n=312

| subset | combined |
|---|--:|
| Aka_Gha | 0.1796 |

**val**: combined 0.1761, test-mix 0.1761, R1 0.2845, RL 0.1914, n=1114

| subset | combined |
|---|--:|
| Aka_Gha | 0.1761 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
