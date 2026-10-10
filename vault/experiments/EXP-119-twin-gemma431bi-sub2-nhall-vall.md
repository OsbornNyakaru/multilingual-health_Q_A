---
type: experiment
id: EXP-119
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-119 Skip paired twin candidates and assign twins one-to-one (highest total cosine)

- run: `exp119_twin_gemma431bi_sub2_nhall-vall_5a2c15` · git `7a2b4a5` · status **ok** · 913.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: twin_one_to_one, twin_skip_paired
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_skip_paired": true, "twin_one_to_one": true, "twin_train": true, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3611, test-mix 0.3611, R1 0.5184, RL 0.4576, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3597 |
| Eng_Gha | 0.3625 |

**val**: combined 0.3603, test-mix 0.3603, R1 0.5169, RL 0.4570, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3504 |
| Eng_Gha | 0.3703 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
