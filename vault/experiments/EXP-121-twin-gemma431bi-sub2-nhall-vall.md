---
type: experiment
id: EXP-121
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-121 Twin translator also trained on the twins' question pairs

- run: `exp121_twin_gemma431bi_sub2_nhall-vall_19751a` · git `7a2b4a5` · status **ok** · 1070.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: twin_train_questions
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true, "twin_train_questions": true, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3592, test-mix 0.3592, R1 0.5172, RL 0.4535, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3613 |
| Eng_Gha | 0.3570 |

**val**: combined 0.3604, test-mix 0.3605, R1 0.5188, RL 0.4553, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3537 |
| Eng_Gha | 0.3673 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
