---
type: experiment
id: EXP-118
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-118 Twin candidates that already have their twin inside the pool are skipped (they cannot be the new row's twin)

- run: `exp118_twin_gemma431bi_sub2_nhall-vall_ac4677` · git `7a2b4a5` · status **ok** · 914.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: twin_skip_paired
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_skip_paired": true, "twin_train": true, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3608, test-mix 0.3608, R1 0.5188, RL 0.4564, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3591 |
| Eng_Gha | 0.3625 |

**val**: combined 0.3580, test-mix 0.3580, R1 0.5153, RL 0.4524, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3494 |
| Eng_Gha | 0.3667 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
