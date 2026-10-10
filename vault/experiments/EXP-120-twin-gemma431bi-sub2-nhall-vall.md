---
type: experiment
id: EXP-120
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-120 Twin translator trained on mutual twin pairs with cosine >= 0.8 instead of 0.9 (more pairs)

- run: `exp120_twin_gemma431bi_sub2_nhall-vall_aabacc` · git `7a2b4a5` · status **ok** · 1272.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: twin_train_min_sim
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true, "twin_train_min_sim": 0.8, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3635, test-mix 0.3636, R1 0.5226, RL 0.4599, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3648 |
| Eng_Gha | 0.3623 |

**val**: combined 0.3637, test-mix 0.3637, R1 0.5232, RL 0.4599, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3561 |
| Eng_Gha | 0.3714 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
