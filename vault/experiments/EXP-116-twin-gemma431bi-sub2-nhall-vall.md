---
type: experiment
id: EXP-116
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-116 Twin translator fine-tuned 3 epochs instead of 2 (4 beams); Akan and Ghana English

- run: `exp116_twin_gemma431bi_sub2_nhall-vall_e94fc8` · git `95ca0da` · status **ok** · 1226.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: twin_train_epochs
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true, "twin_train_epochs": 3.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3585, test-mix 0.3585, R1 0.5163, RL 0.4526, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3586 |
| Eng_Gha | 0.3584 |

**val**: combined 0.3578, test-mix 0.3578, R1 0.5152, RL 0.4518, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3512 |
| Eng_Gha | 0.3644 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
