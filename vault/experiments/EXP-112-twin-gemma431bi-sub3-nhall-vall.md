---
type: experiment
id: EXP-112
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-112 Twin translator fine-tuned 2 epochs instead of 1, 4 beams (NLLB answers only)

- run: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · git `9472a76` · status **ok** · 1054.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · changed: translate_beams, twin_llm, twin_train_epochs
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **Aka_Gha**

**held_out**: combined 0.3441, test-mix 0.3566, R1 0.4930, RL 0.4370, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.3648 |
| Amh_Eth | 0.2524 |
| Eng_Gha | 0.3613 |

**val**: combined 0.3413, test-mix 0.3541, R1 0.4885, RL 0.4341, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.3531 |
| Amh_Eth | 0.2486 |
| Eng_Gha | 0.3682 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
