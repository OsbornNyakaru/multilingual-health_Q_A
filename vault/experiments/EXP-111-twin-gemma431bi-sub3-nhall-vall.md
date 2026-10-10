---
type: experiment
id: EXP-111
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-111 EXP-102's fine-tuned twin translator decoding with 4 beams instead of 2 (NLLB answers only; the router picks NLLB on all three subsets)

- run: `exp111_twin_gemma431bi_sub3_nhall-vall_b23aab` · git `9472a76` · status **ok** · 768.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · changed: translate_beams, twin_llm
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.3440, test-mix 0.3542, R1 0.4920, RL 0.4377, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.3621 |
| Amh_Eth | 0.2694 |
| Eng_Gha | 0.3568 |

**val**: combined 0.3403, test-mix 0.3520, R1 0.4871, RL 0.4325, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.3492 |
| Amh_Eth | 0.2556 |
| Eng_Gha | 0.3667 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
