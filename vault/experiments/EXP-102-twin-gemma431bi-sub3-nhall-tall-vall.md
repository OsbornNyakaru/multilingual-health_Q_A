---
type: experiment
id: EXP-102
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-102 EXP-098 twins translated by NLLB-3.3B fine-tuned on the pool's own twin pairs (mutual matches, cosine >= 0.9; aligned sentences, both directions; Akan<->English, Amharic<->English), trained once in setup on the most honest pool; held-out + Val + test in one run

- run: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · git `57a70c9` · status **ok** · 2596.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp098_twin_gemma431bi_sub4_nhall-vall_fb1199` · changed: twin_train
- config: `{"mode": "twin", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_llm": true, "twin_train": true}` (non-default keys)
- eval: {'held_out': 0, 'val': 0, 'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.3412, test-mix 0.3519, R1 0.4888, RL 0.4335, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.3612 |
| Amh_Eth | 0.2628 |
| Eng_Gha | 0.3537 |

**val**: combined 0.3389, test-mix 0.3496, R1 0.4856, RL 0.4303, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.3476 |
| Amh_Eth | 0.2609 |
| Eng_Gha | 0.3626 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
