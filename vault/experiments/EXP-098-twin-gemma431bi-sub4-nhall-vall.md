---
type: experiment
id: EXP-098
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-098 Twin probe: answer each Akan/Ghana-English/Amharic/Ethiopia-English row with the gold answer of its twin row in the paired subset (best English-question match in the pool), translated by NLLB-200-3.3B sentence by sentence; base Gemma-4-31B translations kept in meta (llm_answer) to compare translators

- run: `exp098_twin_gemma431bi_sub4_nhall-vall_fb1199` · git `96fa548` · status **ok** · 1795.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `—` · changed: gen_engine, mode, model_id, precision, twin_llm
- config: `{"mode": "twin", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_llm": true}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth', 'Eng_Eth']
- adopted for subsets: **Aka_Gha, Amh_Eth, Eng_Gha**

**held_out**: combined 0.2613, test-mix 0.2744, R1 0.3802, RL 0.3260, n=1026

| subset | combined |
|---|--:|
| Aka_Gha | 0.2608 |
| Amh_Eth | 0.2070 |
| Eng_Eth | 0.2433 |
| Eng_Gha | 0.3002 |

**val**: combined 0.2647, test-mix 0.2753, R1 0.3871, RL 0.3282, n=3244

| subset | combined |
|---|--:|
| Aka_Gha | 0.2578 |
| Amh_Eth | 0.1938 |
| Eng_Eth | 0.2571 |
| Eng_Gha | 0.3051 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
