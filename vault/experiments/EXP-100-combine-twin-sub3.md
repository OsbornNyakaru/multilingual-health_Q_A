---
type: experiment
id: EXP-100
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-100 Twin router: the twin's translated gold answer (NLLB, or Gemma for Amharic; source and min twin_sim tuned per subset on held-out) when the twin match is confident, else EXP-077/EXP-061

- run: `exp100_combine_twin_sub3_5777e0` · git `9af0eaf` · status **ok** · 0 s on none (offline combine)
- parent: `exp098_twin_gemma431bi_sub4_nhall-vall_fb1199` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp098_twin_gemma431bi_sub4_nhall-vall_fb1199", "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Amh_Eth": "exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.84}, "Eng_Gha": {"source": "nllb", "min_sim": 0.86}, "Amh_Eth": {"source": "llm", "min_sim": 0.67}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **Aka_Gha, Amh_Eth, Eng_Gha**

**held_out**: combined 0.2965, test-mix 0.3030, R1 0.4372, RL 0.3641, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.2682 |
| Amh_Eth | 0.2492 |
| Eng_Gha | 0.3445 |

**val**: combined 0.2928, test-mix 0.2995, R1 0.4314, RL 0.3601, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.2610 |
| Amh_Eth | 0.2455 |
| Eng_Gha | 0.3447 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
