---
type: experiment
id: EXP-103
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-103 MBR over the EXP-100 twin-router answer (weighted w, tuned per subset on held-out) + EXP-077's 13 Gemma candidates: a fluent in-style generation that agrees with the twin's content

- run: `exp103_combine_pool_sub2_644b32` · git `ade0241` · status **ok** · 0 s on none (offline combine)
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: combine
- config: `{"combine": {"rule": "pool", "gen": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "base": {"Aka_Gha": "exp100_combine_twin_sub3_5777e0", "Eng_Gha": "exp100_combine_twin_sub3_5777e0"}, "w": {"Aka_Gha": null, "Eng_Gha": null}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3063, test-mix 0.3063, R1 0.4563, RL 0.3714, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.2682 |
| Eng_Gha | 0.3445 |

**val**: combined 0.3027, test-mix 0.3028, R1 0.4503, RL 0.3678, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.2610 |
| Eng_Gha | 0.3447 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
