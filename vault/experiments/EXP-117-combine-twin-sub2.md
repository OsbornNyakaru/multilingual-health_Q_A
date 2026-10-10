---
type: experiment
id: EXP-117
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-117 Twin router with more signals on EXP-112 (twin margin, agreement with the Gemma answer, rows that lose a shared twin fall back), test twins from EXP-114

- run: `exp117_combine_twin_sub2_5dc5a1` · git `b8574ee` · status **ok** · 0 s on none (offline combine)
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp112_twin_gemma431bi_sub3_nhall-vall_e257ce", "test_twin": "exp114_twin_gemma431bi_sub2_ntall_1be16e", "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.81, "margin_w": 2.0, "agree_w": 0.0, "drop_losers": false}, "Eng_Gha": {"source": "nllb", "min_sim": 0.99, "margin_w": 2.0, "agree_w": 0.4, "drop_losers": true}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **Eng_Gha**

**held_out**: combined 0.3863, test-mix 0.3863, R1 0.5571, RL 0.4871, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3664 |
| Eng_Gha | 0.4063 |

**val**: combined 0.3827, test-mix 0.3828, R1 0.5510, RL 0.4833, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3547 |
| Eng_Gha | 0.4109 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
