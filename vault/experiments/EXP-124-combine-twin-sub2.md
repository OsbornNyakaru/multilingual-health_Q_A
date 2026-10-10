---
type: experiment
id: EXP-124
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-124 Rich twin router on EXP-119 (paired candidates skipped, one-to-one twins)

- run: `exp124_combine_twin_sub2_c60ff1` · git `b765a24` · status **ok** · 0 s on none (offline combine)
- parent: `exp119_twin_gemma431bi_sub2_nhall-vall_5a2c15` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp119_twin_gemma431bi_sub2_nhall-vall_5a2c15", "test_twin": null, "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.92, "margin_w": 2.0, "agree_w": 0.4, "drop_losers": false}, "Eng_Gha": {"source": "nllb", "min_sim": 0.98, "margin_w": 1.0, "agree_w": 0.4, "drop_losers": false}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **Aka_Gha, Eng_Gha**

**held_out**: combined 0.3877, test-mix 0.3877, R1 0.5570, RL 0.4908, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3672 |
| Eng_Gha | 0.4083 |

**val**: combined 0.3859, test-mix 0.3860, R1 0.5551, RL 0.4879, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3565 |
| Eng_Gha | 0.4155 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
