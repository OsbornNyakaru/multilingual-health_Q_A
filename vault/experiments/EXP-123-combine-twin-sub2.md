---
type: experiment
id: EXP-123
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-123 Rich twin router on EXP-118 (paired twin candidates skipped)

- run: `exp123_combine_twin_sub2_959d3f` · git `2ce2787` · status **ok** · 0 s on none (offline combine)
- parent: `exp118_twin_gemma431bi_sub2_nhall-vall_ac4677` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp118_twin_gemma431bi_sub2_nhall-vall_ac4677", "test_twin": null, "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.87, "margin_w": 2.0, "agree_w": 0.0, "drop_losers": false}, "Eng_Gha": {"source": "nllb", "min_sim": 0.98, "margin_w": 1.0, "agree_w": 0.4, "drop_losers": false}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.3881, test-mix 0.3881, R1 0.5588, RL 0.4902, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3675 |
| Eng_Gha | 0.4088 |

**val**: combined 0.3807, test-mix 0.3808, R1 0.5490, RL 0.4799, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3528 |
| Eng_Gha | 0.4088 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
