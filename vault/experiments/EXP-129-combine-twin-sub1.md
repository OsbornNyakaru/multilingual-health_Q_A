---
type: experiment
id: EXP-129
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-129 Twin router (similarity only) on EXP-122 (Amharic, English answer prefix dropped)

- run: `exp129_combine_twin_sub1_de56ef` · git `6a32ae3` · status **ok** · 0 s on none (offline combine)
- parent: `exp122_twin_gemma431bi_sub1_nhall-vall_7b3ccb` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp122_twin_gemma431bi_sub1_nhall-vall_7b3ccb", "test_twin": null, "base": {"Amh_Eth": "exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1"}, "route": {"Amh_Eth": {"source": "nllb", "min_sim": 0.66, "margin_w": 0.0, "agree_w": 0.0, "drop_losers": false}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.2794, test-mix 0.2794, R1 0.3836, RL 0.3716, n=129

| subset | combined |
|---|--:|
| Amh_Eth | 0.2794 |

**val**: combined 0.2733, test-mix 0.2733, R1 0.3805, RL 0.3582, n=462

| subset | combined |
|---|--:|
| Amh_Eth | 0.2733 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
