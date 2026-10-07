---
type: experiment
id: EXP-050
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-050 re-pick among the selector's top-20 answers with selector rank + retriever rank + question-to-answer word overlap (weights tuned on held-out)

- run: `exp050_combine_rescore_sub2_d7c391` · git `b0c6f06` · status **ok** · 0 s on none (offline combine)
- parent: `exp048_ret_bgem3_rrq_sub5_nhall-vall_9294c3` · changed: combine
- config: `{"combine": {"rule": "rescore", "base": "exp048_ret_bgem3_rrq_sub5_nhall-vall_9294c3", "weights": {"Lug_Uga": [0.25, 8.0], "Swa_Ken": [0.25, 4.0]}, "top": 20}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Lug_Uga', 'Swa_Ken']
- adopted for subsets: **Lug_Uga, Swa_Ken**

**held_out**: combined 0.5098, test-mix 0.5098, R1 0.6986, RL 0.6792, n=382

| subset | combined |
|---|--:|
| Lug_Uga | 0.4461 |
| Swa_Ken | 0.6138 |

**val**: combined 0.5173, test-mix 0.5173, R1 0.7083, RL 0.6899, n=1364

| subset | combined |
|---|--:|
| Lug_Uga | 0.4641 |
| Swa_Ken | 0.6042 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
