---
type: experiment
id: EXP-045
created: 2026-10-07
status: confirmed
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-045 agreement: learned selector (EXP-039) unless fine-tuned Qwen (EXP-038) names a stored answer in the selector's top-k (k tuned on held-out)

- run: `exp045_combine_agree_sub2_494619` · git `0a8c54b` · status **ok** · 0 s on none (offline combine)
- parent: `exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df` · changed: combine
- config: `{"combine": {"rule": "agree", "selector": "exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df", "gen": "exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e", "k": {"Eng_Uga": 3, "Eng_Ken": 10}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Eng_Ken']
- adopted for subsets: **Eng_Ken, Eng_Uga**

**held_out**: combined 0.6129, test-mix 0.6134, R1 0.8357, RL 0.8208, n=680

| subset | combined |
|---|--:|
| Eng_Ken | 0.6029 |
| Eng_Uga | 0.6157 |

**val**: combined 0.6063, test-mix 0.6064, R1 0.8271, RL 0.8115, n=2078

| subset | combined |
|---|--:|
| Eng_Ken | 0.5902 |
| Eng_Uga | 0.6100 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
