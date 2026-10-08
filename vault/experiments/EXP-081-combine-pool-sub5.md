---
type: experiment
id: EXP-081
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-081 pool: best run's answer (weight w) + EXP-079's 13 Gemma candidates

- run: `exp081_combine_pool_sub5_80a5ef` · git `24ed879` · status **ok** · 0 s on none (offline combine)
- parent: `exp079_rag_bgem3_gemma431bi_k3_sub5_nhall-vall_94b2de` · changed: combine
- config: `{"combine": {"rule": "pool", "gen": "exp079_rag_bgem3_gemma431bi_k3_sub5_nhall-vall_94b2de", "base": {"Eng_Uga": "exp062_combine_ltr_sub4_5f85f5", "Lug_Uga": "exp060_combine_ltr_sub4_eed26c", "Swa_Ken": "exp062_combine_ltr_sub4_5f85f5", "Eng_Ken": "exp062_combine_ltr_sub4_5f85f5", "Eng_Eth": "exp038_rag_bgem3_qwen257bin_k3_sub5_nhall-vall_11871e"}, "w": {"Eng_Uga": null, "Lug_Uga": null, "Swa_Ken": null, "Eng_Ken": null, "Eng_Eth": null}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.5903, test-mix 0.5981, R1 0.8049, RL 0.7906, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.5409 |
| Eng_Ken | 0.6270 |
| Eng_Uga | 0.6299 |
| Lug_Uga | 0.5104 |
| Swa_Ken | 0.6320 |

**val**: combined 0.5760, test-mix 0.5876, R1 0.7862, RL 0.7705, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4777 |
| Eng_Ken | 0.6152 |
| Eng_Uga | 0.6268 |
| Lug_Uga | 0.4932 |
| Swa_Ken | 0.6232 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
