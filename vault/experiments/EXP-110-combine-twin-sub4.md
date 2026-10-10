---
type: experiment
id: EXP-110
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-110 Twin router for the closed subsets: EXP-108's translated twin answer (Lug<->Eng_Uga, Swa<->Eng_Ken) when the twin match is confident, else the best ranker (EXP-091/062/106)

- run: `exp110_combine_twin_sub4_968b27` · git `760f6e8` · status **ok** · 0 s on none (offline combine)
- parent: `exp108_twin_gemma431bi_sub4_nhall-tall-vall_30b0c4` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp108_twin_gemma431bi_sub4_nhall-tall-vall_30b0c4", "test_twin": "exp108_twin_gemma431bi_sub4_nhall-tall-vall_30b0c4", "base": {"Lug_Uga": "exp091_combine_ltr_sub1_489d9a", "Eng_Uga": "exp062_combine_ltr_sub4_5f85f5", "Swa_Ken": "exp062_combine_ltr_sub4_5f85f5", "Eng_Ken": "exp106_combine_ltr_sub4_1cb7a1"}, "route": {"Lug_Uga": {"source": "nllb", "min_sim": null}, "Eng_Uga": {"source": "nllb", "min_sim": 0.96}, "Swa_Ken": {"source": "nllb", "min_sim": null}, "Eng_Ken": {"source": "nllb", "min_sim": 0.92}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Lug_Uga', 'Eng_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **none**

**held_out**: combined 0.6074, test-mix 0.6046, R1 0.8274, RL 0.8141, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.6370 |
| Eng_Uga | 0.6312 |
| Lug_Uga | 0.5205 |
| Swa_Ken | 0.6320 |

**val**: combined 0.5971, test-mix 0.5970, R1 0.8141, RL 0.7996, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.6215 |
| Eng_Uga | 0.6262 |
| Lug_Uga | 0.5118 |
| Swa_Ken | 0.6232 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
