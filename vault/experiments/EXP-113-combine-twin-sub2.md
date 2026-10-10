---
type: experiment
id: EXP-113
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-113 Twin router on EXP-112 (translator fine-tuned 2 epochs, 4 beams), else EXP-077

- run: `exp113_combine_twin_sub2_5ee429` · git `599aadd` · status **ok** · 0 s on none (offline combine)
- parent: `exp112_twin_gemma431bi_sub3_nhall-vall_e257ce` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp112_twin_gemma431bi_sub3_nhall-vall_e257ce", "test_twin": null, "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.63}, "Eng_Gha": {"source": "nllb", "min_sim": 0.83}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **Eng_Gha**

**held_out**: combined 0.3788, test-mix 0.3788, R1 0.5474, RL 0.4764, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.3639 |
| Eng_Gha | 0.3938 |

**val**: combined 0.3734, test-mix 0.3734, R1 0.5380, RL 0.4711, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.3528 |
| Eng_Gha | 0.3941 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
