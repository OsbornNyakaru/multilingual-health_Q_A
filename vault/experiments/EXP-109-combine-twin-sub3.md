---
type: experiment
id: EXP-109
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-109 EXP-104 twin router (thresholds tuned on EXP-102 held-out) with test twins translated by the translator retrained on Train + Val (EXP-107)

- run: `exp109_combine_twin_sub3_47d7b6` · git `63cb659` · status **ok** · 0 s on none (offline combine)
- parent: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · changed: combine
- config: `{"combine": {"rule": "twin", "twin": "exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948", "test_twin": "exp107_twin_gemma431bi_sub3_ntall_5a9d91", "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Amh_Eth": "exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1"}, "route": {"Aka_Gha": {"source": "nllb", "min_sim": 0.63}, "Eng_Gha": {"source": "nllb", "min_sim": 0.83}, "Amh_Eth": {"source": "nllb", "min_sim": 0.67}}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.3558, test-mix 0.3678, R1 0.5110, RL 0.4507, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.3603 |
| Amh_Eth | 0.2685 |
| Eng_Gha | 0.3876 |

**val**: combined 0.3506, test-mix 0.3623, R1 0.5034, RL 0.4440, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.3472 |
| Amh_Eth | 0.2656 |
| Eng_Gha | 0.3895 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
