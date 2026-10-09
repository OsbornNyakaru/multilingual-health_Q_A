---
type: experiment
id: EXP-088
created: 2026-10-09
status: rejected
links: ["[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# EXP-088 translate-then-retrieve on Lug_Uga: NLLB-200 1.3B translates Luganda questions (pool + eval) to English; BGE-M3 on the original, the translation, and a 50/50 blend; records each view's top-50 for recall@k (FND-004)

- run: `exp088_ret_bgem3_sub1_nhall-vall_a31b27` · git `3413aa6` · status **ok** · 143.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `—` · changed: diag_k, translate_model
- config: `{"translate_model": "facebook/nllb-200-distilled-1.3B", "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Lug_Uga']
- adopted for subsets: **none**

**held_out**: combined 0.4073, test-mix 0.4073, R1 0.5620, RL 0.5390, n=237

| subset | combined |
|---|--:|
| Lug_Uga | 0.4073 |

**val**: combined 0.4200, test-mix 0.4200, R1 0.5777, RL 0.5575, n=846

| subset | combined |
|---|--:|
| Lug_Uga | 0.4200 |

## Links
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
