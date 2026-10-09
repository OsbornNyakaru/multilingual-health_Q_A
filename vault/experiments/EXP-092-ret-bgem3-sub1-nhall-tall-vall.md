---
type: experiment
id: EXP-092
created: 2026-10-09
status: rejected
links: ["[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# EXP-092 translate-then-retrieve on Swa_Ken (NLLB swh_Latn -> English; Lug_Uga gained Val +0.019 via EXP-091); held-out + Val + test in one run so the ranker can record straight away

- run: `exp092_ret_bgem3_sub1_nhall-tall-vall_1ba790` · git `3f3b977` · status **ok** · 130.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp088_ret_bgem3_sub1_nhall-vall_a31b27` · changed: translate_subsets
- config: `{"translate_model": "facebook/nllb-200-distilled-1.3B", "translate_subsets": ["Swa_Ken"], "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0, 'test': 0} · subsets: ['Swa_Ken']
- adopted for subsets: **none**

**held_out**: combined 0.5639, test-mix 0.5639, R1 0.7710, RL 0.7532, n=145

| subset | combined |
|---|--:|
| Swa_Ken | 0.5639 |

**val**: combined 0.5671, test-mix 0.5671, R1 0.7761, RL 0.7565, n=518

| subset | combined |
|---|--:|
| Swa_Ken | 0.5671 |

## Links
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
