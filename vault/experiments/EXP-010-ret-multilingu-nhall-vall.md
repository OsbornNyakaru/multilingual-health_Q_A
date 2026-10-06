---
type: experiment
id: EXP-010
created: 2026-10-06
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-010 dense retrieval with multilingual-e5-large (query:/passage: prefixes are part of the model)

- run: `exp010_ret_multilingu_nhall-vall_7bc6f6` · git `134aa90` · status **ok** · 78.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp007_ret_tfidfchar_nhall-vall_6d656b` · changed: embedder, passage_prefix, query_prefix
- config: `{"embedder": "intfloat/multilingual-e5-large", "query_prefix": "query: ", "passage_prefix": "passage: "}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: all
- adopted for subsets: **none**

**held_out**: combined 0.3687, test-mix 0.3614, R1 0.5220, RL 0.4745, n=2088

| subset | combined |
|---|--:|
| Aka_Gha | 0.1756 |
| Amh_Eth | 0.1020 |
| Eng_Eth | 0.4407 |
| Eng_Gha | 0.1681 |
| Eng_Ken | 0.5773 |
| Eng_Uga | 0.5450 |
| Lug_Uga | 0.3315 |
| Swa_Ken | 0.5177 |

**val**: combined 0.3553, test-mix 0.3639, R1 0.5049, RL 0.4555, n=6686

| subset | combined |
|---|--:|
| Aka_Gha | 0.1621 |
| Amh_Eth | 0.1153 |
| Eng_Eth | 0.4074 |
| Eng_Gha | 0.1747 |
| Eng_Ken | 0.5881 |
| Eng_Uga | 0.5375 |
| Lug_Uga | 0.3515 |
| Swa_Ken | 0.5511 |

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
