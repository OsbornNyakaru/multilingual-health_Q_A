---
type: experiment
id: EXP-122
created: 2026-10-10
status: confirmed
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-122 Amharic: drop the 'This is a question about, X.' opening of Ethiopia English twin answers before translating and in the training pairs (Amharic answers never have it)

- run: `exp122_twin_gemma431bi_sub1_nhall-vall_7b3ccb` · git `7a2b4a5` · status **ok** · 127.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp111_twin_gemma431bi_sub3_nhall-vall_b23aab` · changed: twin_drop_prefix
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_drop_prefix": "^This is a question about,[^.]*\\.\\s*", "twin_train": true}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Amh_Eth']
- adopted for subsets: **Amh_Eth**

**held_out**: combined 0.2744, test-mix 0.2744, R1 0.3769, RL 0.3647, n=129

| subset | combined |
|---|--:|
| Amh_Eth | 0.2744 |

**val**: combined 0.2711, test-mix 0.2711, R1 0.3774, RL 0.3553, n=462

| subset | combined |
|---|--:|
| Amh_Eth | 0.2711 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
