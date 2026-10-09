---
type: experiment
id: EXP-094
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-094 MBR pool: current best answer (weighted) + EXP-089 Qwen3.8-27B LoRA candidates (13 per row) on the generation subsets

- run: `exp094_combine_pool_sub3_7490b4` · git `12ae299` · status **ok** · 0 s on none (offline combine)
- parent: `exp089_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_6c34df` · changed: combine
- config: `{"combine": {"rule": "pool", "gen": "exp089_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_6c34df", "base": {"Aka_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Eng_Gha": "exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01", "Amh_Eth": "exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1"}, "w": {"Aka_Gha": null, "Eng_Gha": null, "Amh_Eth": null}}}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.2313, test-mix 0.2422, R1 0.3596, RL 0.2655, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.2216 |
| Amh_Eth | 0.1516 |
| Eng_Gha | 0.2741 |

**val**: combined 0.2352, test-mix 0.2427, R1 0.3652, RL 0.2705, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.2162 |
| Amh_Eth | 0.1810 |
| Eng_Gha | 0.2770 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
