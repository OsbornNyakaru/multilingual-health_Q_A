---
type: experiment
id: EXP-020
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-020 SMOKE: rag few-shot generation path on 40 held-out rows (Qwen2.5-7B, k=2)

- run: `exp020_rag_bgem3_qwen257bin_k2_sub2_nh40_4658cc` · git `c9508b1` · status **ok** · 82.8 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp009_ret_bgem3_nhall-vall_ad6575` · changed: mode
- config: `{"mode": "rag_few_shot"}` (non-default keys)
- eval: {'held_out': 40} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.0872, test-mix 0.0872, R1 0.1330, RL 0.1027, n=40

| subset | combined |
|---|--:|
| Aka_Gha | 0.0718 |
| Eng_Gha | 0.1026 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
