---
type: experiment
id: EXP-021
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-021 rag few-shot: Qwen2.5-7B writes answers after 2 nearest bge-m3 Q&A pairs (Aka_Gha, Eng_Gha)

- run: `exp021_rag_bgem3_qwen257bin_k2_sub2_nhall-vall_a26d61` · git `c9508b1` · status **ok** · 1224.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp009_ret_bgem3_nhall-vall_ad6575` · changed: mode
- config: `{"mode": "rag_few_shot", "no_repeat_ngram": 3}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**held_out**: combined 0.0983, test-mix 0.0983, R1 0.1500, RL 0.1156, n=623

| subset | combined |
|---|--:|
| Aka_Gha | 0.0725 |
| Eng_Gha | 0.1242 |

**val**: combined 0.0983, test-mix 0.0984, R1 0.1505, RL 0.1152, n=2218

| subset | combined |
|---|--:|
| Aka_Gha | 0.0736 |
| Eng_Gha | 0.1232 |

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
