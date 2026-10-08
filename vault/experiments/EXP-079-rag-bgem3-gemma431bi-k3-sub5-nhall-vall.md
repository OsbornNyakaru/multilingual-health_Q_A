---
type: experiment
id: EXP-079
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-079 13-candidate Gemma (EXP-061 adapter, vLLM) on the five retrieval subsets: candidates to pool with the ranker's pick

- run: `exp079_rag_bgem3_gemma431bi_k3_sub5_nhall-vall_94b2de` · git `c7beaab` · status **ok** · 6101.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: —
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.3667, test-mix 0.3429, R1 0.5111, RL 0.4799, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.4616 |
| Eng_Ken | 0.3383 |
| Eng_Uga | 0.4054 |
| Lug_Uga | 0.2044 |
| Swa_Ken | 0.3386 |

**val**: combined 0.3455, test-mix 0.3368, R1 0.4840, RL 0.4497, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4166 |
| Eng_Ken | 0.3538 |
| Eng_Uga | 0.3922 |
| Lug_Uga | 0.1977 |
| Swa_Ken | 0.3509 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
