---
type: experiment
id: EXP-077
created: 2026-10-08
status: confirmed
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-077 13-candidate MBR on EXP-061's adapter, all held-out + Val, via vLLM (sampler fix)

- run: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · git `67837df` · status **ok** · 6756.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp075_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_197dc2` · changed: —
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **Aka_Gha, Eng_Gha**

**held_out**: combined 0.2318, test-mix 0.2424, R1 0.3604, RL 0.2661, n=752

| subset | combined |
|---|--:|
| Aka_Gha | 0.2216 |
| Amh_Eth | 0.1546 |
| Eng_Gha | 0.2741 |

**val**: combined 0.2348, test-mix 0.2426, R1 0.3647, RL 0.2699, n=2680

| subset | combined |
|---|--:|
| Aka_Gha | 0.2162 |
| Amh_Eth | 0.1789 |
| Eng_Gha | 0.2770 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
