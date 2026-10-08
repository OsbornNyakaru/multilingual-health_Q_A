---
type: experiment
id: EXP-076
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-076 vLLM smoke test with the FlashInfer sampler off (its JIT build needed ninja, EXP-074)

- run: `exp076_rag_bgem3_gemma431bi_k3_sub3_nv150_3429fd` · git `67837df` · status **ok** · 527.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp074_rag_bgem3_gemma431bi_k3_sub3_nv150_55e631` · changed: —
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'val': 150} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**val**: combined 0.2570, test-mix 0.2618, R1 0.3933, RL 0.3011, n=150

| subset | combined |
|---|--:|
| Aka_Gha | 0.2333 |
| Amh_Eth | 0.2218 |
| Eng_Gha | 0.2954 |

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
