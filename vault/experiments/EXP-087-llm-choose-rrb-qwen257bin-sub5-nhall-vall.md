---
type: experiment
id: EXP-087
created: 2026-10-09
status: rejected
links: ["[[H-016-fine-tuned-listwise-chooser]]", "[[00_INDEX]]"]
---
# EXP-087 fine-tuned listwise chooser: Qwen2.5-7B LoRA trained on leave-one-out top-5 lists from the pool (base BGE-M3 + reranker), target = option closest to the gold, random cyclic shifts; options show their matched Train question; inference on EXP-055's top-5 with shift averaging

- run: `exp087_llm_choose_rrb_qwen257bin_sub5_nhall-vall_50b2f1` · git `5780e60` · status **ok** · 952.8 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp085_llm_choose_gemma431bi_sub5_nhall-vall_d9248a` · changed: choose_questions, choose_train, choose_train_subsets, lora_batch, lora_dropout, lora_epochs, lora_lr, model_id, rerank_k, rerank_model, rerank_on
- config: `{"mode": "llm_choose", "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 30, "rerank_on": "both", "choose_from": "run:exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57", "choose_questions": true, "choose_train": true, "choose_train_subsets": ["Eng_Uga", "Lug_Uga", "Swa_Ken", "Eng_Ken", "Eng_Eth"], "lora_dropout": 0.05, "lora_lr": 0.0001, "lora_epochs": 1.0, "lora_batch": 8}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.5733, test-mix 0.5827, R1 0.7823, RL 0.7673, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.5066 |
| Eng_Ken | 0.6303 |
| Eng_Uga | 0.6390 |
| Lug_Uga | 0.4703 |
| Swa_Ken | 0.5687 |

**val**: combined 0.5653, test-mix 0.5745, R1 0.7721, RL 0.7556, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.4863 |
| Eng_Ken | 0.6112 |
| Eng_Uga | 0.6262 |
| Lug_Uga | 0.4767 |
| Swa_Ken | 0.5627 |

## Links
- [[H-016-fine-tuned-listwise-chooser]]
- [[00_INDEX]]
