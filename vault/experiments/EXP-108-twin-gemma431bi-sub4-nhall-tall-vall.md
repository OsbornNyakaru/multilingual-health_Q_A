---
type: experiment
id: EXP-108
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-108 Twins for the closed subsets: Luganda<->Uganda English and Swahili<->Kenya English rows answered by the twin's gold answer, translated by NLLB fine-tuned on the pool's own twin pairs (as EXP-102); a router decides per subset against the rankers

- run: `exp108_twin_gemma431bi_sub4_nhall-tall-vall_30b0c4` · git `b288648` · status **ok** · 693.2 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · changed: twin_llm
- config: `{"mode": "twin", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true}` (non-default keys)
- eval: {'held_out': 0, 'val': 0, 'test': 0} · subsets: ['Lug_Uga', 'Eng_Uga', 'Swa_Ken', 'Eng_Ken']
- adopted for subsets: **none**

**held_out**: combined 0.3302, test-mix 0.3269, R1 0.4658, RL 0.4267, n=1062

| subset | combined |
|---|--:|
| Eng_Ken | 0.5430 |
| Eng_Uga | 0.2453 |
| Lug_Uga | 0.2693 |
| Swa_Ken | 0.5283 |

**val**: combined 0.3353, test-mix 0.3346, R1 0.4727, RL 0.4335, n=3442

| subset | combined |
|---|--:|
| Eng_Ken | 0.5569 |
| Eng_Uga | 0.2461 |
| Lug_Uga | 0.2890 |
| Swa_Ken | 0.5348 |

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
