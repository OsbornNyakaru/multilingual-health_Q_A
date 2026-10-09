---
type: experiment
id: EXP-085
created: 2026-10-09
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-085 PROBE: base Gemma-4-31B as a listwise chooser over EXP-055's top-5 distinct answers (option-letter logprobs, 5 cyclic shifts); does it pick the gold more often than the ranker?

- run: `exp085_llm_choose_gemma431bi_sub5_nhall-vall_d9248a` · git `3ad69ab` · status **ok** · 1178.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `—` · changed: choose_from, mode, model_id
- config: `{"mode": "llm_choose", "model_id": "google/gemma-4-31B-it", "choose_from": "run:exp055_ret_bgem3_rrb_sub5_nhall-vall_c31f57"}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**held_out**: combined 0.4849, test-mix 0.5064, R1 0.6683, RL 0.6423, n=1336

| subset | combined |
|---|--:|
| Eng_Eth | 0.3747 |
| Eng_Ken | 0.4907 |
| Eng_Uga | 0.5537 |
| Lug_Uga | 0.4438 |
| Swa_Ken | 0.5012 |

**val**: combined 0.4824, test-mix 0.5004, R1 0.6649, RL 0.6388, n=4006

| subset | combined |
|---|--:|
| Eng_Eth | 0.3310 |
| Eng_Ken | 0.4995 |
| Eng_Uga | 0.5350 |
| Lug_Uga | 0.4539 |
| Swa_Ken | 0.5093 |

## Links
- [[00_INDEX]]
