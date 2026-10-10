---
type: experiment
id: EXP-107
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-107 Test predictions for EXP-102 with the twin translator trained on the test pool (Train + Val, ~35% more twin pairs than work_train); routed with EXP-104's thresholds

- run: `exp107_twin_gemma431bi_sub3_ntall_5a9d91` · git `b288648` · status **ok** · 698.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp102_twin_gemma431bi_sub3_nhall-tall-vall_238948` · changed: twin_llm
- config: `{"mode": "twin", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
