---
type: experiment
id: EXP-114
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-114 Test predictions for EXP-112: twin translator fine-tuned 2 epochs on the test pool (Train + Val), 4 beams; Akan and Ghana English

- run: `exp114_twin_gemma431bi_sub2_ntall_1be16e` · git `7f7ef9e` · status **ok** · 1347.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp107_twin_gemma431bi_sub3_ntall_5a9d91` · changed: translate_beams, twin_train_epochs
- config: `{"mode": "twin", "translate_beams": 4, "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_train": true, "twin_train_epochs": 2.0}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
