---
type: experiment
id: EXP-099
created: 2026-10-10
status: rejected
links: ["[[H-019-translated-twin-answers]]", "[[00_INDEX]]"]
---
# EXP-099 Test predictions for EXP-098 (twin answers: NLLB, plus Gemma translations in meta); test pool Train + Val holds ~92% of twins

- run: `exp099_twin_gemma431bi_sub4_ntall_1cc0df` · git `4e5faf9` · status **ok** · 553.7 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp098_twin_gemma431bi_sub4_nhall-vall_fb1199` · changed: —
- config: `{"mode": "twin", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "gen_engine": "vllm", "twin_llm": true}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth', 'Eng_Eth']
- adopted for subsets: **none**

## Links
- [[H-019-translated-twin-answers]]
- [[00_INDEX]]
