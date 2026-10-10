---
type: hypothesis
id: H-019
created: 2026-10-10
status: supported
links: ["[[H-014-bigger-generator]]", "[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# H-019 Answer open-subset rows with their twin's translated gold answer

**Statement:** Aka_Gha rows are row-by-row human translations of Eng_Gha rows: question and answer, sentence by sentence. Counts match in every split (Train 4,455 / 4,443, Val 1,114 / 1,104, Test 492 / 491), and the split assignment is independent, so the twin of a test row is in Train + Val about 92% of the time. Translating the twin's known gold answer should beat generating from scratch on our two weakest large subsets (Val about 0.22 and 0.28; 37.6% of the test mix together). The same may hold for Amh_Eth ↔ Eng_Eth.

## Evidence so far
- 2026-10-10 audit: confident rare-word pairs (anchor score ≥ 8, margin ≥ 4) are clear translations; four random pairs checked by hand (safe spaces, innovative technologies, menstrual cycle, media representations).

## Design (EXP-098 probe, `mode: twin`)
- Matching: NLLB-200-3.3B translates non-English questions to English; BGE-M3 cosine against the paired subset's pool questions; the best match is the twin (meta `twin_id`, `twin_sim`, `twin_margin`).
- Answer: the twin's gold answer translated into the row's language, by NLLB sentence by sentence (the prediction) and by base Gemma-4-31B via vLLM (meta `llm_answer`).
- The Val pool is Train only, so about 74% of Val rows have their twin in the pool. At test time (pool Train + Val) about 92% do.

## Tests
- Per subset: twin answers vs EXP-077/061 on held-out and Val, overall and by `twin_sim` / `twin_margin` band (a router: twin answer when confident, else the generation).
- NLLB vs Gemma translations on the same rows.
- If the twin answer clearly wins where the twin exists: matching at scale (one-to-one assignment) and a twin-aware Gemma (twin answer in the prompt).

## Links
- [[H-014-bigger-generator]]
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]

## Result (2026-10-10)
- EXP-098 probe, Val (R1+RL)/2: twin vs current Aka 0.348 vs 0.292, Eng_Gha 0.412 vs 0.374; Amharic best with Gemma's translation (0.325 vs 0.242). NLLB beats base Gemma on Akan and Ghana English.
- EXP-100 router (source + min twin_sim tuned on held-out): Val Aka 0.216 → 0.261, Eng_Gha 0.277 → 0.345, Amh 0.181 → 0.246. Test rows routed: Aka 71%, Eng_Gha 69%, Amh 93%.
- Leaderboard #14: public 0.724483, private 0.711607 (+0.028 on both vs #12). ROUGE-1 0.7074 / ROUGE-L 0.6729 public; judge 0.8222 (down 0.011).
- EXP-103 (MBR of the twin-router answer + EXP-077's 13 Gemma candidates, base weight w tuned on held-out): w = inf on both subsets, so any vote for a generation lowers the score. Translated gold content beats fluent generation; dead end.
