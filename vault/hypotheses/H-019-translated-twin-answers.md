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
- EXP-102 (NLLB-3.3B fine-tuned in setup on the pool's mutual twin pairs, sentence-aligned, both directions) + EXP-104 router: Val Aka 0.261 → 0.347, Eng_Gha 0.345 → 0.390, Amh 0.246 → 0.266 vs EXP-100. Leaderboard #15: public 0.760497, private 0.747843 (+0.036 on both vs #14; 1st place was 0.7292 private). Judge 0.8352 (up from 0.8222).
- Learned Akan spelling normalisation ("-ɔ/-ɛ" endings) on the fine-tuned output: ±0.0006, dead end.
- #16: test twins from the translator retrained on Train + Val (EXP-107, routed by EXP-109 with EXP-104's thresholds) + Eng_Ken ranker EXP-106: public 0.762889, private 0.751234 (+0.0024 / +0.0034 vs #15). Retraining on all data for test pays off.
- Closed subsets (EXP-108, EXP-110 router): twin answers alone lose to the rankers (Val Lug 0.289 vs 0.512, Eng_Uga 0.246 vs 0.627, Swa 0.535 vs 0.623, Eng_Ken 0.557 vs 0.626); the router never routes Lug/Swa, and its Eng_Uga/Eng_Ken thresholds gain on held-out but lose on Val. Canned answers are better copied than translated; mutual twins are scarce where questions repeat (Lug<->Eng_Uga 501, Swa<->Eng_Ken 817 pairs). Untested: the twin translation as a ranker feature.

## 2026-10-10: translator tuning and a router with more signals

- EXP-111 (4 beams) and EXP-112 (2 epochs + 4 beams): 2 epochs help Akan and Ghana English (unrouted: Aka 0.3612/0.3476 -> 0.3648/0.3531, Eng_Gha 0.3537/0.3626 -> 0.3613/0.3682 on held-out/Val) and hurt Amharic (0.2628/0.2609 -> 0.2524/0.2486).
- Submission #17 (EXP-115 router, EXP-114 test twins): 0.768251 public / 0.756522 private, +0.0054 over #16. The LLM judge rose 0.8333 -> 0.8451: a better translator pays more on the leaderboard than local ROUGE shows.
- EXP-117, `combine.py twin --rich`: route on `twin_sim + margin_w * twin_margin + agree_w * agree`, and rows that lose a shared twin fall back. Eng_Gha 0.3938/0.3941 -> 0.4063/0.4109 (margin 2, agreement 0.4, losers dropped); Aka_Gha 0.3639/0.3528 -> 0.3664/0.3547 (margin 2). The same rule wins when tuned on Val and checked on held-out.
- Twin matching is the bottleneck now: on Val 9% of Akan and 12% of Ghana English rows lose a shared twin, and those rows score 0.15-0.18 with the twin answer against 0.21-0.27 with Gemma's.
- Amharic: the translator turns the "This is a question about, X." opening of Ethiopia English answers into noise. Dropping the first predicted sentence on CPU gives only +0.003 on EXP-102; EXP-122 removes the opening before translating and in the training pairs.
- Queued: EXP-116 (3 epochs), EXP-118 (skip paired candidates), EXP-119 (+ one-to-one), EXP-120 (pairs at cosine >= 0.8), EXP-121 (question pairs), EXP-122 (Amharic prefix).
