---
type: fact
id: F-007
created: 2026-10-07
status: confirmed
links: ["[[F-003-reference-approach]]", "[[D-007-autoresearch-loop-on-molab]]", "[[00_INDEX]]"]
---
# F-007 How the top solutions worked (and how they differ from 11th place)

Sources: Zindi winners announcement (2026-09-10, pasted by the user); 1st place's public repo and README (github.com/DariusTheGeek/Multilingual-Health-QA-ITU-Zindi-Challenge, weights at huggingface.co/DariusTheGeek/mhqa-itu-adapters); the public model card huggingface.co/MagicCard/msrh-zindi-magic; our [[F-003-reference-approach]].

**Final podium (private):** 1st Darius Moruri 0.7292 · 2nd Linh Anh 0.7291 · 3rd Obiti 0.7290.

**1st, Darius: selection-first.** About 60% of the test set (Eng_Uga, Eng_Ken, Eng_Eth, Swa_Ken, Lug_Uga) picks a stored Train+Val answer; only Aka_Gha and Eng_Gha are generated. Picking: BGE-M3 candidates (Luganda pool expanded to BGE top-100 + BM25 top-50); a fine-tuned bge-reranker-v2-m3 trained on all of Train with graded targets, 5-fold cross-fitted and averaged, used as a feature in a LightGBM LambdaRank ranker, not as the chooser; three selectors (Swa/Eng_Ken, Eng_Uga/Eng_Eth, Luganda). Generation: LoRA adapters on gemma-4-31B / medgemma-27b with a cross-model MBR medoid. Reproduced public 0.7285 = R1 0.7187, RL 0.6514, judge 0.8522.

**2nd, Linh Anh (per Zindi): several fine-tuned Qwen models with different configs, combined; heavy validation and error analysis.** A public model card (MagicCard, claims private 0.7309, which matches no podium score, so it's unconfirmed as 2nd place) describes such an approach: 19 LoRA adapters over Qwen3.5-27B / Qwen3.6-27B / Qwen3-32B (r128, α256, 3 epochs), AfriE5 top-3–7 retrieved few-shot examples, a copy-verbatim "anchored extraction" prompt, and a per-row consensus pick. That's generation-centric: big models write every answer, including the repeated-answer subsets.

**11th, koleshjr (our starting reference):** BGE-M3 retrieval baseline (0.5756 by the formula), then a single RAG-enriched LoRA on Sunbird/Sunflower-32B (k=3, r64, dropout 0.5, 3 epochs). No selection layer, no routing, no ensemble.

**Our position (2026-10-07):** we follow 1st place's path (fine-tuned retriever + answer-aware selector + learned final ranker, LoRA generation for the non-repeating subsets). Best LB 0.66543 (R1 0.641, RL 0.594, judge 0.802). Largest structural gaps vs 1st: selection accuracy on the repeated-answer subsets, and generation model size (7–8B vs 27–32B).

## Links
- [[F-003-reference-approach]]
- [[D-007-autoresearch-loop-on-molab]]
- [[00_INDEX]]
