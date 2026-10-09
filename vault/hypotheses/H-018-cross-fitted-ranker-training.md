---
type: hypothesis
id: H-018
created: 2026-10-10
status: testing
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[H-017-translate-then-retrieve-luganda]]", "[[00_INDEX]]"]
---
# H-018 Train the final ranker on cross-fitted lists for the training questions

**Statement:** The final ranker (scripts/ltr.py) learns from held-out + Val only, about 5.3k questions on the closed subsets. The leaderboard has rejected two of our last three small Val gains, so per-subset Val noise (about ±0.003–0.005) is as large as what ranker tweaks bring. Training it on about 17.7k more questions, the work_train pool's own questions, should make its picks steadier and better. 1st place's largest single step (+0.0097 LB) came from training their selector on all Train pools.

## Design (EXP-097, `oof_folds`)
- 5 folds over work_train. For each fold, the EXP-055 pipeline (fine-tuned BGE-M3 + 3 answer-aware selectors) trains on the other 4 folds. Each fold question then retrieves from the whole pool minus itself.
- **Consistency (1st place's −0.0105 lesson):** held-out, Val and test are scored by the same 5 fold models, with scores averaged. They are never scored by a model refit on everything, so the ranker sees the same kind of features at training and at test time.
- Lists per question: `ret_ids` (retriever), `cand_ids` (3-selector ensemble), `cand1_ids` (seed-0 selector, standing in for EXP-051), and `tr_ids` (NLLB-translated view, Luganda).
- ltr.py builds the training questions' features leaving the question itself out of the pool (frequency and sibling similarity).

## Tests
- First, EXP-097's own picks against EXP-055 (do fold-averaged models lose much from training on 80%?).
- Then the ranker trained on held-out + Val + the training-question lists, against EXP-062/091, on held-out and Val.

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[H-017-translate-then-retrieve-luganda]]
- [[00_INDEX]]
