---
type: hypothesis
id: H-015
created: 2026-10-09
status: rejected
links: ["[[FND-003-research-synthesis-2026-10-09]]", "[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# H-015 zero-shot LLM chooser over the ranker's top-5

**Statement:** On the closed subsets the gold answer is in the ranker's top-5 distinct answers far more often than it is picked (Eng_Uga 0.87 vs 0.78, Lug_Uga 0.76 vs 0.61). An instruction-tuned 31B model shown the question and the top-5 answers (letters, 5 cyclic shifts to cancel position bias) picks the gold one more often than the ranker.

## Test
EXP-085: gemma-4-31B-it, `llm_choose`, candidates from EXP-055 (BGE-M3 + reranker), k=5, held-out + Val, five closed subsets.

## Result: refuted, by a wide margin
ROUGE part (0.74·mean(R1, RL)) per subset, Val; held-out agrees:

| subset | chooser | ranker (EXP-055) | current best |
|---|--:|--:|--:|
| Eng_Uga | 0.535 | 0.623 | 0.627 |
| Lug_Uga | 0.454 | 0.488 | 0.492 |
| Swa_Ken | 0.509 | 0.610 | 0.623 |
| Eng_Ken | 0.500 | 0.589 | 0.615 |
| Eng_Eth | 0.331 | 0.490 | 0.478 |

- Not a bug: on the 586 Val rows that kept their scores (the rest lost meta at the session restart), shift agreement is high and sane. With the gold in the options (85% of rows), the chooser is right 68% of the time and option A (the ranker's top) 89%.
- Confident overrides are the worst: where the chooser is ≥ 0.99 sure and disagrees with the ranker, it is right 22% of the time and the ranker 72%.
- Reading: the dataset's "right" answer is a canned label, not the most helpful-looking text. A general model judges helpfulness, which is the wrong target. This matches 1st place, whose LLM chooser lost on the leaderboard.

## Consequence
- Drop zero-shot choosing as an override, and drop it as an LTR feature (pipeline item 2): the confident-disagreement signal is anti-correlated with gold.
- What remains for selection is a chooser *trained on this dataset's labels* (pipeline item 3, Eedi-style listwise QLoRA over out-of-fold top-5 lists).

## Links
- [[FND-003-research-synthesis-2026-10-09]]
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
