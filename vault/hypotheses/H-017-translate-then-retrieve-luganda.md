---
type: hypothesis
id: H-017
created: 2026-10-09
status: testing
links: ["[[FND-004-luganda-answers-translate-english]]", "[[H-016-fine-tuned-listwise-chooser]]", "[[00_INDEX]]"]
---
# H-017 translate-then-retrieve for Lug_Uga

**Statement:** Lug_Uga loses most at finding the answer: gold is in the top-5 for 76% of questions, although 88% exist in Train. English retrieval is much stronger (gold in the top-5 for 87% on Eng_Uga), and Lug_Uga's stock answers are largely translations of Eng_Uga's ([[FND-004-luganda-answers-translate-english]]). Machine-translating the Luganda questions (pool and queries) to English and retrieving on the English text should therefore raise recall, either alone or blended with the Luganda view.

## Tests
- EXP-088: NLLB-200 distilled 1.3B, base BGE-M3, Lug_Uga, held-out + Val.
  - It records each view's top-50 (original, translated, 50/50 blend), so `exp.py pull` prints recall@1/5/20/50 per view.
  - The baseline is the untranslated view in the same run. The trained EXP-055 stack (cand_ids) is the reference to beat.
- If the translated view adds recall, wire it into the trained stack: as a second retrieval view feeding the reranker, and as a ranker input.

## Links
- [[FND-004-luganda-answers-translate-english]]
- [[H-016-fine-tuned-listwise-chooser]]
- [[00_INDEX]]
