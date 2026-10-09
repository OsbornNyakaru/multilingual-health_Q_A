---
type: hypothesis
id: H-016
created: 2026-10-09
status: testing
links: ["[[H-015-llm-chooser-over-top5]]", "[[FND-003-research-synthesis-2026-10-09]]", "[[00_INDEX]]"]
---
# H-016 fine-tuned listwise chooser over the top-5

**Statement:** A zero-shot LLM picks the most helpful-looking answer, not the dataset's stock answer ([[H-015-llm-chooser-over-top5]]). A chooser trained on this dataset's own labels learns which stock answer goes with which kind of question, and beats the ranker's first pick on the closed subsets. This is the Eedi 1st-place recipe: a listwise QLoRA over the retriever's top-k, letter logits, permutation averaging.

## Design
- Training lists come from the pool alone. Each pool row is a query, and its options are the top-5 distinct answers of the other pool rows, found with the base BGE-M3 + bge-reranker-v2-m3. That retriever never trained on these rows, so the lists are honest.
- Target: the option with the highest answer overlap with the row's gold. Lists where no option reaches 0.5 are skipped.
- Random cyclic shift per list, so the model learns content, not position. Inference averages all shifts.
- Each option shows its answer and the Train question it was matched from.
- Inference reads EXP-055's top-5 lists, as in EXP-085.

## Tests
- EXP-087: Qwen2.5-7B-Instruct LoRA, 1 epoch, held-out + Val, five closed subsets.
- If it helps but does not win outright, its per-option scores become ranker inputs. They are compared only within a question, so they don't depend on pool size.

## Links
- [[H-015-llm-chooser-over-top5]]
- [[FND-003-research-synthesis-2026-10-09]]
- [[00_INDEX]]
