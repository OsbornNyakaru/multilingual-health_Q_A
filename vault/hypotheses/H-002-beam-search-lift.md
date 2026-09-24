---
type: hypothesis
id: H-002
created: 2026-09-24
status: open
links: ["[[D-005-held-out-protocol]]", "[[00_INDEX]]"]
---
# H-002 beam search lift

**Statement:** Beam search (num_beams=5) lifts combined over greedy by ≥ 0.01 at no additional training cost.

## Notes
Planned ablation 'Decoding strategy' (greedy vs beam-5 vs beam-8+reranker) from `experiments/ABLATIONS.md`. A reranker must justify ≥ 0.005 combined lift. `autoresearch_nlp/train.py` defaults to beams=4, no_repeat_ngram=3.

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[D-005-held-out-protocol]]
- [[00_INDEX]]
