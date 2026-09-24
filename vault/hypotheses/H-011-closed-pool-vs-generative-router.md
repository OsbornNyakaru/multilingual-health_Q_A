---
type: hypothesis
id: H-011
created: 2026-09-24
status: open
links: ["[[F-003-reference-approach]]", "[[FND-001-retrieval-strength-by-subset]]", "[[D-003-retrieval-first-baseline]]", "[[00_INDEX]]"]
---
# H-011 closed pool vs generative router

**Statement:** A per-subset (or per-row similarity-threshold) router — retrieval for closed-pool/paraphrase-heavy subsets, fine-tuned generation for the rest — beats both pure retrieval and pure generation on held-out combined score.

## Notes
From sweetlhare via [[F-003-reference-approach]] and our own strategy in `autoresearch_nlp/COMPETITION_INTEL.md` §5. Evidence for the split: [[FND-001-retrieval-strength-by-subset]]. `docs/ROADMAP_TOP5.md` estimates held-out R1 0.560 → 0.643 by generating only for weak subsets — that number is a **projection, not a measurement**. Select threshold by ROUGE on held-out, never by AfroLM. Decision context: [[D-003-retrieval-first-baseline]].

Source: new, from the reference approach ([[F-003-reference-approach]]) and `autoresearch_nlp/COMPETITION_INTEL.md`.

## Links
- [[F-003-reference-approach]]
- [[FND-001-retrieval-strength-by-subset]]
- [[D-003-retrieval-first-baseline]]
- [[00_INDEX]]
