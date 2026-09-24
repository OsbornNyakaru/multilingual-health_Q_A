---
type: hypothesis
id: H-010
created: 2026-09-24
status: open
links: ["[[F-001-competition-metric]]", "[[00_INDEX]]"]
---
# H-010 per language prompts

**Statement:** Per-language prompt templates (native-language labels, in-language instruction) beat a single English `Question:/Answer:` prompt.

## Notes
Planned ablation 'Prompt template'. The starter notebook's `build_prompt()` omits the language line entirely, so even adding the language name is expected to help (`autoresearch_nlp/COMPETITION_INTEL.md` §1). Per-language prompts are already wired in `autoresearch_nlp/train.py`.

Migrated from `experiments/HYPOTHESES.md` / `experiments/ABLATIONS.md`.

## Links
- [[F-001-competition-metric]]
- [[00_INDEX]]
