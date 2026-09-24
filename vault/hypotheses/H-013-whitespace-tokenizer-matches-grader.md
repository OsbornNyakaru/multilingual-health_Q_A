---
type: hypothesis
id: H-013
created: 2026-09-24
status: open
links: ["[[D-001-metric-weights-and-tokenizer]]", "[[F-001-competition-metric]]", "[[00_INDEX]]"]
---
# H-013 whitespace tokenizer matches grader

**Statement:** The host grader scores ROUGE with a whitespace tokenizer (as the starter notebook does), so whitespace-token local ROUGE tracks the leaderboard — including Amharic.

## Notes
From `vault/Open Questions.md` and `autoresearch_nlp/COMPETITION_INTEL.md` §2 (inference, unconfirmed). Test: submit one run and compare per-component LB scores with local; Subagent C's metric replica will quantify default vs whitespace tokenizer gaps. See [[D-001-metric-weights-and-tokenizer]].

**2026-09-24 update ([[metric-replica]]):** The starter notebook (cell 13) is verified to use `str(text).strip().split()` with `use_stemmer=False`, which is case-sensitive and leaves punctuation attached. The replica's `whitespace` tokenizer copies it exactly. rouge-score's default tokenizer turns Ge'ez into `[]`, so every Amharic pair scores 0. The tokenizer gap is measured: BGE-M3 on EXP-004's rows scores ROUGE-only 0.3700 (whitespace) vs 0.3892 (default + stemmer), and Amh_Eth 0.115 vs 0.012. Belief: whitespace, moderate confidence. Status stays **open** until one LB submission's per-component scores are compared with local `--compare-tokenizers` output.

Source: new, from `vault/Open Questions.md` and `autoresearch_nlp/COMPETITION_INTEL.md`.

## Links
- [[metric-replica]]
- [[D-001-metric-weights-and-tokenizer]]
- [[F-001-competition-metric]]
- [[00_INDEX]]
