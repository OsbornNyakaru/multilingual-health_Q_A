---
type: hypothesis
id: H-013
created: 2026-09-24
status: testing
links: ["[[D-001-metric-weights-and-tokenizer]]", "[[F-001-competition-metric]]", "[[00_INDEX]]"]
---
# H-013 whitespace tokenizer matches grader

**Statement:** The host grader scores ROUGE with a whitespace tokenizer (as the starter notebook does), so whitespace-token local ROUGE tracks the leaderboard — including Amharic.

## Notes
From `vault/Open Questions.md` and `autoresearch_nlp/COMPETITION_INTEL.md` §2 (inference, unconfirmed). Test: submit one run and compare per-component LB scores with local; Subagent C's metric replica will quantify default vs whitespace tokenizer gaps. See [[D-001-metric-weights-and-tokenizer]].

**2026-09-24 update ([[metric-replica]]):** The starter notebook (cell 13) is verified to use `str(text).strip().split()` with `use_stemmer=False`, which is case-sensitive and leaves punctuation attached. The replica's `whitespace` tokenizer copies it exactly. rouge-score's default tokenizer turns Ge'ez into `[]`, so every Amharic pair scores 0. The tokenizer gap is measured: BGE-M3 on EXP-004's rows scores ROUGE-only 0.3700 (whitespace) vs 0.3892 (default + stemmer), and Amh_Eth 0.115 vs 0.012. Belief: whitespace, moderate confidence. Status stays **open** until one LB submission's per-component scores are compared with local `--compare-tokenizers` output.

Source: new, from `vault/Open Questions.md` and `autoresearch_nlp/COMPETITION_INTEL.md`.

## 2026-10-06: first leaderboard evidence (points against the hypothesis)
- First Zindi submission (per-subset retrieval composite, `experiments/RESULTS.md`): public 0.59202 = 0.37·R1 0.5616 + 0.37·RL 0.5096 + 0.26·judge 0.7526, so the weights are confirmed. LB rouge-only = 0.3963.
- The same composite scored on Val (test-mix weighted): whitespace (ours) 0.3747; whitespace lowercased 0.3793; **rouge-score default tokenizer 0.3952**; default + stemmer 0.3996. The default tokenizer matches the LB within 0.001.
- Confound: test predictions retrieve from Train + Val, Val predictions only from Train, which by itself lifts test scores. So this is suggestive, not proof.
- Decisive probe queued with the user: resubmit the same file with every answer UPPERCASED (`experiments/runs/submissions/probe_uppercase_20261006.csv`). If the LB's ROUGE-1/ROUGE-L stay at 0.5616/0.5096, the grader lowercases (default tokenizer: lowercase, keep only [a-z0-9], so Ge'ez scores ~0). If they collapse, the grader is case-sensitive whitespace, as assumed.
- If the default tokenizer is confirmed: the frozen harness must switch tokenizer (a new pinned prepare.py hash in the runner), all runs get re-scored, and Amharic ROUGE becomes worthless (only the judge counts there).

## Links
- [[metric-replica]]
- [[D-001-metric-weights-and-tokenizer]]
- [[F-001-competition-metric]]
- [[00_INDEX]]
