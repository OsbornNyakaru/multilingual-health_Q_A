---
type: fact
id: F-001
created: 2026-09-24
status: confirmed
links: ["[[F-002-data-shape]]", "[[F-004-rules]]", "[[D-001-metric-weights-and-tokenizer]]", "[[00_INDEX]]"]
---
# F-001 Competition metric and submission format

## Leaderboard metric
Weighted mean of three per-row scores, averaged over the test set:

| Component | Weight | Notes |
|-----------|--------|-------|
| ROUGE-1 F1 | **0.37** | `rouge-score`, whitespace tokenizer, no stemming (see below) |
| ROUGE-L F1 | **0.37** | same tokenizer |
| LLM-as-a-Judge | **0.26** | 1–5 score normalized to [0,1] |

- 74% of the score is lexical overlap (ROUGE).
- **AfroLM BertScore F1** is a host-side *secondary* check on top solutions. It is **not** on the leaderboard (weight 0). Don't trade meaning for overlap, but don't select models by it either (see `autoresearch_nlp/LESSONS.md` #1 and #5).
- Resolved discrepancy: `configs/base.yaml` still carries the pre-drop guess 0.25 R1 / 0.25 RL / 0.30 AfroLM-BS / 0.20 judge. That was wrong. The actual leaderboard weights are 0.37/0.37/0.26 (orchestrator context, `autoresearch_nlp/prepare.py`, `autoresearch_nlp/LESSONS.md`). `experiments/LOG.md` column definitions still quote the old weights — ignore them.

## ROUGE tokenization (matters for Amharic / Ge'ez)
- The official starter notebook scores ROUGE with a `WhitespaceTokenizer` and `use_stemmer=False`. The `rouge-score` default tokenizer lowercases and regex-splits on `[a-z0-9]`, which deletes Ge'ez script entirely (Amharic would score ~0).
- FACT: the starter does this. INFERENCE: the host grader does too (likely, unconfirmed). Source: `autoresearch_nlp/COMPETITION_INTEL.md` §2.
- Local harness `autoresearch_nlp/prepare.py` uses whitespace tokenization. Detail on the local replica: [[metric-replica]] (2026-09-24). It verifies the starter code and shows that EXP-004 was actually scored with the default tokenizer.
- Locally the judge term is 0 unless a stand-in judge is run; ROUGE-only combined = 0.37·R1 + 0.37·RL.

## Submission format
Exactly 4 columns, mirroring `SampleSubmission.csv`:

```
ID,TargetRLF1,TargetR1F1,TargetLLM
ID_TS_Aka_Gha_A3B1799D,"<answer>","<answer>","<answer>"
```

- All three target columns hold the **same generated answer text** per row (one answer scored three ways).
- 2,618 rows; IDs must equal Test IDs; no duplicates, no empties; CSV not `.xlsx`; no extra 5th column.
- Use `autoresearch_nlp/prepare.py` `build_submission()` / `validate_submission()`; don't hand-roll.

Sources migrated: `vault/Scoring Metric.md`, `vault/Submission Format.md`, `autoresearch_nlp/COMPETITION_INTEL.md`, `autoresearch_nlp/LESSONS.md`, `prompts/00_orchestrator.md`.

## Links
- [[00_INDEX]]
- [[F-002-data-shape]]
- [[F-004-rules]]
- [[D-001-metric-weights-and-tokenizer]]
- Legacy: [[Scoring Metric]], [[Submission Format]]
