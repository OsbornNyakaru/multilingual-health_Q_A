---
tags: [data, submission]
---
> Migrated to vault/ graph, see [[00_INDEX]]

# Submission Format

Mirror `SampleSubmission.csv` exactly:

```
ID,TargetRLF1,TargetR1F1,TargetLLM
ID_TS_Aka_Gha_A3B1799D,"<answer>","<answer>","<answer>"
```

- Three target columns, **identical text per row**. One answer is scored three ways (see [[Scoring Metric]]).
- 2,618 rows, IDs must equal the Test IDs, no duplicates, no empties.
- `autoresearch_nlp/prepare.py` has `build_submission()` and `validate_submission()` that enforce this. Reuse them; do not hand-roll.
- Previous real submissions are in `submissions/` (`final_checkpoint.csv`, `submission_ready.csv`, an e5-small retrieval baseline with JSON sidecar).

Related: [[Data Schema]], [[Autoresearch Harness]].
