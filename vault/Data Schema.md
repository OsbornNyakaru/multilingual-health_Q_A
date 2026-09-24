---
tags: [data]
---
> Migrated to vault/ graph, see [[00_INDEX]]

# Data Schema

The canonical copies now live in `data/raw/` (see [[File Organisation]]). All four CSVs from the Zindi zip in `resources/` are byte-identical to the older copies in `data/multilingual_data/` (verified by sha1 on 2026-09-18).

| File | Rows (excl. header) | Columns |
|------|------|---------|
| Train.csv | 29,815 | `ID, input, output, subset` |
| Val.csv | 6,686 | `ID, input, output, subset` |
| Test.csv | 2,618 | `ID, input, subset` |
| SampleSubmission.csv | 2,618 | `ID, TargetRLF1, TargetR1F1, TargetLLM` |

- `input` = question, `output` = reference answer, `subset` = locale code.
- IDs look like `ID_TR_Aka_Gha_A3B1799D`; the third token is the language.
- **Gotcha:** `configs/base.yaml` still expects the pre-drop boilerplate columns (`Question`, `Language`, `Response`, `TargetBert`). The real schema is the one above. The [[Autoresearch Harness]] `prepare.py` is already correct.

## Subsets in Train (8 locales, 5 languages)

| subset | rows | language |
|--------|------|----------|
| Eng_Uga | 7,624 | eng |
| Aka_Gha | 4,455 | aka |
| Eng_Gha | 4,443 | eng |
| Eng_Eth | 3,915 | eng |
| Lug_Uga | 3,383 | lug |
| Eng_Ken | 2,080 | eng |
| Swa_Ken | 2,070 | swa |
| Amh_Eth | 1,845 | amh |

English is ~61% of Train. Answer in the same language as the question, per subset.

Related: [[Submission Format]], [[Scoring Metric]], [[Molab Notebook Plan]].
