---
type: fact
id: F-002
created: 2026-09-24
status: confirmed
links: ["[[F-001-competition-metric]]", "[[FND-001-retrieval-strength-by-subset]]", "[[00_INDEX]]"]
---
# F-002 Data shape

Zindi **Multilingual Health Question Answering in Low-Resource African Languages Challenge**. Maternal, sexual and reproductive health questions; answer in the question's language. Languages: Akan (Twi), Amharic, Luganda, Swahili, English.

Canonical copies in `data/raw/` (gitignored). Counts re-verified 2026-09-24 by reading the CSVs.

| File | Rows | Columns |
|------|------|---------|
| Train.csv | 29,815 | `ID, input, output, subset` |
| Val.csv | 6,686 | `ID, input, output, subset` |
| Test.csv | 2,618 | `ID, input, subset` (no `output`) |
| SampleSubmission.csv | 2,618 | `ID, TargetRLF1, TargetR1F1, TargetLLM` |

- `input` = question, `output` = reference answer, `subset` = `<Lang>_<Country>` locale code.
- IDs look like `ID_TR_Aka_Gha_A3B1799D` (TR = train, TS = test); the third token is the language.

## Subsets (language / country)

| subset | language | country | Train | Val | Test |
|--------|----------|---------|------:|----:|-----:|
| Eng_Uga | English | Uganda | 7,624 | 1,688 | 744 |
| Aka_Gha | Akan (Twi) | Ghana | 4,455 | 1,114 | 492 |
| Eng_Gha | English | Ghana | 4,443 | 1,104 | 491 |
| Eng_Eth | English | Ethiopia | 3,915 | 564 | 60 |
| Lug_Uga | Luganda | Uganda | 3,383 | 846 | 374 |
| Eng_Ken | English | Kenya | 2,080 | 390 | 167 |
| Swa_Ken | Swahili | Kenya | 2,070 | 518 | 229 |
| Amh_Eth | Amharic | Ethiopia | 1,845 | 462 | 61 |

- **Discrepancy:** the orchestrator prompt says "9 language/country subsets". The local CSVs contain **8** distinct `subset` values in all of Train, Val and Test (checked 2026-09-24). Treat 8 as ground truth unless a 9th appears in a new data drop.
- English is ~61% of Train and ~56% of Test (Eng_Uga alone ~28% of Test). More than half the score comes from English.
- Amharic is the smallest subset (61 test rows) and the hardest for ROUGE.
- `configs/base.yaml` still expects pre-drop boilerplate columns (`Question`, `Language`, `Response`, `TargetBert`). The real schema is above; `autoresearch_nlp/prepare.py` is already correct.
- Local held-out: `autoresearch_nlp/prepare.py make_splits()` = 7% of Train stratified by subset, seed 1234 → `work_train.csv` (27,727) + `held_out.csv` (2,088). See [[D-005-held-out-protocol]].

Sources migrated: `vault/Data Schema.md`, `vault/Competition Overview.md`, `data/processed/data_audit_report.md`.

## Links
- [[00_INDEX]]
- [[F-001-competition-metric]]
- [[D-005-held-out-protocol]]
- [[FND-001-retrieval-strength-by-subset]]
- Legacy: [[Data Schema]], [[Competition Overview]]
