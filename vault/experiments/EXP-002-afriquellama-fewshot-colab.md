---
type: experiment
id: EXP-002
created: 2026-09-24
run_date: 2026-05-12
status: confirmed
hypothesis: "[[H-010-per-language-prompts]]"
links: ["[[H-010-per-language-prompts]]", "[[D-004-base-model-for-molab]]", "[[00_INDEX]]"]
---
# EXP-002 AfriqueLlama-8B few-shot inference (Colab T4)

- **Model:** `McGill-NLP/AfriqueLlama-8B`, 4-bit NF4, Llama-3 chat template.
- **Setup:** 2 few-shot examples per language (answers 50–300 chars, `random_state=42`), per-language doctor-persona system prompts (tests the spirit of [[H-010-per-language-prompts]]), refusal detection, greedy, `max_new_tokens 80`, batch 8, prompts sorted by length. Notebook: `resources/multilingual_qa.ipynb` (see legacy [[Latest Colab Notebook]]); related May notebooks `notebooks/exp002_*`.
- **Speed:** ~130 s/batch before prompt trimming, ~8–12 s/batch after.
- **Output:** `submissions/final_checkpoint.csv` (2026-05-12). Related May files: `submissions/submission_ready.csv`, `submissions/submission_20260512_0241.xlsx` (xlsx is not a valid submission format).
- **Local score:** **none** — never scored on held-out. **Public LB:** not recorded anywhere in the repo.
- **Known bugs of that era** (`autoresearch_nlp/LESSONS.md` #6): mixed-language batches applied length caps to wrong rows; an English fallback string was inserted into all languages; stale 5th column.

## Links
- [[00_INDEX]]
- [[H-010-per-language-prompts]]
- [[D-004-base-model-for-molab]]
- [[Latest Colab Notebook]]
