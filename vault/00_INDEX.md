---
type: index
id: INDEX
created: 2026-09-24
status: open
links: []
---
# 00_INDEX — Afro Health QA vault (entry point)

> Any agent working in this repo: read this file and every note it
> links to before writing code. When you finish a unit of work, add
> or update a vault note and a link back here. Never leave a result
> only in chat or terminal output — if it's not in the vault, it
> didn't happen.

## Project summary and current goal
Post-close practice run of Zindi's **Multilingual Health Question Answering in Low-Resource African Languages Challenge**: answer maternal, sexual and reproductive health questions in Akan, Amharic, Luganda, Swahili and English (8 locale subsets). The score is 0.37 ROUGE-1 F1 + 0.37 ROUGE-L F1 + 0.26 LLM-judge. Submissions are unlimited, and only open-weight models with no paid APIs are allowed. **Current goal: beat the 11th-place reference approach (BGE-M3 retrieval → RAG-enriched LoRA fine-tune, plus the untried closed-pool vs generative router) on our own held-out eval, measured per subset.** Code is written locally; GPU work runs on molab. Every experiment follows the loop: hypothesis → single change → run → score locally per language → log here → keep or revert → submit only on a real margin (≥ +0.003, [[D-005-held-out-protocol]]).

## Facts (stable ground truth)
<!-- Append new fact notes to this list: one bullet per note, a wikilink to the note basename, then a one-line summary. -->
- [[F-001-competition-metric]] — weights 0.37/0.37/0.26, whitespace ROUGE, 4-column submission, AfroLM BertScore secondary only
- [[F-002-data-shape]] — row counts, columns, the 8 subsets (orchestrator said 9; data has 8)
- [[F-003-reference-approach]] — koleshjr 11th place: BGE-M3 retrieval R1 0.5548 / RL 0.4823 / Judge 0.7379, Sunflower-32B RAG LoRA, sweetlhare router
- [[F-004-rules]] — open-source only, no paid APIs, seeds, unlimited practice submissions
- [[repo-state]] — F-005 repo audit: `src/` package broken (missing `models` subpackage, 0.25/0.25/0.30/0.20 weights); working code = `autoresearch_nlp/`, molab notebook, retrieval scripts; RAG builder partial, LoRA partial, vLLM + router missing; no tests; EXP-004 was scored with the default (non-whitespace) tokenizer
- [[metric-replica]] — F-006 per-subset scorer (`evaluation/scorer.py`, 0.37/0.37/0.26, injectable judge); starter = whitespace ROUGE; EXP-004 re-scored whitespace = 0.3700 (not 0.3892), Amh_Eth 0.115 (0.012 was a tokenizer artifact)

## Open hypotheses
Hand-maintained list of every `hypotheses/*.md` with `status: open` or `status: testing`. If the Dataview plugin is installed, the query below lists them automatically.

```dataview
TABLE id, status FROM "hypotheses" WHERE status = "open" OR status = "testing" SORT id ASC
```

- [[H-002-beam-search-lift]] — open
- [[H-003-per-language-length-bounds]] — testing (retrieval-side result negative, generation untested)
- [[H-004-qlora-comp-only-beats-zero-shot]] — open
- [[H-005-medmcqa-translated-augmentation]] — open
- [[H-006-who-synthetic-qa]] — open
- [[H-008-ulizallama-swahili]] — open
- [[H-010-per-language-prompts]] — open
- [[H-011-closed-pool-vs-generative-router]] — open, **highest priority**
- [[H-012-rag-enriched-finetune]] — open, **highest priority**
- [[H-013-whitespace-tokenizer-matches-grader]] — open

Closed or superseded: [[H-001-zero-shot-aya-floor]], [[H-007-afrolm-reranker]], [[H-009-gemma-aya-adapter-ensemble]].

## Current best
Updated by hand after each experiment closes.

| | |
|---|---|
| Experiment | [[EXP-004-bge-m3-retrieval-heldout]] (BGE-M3 within-subset retrieval, no generation) |
| Held-out | 1,491 rows (older split, **not** the current 2,088-row harness split) |
| ROUGE-1 F1 | 0.5237 (whitespace tokenizer; was 0.5604 under default+stemmer) |
| ROUGE-L F1 | 0.4764 (was 0.4914) |
| Judge | not measured |
| Weighted (ROUGE-only, 0.37·R1 + 0.37·RL) | **0.3700** (2026-09-24 re-score with whitespace tokenizer, see [[metric-replica]]; old 0.3892 used the wrong tokenizer) |
| Source | `docs/competition_report/data/bge_retrieval_heldout.csv` re-scored via `afro_health_qa.evaluation.scorer` |
| Public LB | never recorded |

On the current 2,088-row split, measured numbers are the crude token-overlap retrieval, ROUGE-only 0.279 ([[EXP-003-token-overlap-retrieval]]; scorer at the time unverified), and a CPU TF-IDF within-subset retrieval sanity check, ROUGE-only 0.3068 with the whitespace scorer ([[metric-replica]]). First task for the loop: re-measure BGE-M3 retrieval on the current split with the metric replica, including a judge stand-in.

## Experiments
- [[EXP-000-scaffold]] — repo bootstrap (the only row in `experiments/LOG.md`)
- [[EXP-001-aya-expanse-oom]] — Aya-Expanse-8B on T4, OOM, no score
- [[EXP-002-afriquellama-fewshot-colab]] — AfriqueLlama-8B few-shot, produced `final_checkpoint.csv`, never scored
- [[EXP-003-token-overlap-retrieval]] — crude retrieval, held-out 2,088, ROUGE-only 0.279
- [[EXP-004-bge-m3-retrieval-heldout]] — BGE-M3 retrieval, held-out 1,491, ROUGE-only 0.3892 (current best)
- [[EXP-005-length-truncation-ablation]] — truncating retrieval outputs, reverted
- [[EXP-006-e5-small-retrieval-test-submission]] — retrieval test submission file, no LB score recorded

## Decisions
- [[D-001-metric-weights-and-tokenizer]] — optimise the real weights; whitespace ROUGE; AfroLM weight 0
- [[D-002-reject-aya-expanse]] — licence + language coverage
- [[D-003-retrieval-first-baseline]] — retrieval is the per-subset floor; aim for a router
- [[D-004-base-model-for-molab]] — OPEN: which licence-clean base model
- [[D-005-held-out-protocol]] — 7% stratified split, seed 1234, ≥ +0.003 keep threshold

## Findings
- [[FND-001-retrieval-strength-by-subset]] — closed-pool vs generative subsets
- [[FND-002-length-truncation-on-retrieval]] — don't truncate retrieved answers

## Legacy notes (pre-graph, 2026-09-18)
Notes marked "Migrated" have been folded into the graph above but are kept for their detail. The others are operational notes that haven't been migrated.
- [[00 Index]] — old map of content (now points here)
- Migrated: [[Scoring Metric]], [[Submission Format]], [[Competition Overview]], [[Data Schema]], [[Model Decisions]], [[Experiment History]], [[Open Questions]]
- Platform: [[Molab Platform]], [[Molab Notebook Plan]], [[Marimo MCP and Pairing]], [[File Organisation]]
- Code: [[Autoresearch Harness]], [[Repo Layout]], [[Latest Colab Notebook]]
- Ops: [[Repo Recovery 2026-09-18]], [[Session Log]]

Repo-side trackers `experiments/LOG.md`, `experiments/HYPOTHESES.md` and `experiments/ABLATIONS.md` are kept but superseded by this vault.

## Links
See all sections above.
