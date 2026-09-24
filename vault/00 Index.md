---
tags: [moc, afro-health-qa]
created: 2026-09-18
---
> **Superseded entry point.** Start at [[00_INDEX]] (facts / hypotheses / experiments / decisions / findings graph). This page is kept as the legacy map of the pre-graph notes.

# Afro Health QA — Knowledge Graph (Map of Content)

Open this folder (`vault/`) as an Obsidian vault, or open the whole repo as a vault. Every note links to its neighbours with `[[wikilinks]]` so the graph view shows the structure. Start here.

## Competition
- [[Competition Overview]] — what Zindi asked for, dates, languages
- [[Data Schema]] — the real columns, row counts, subsets
- [[Submission Format]] — the exact CSV Zindi accepts
- [[Scoring Metric]] — ROUGE-1 / ROUGE-L / LLM judge and the local harness weights

## Code and experiments
- [[Repo Layout]] — what lives where in the workspace
- [[Autoresearch Harness]] — `autoresearch_nlp/` frozen prepare.py + editable train.py loop
- [[Model Decisions]] — which base models were tried, licences, VRAM reality
- [[Latest Colab Notebook]] — the AfriqueLlama-8B T4 inference notebook from `resources/`
- [[Experiment History]] — pointer to `experiments/LOG.md` and what was actually submitted

## Platform move: Molab
- [[Molab Platform]] — free GPU notebook host, limits, storage rules
- [[Marimo MCP and Pairing]] — how Claude Code drives a marimo / molab notebook
- [[Molab Notebook Plan]] — the notebook's safe operating sequence and handoff status
- [[File Organisation]] — where data lives locally vs on molab

## Operations
- [[Repo Recovery 2026-09-18]] — the corrupted git repo and how it was repaired
- [[Session Log]] — running diary, newest entry at the top
- [[Open Questions]] — decisions still pending
