# Agent Prompts — afro-health-qa

Drop this folder into the repo as `prompts/` and commit it. Feed the
prompts to your local coding agent (Claude Code) from the repo root, in
this order:

| # | file | when | what it produces |
|---|---|---|---|
| 00 | `00_orchestrator.md` | once, first | operating rules, subagents A–C: vault, repo audit, metric replica |
| 01 | `01_vault_graph_setup.md` | called by 00 (re-runnable) | `vault/` Obsidian graph, migrated experiment logs, `00_INDEX.md` |
| 02 | `02_reproduce_baseline.md` | once, after 00 | retrieval-only + RAG-FT baselines matching 11th place, first submission |
| 03 | `03_marimo_molab_notebooks.md` | after 00, alongside 02 | `notebooks/*.py` that open in molab from GitHub |
| 04 | `04_experiment_loop.md` | repeatedly | one improvement loop per run: hypotheses → EXP notes → new best |

How to feed them: `Read prompts/00_orchestrator.md and carry it out.`
Referencing the file keeps the agent reading the version in git, so an
edit to a prompt is picked up next time.

Every session after the first starts with:
`Read vault/00_INDEX.md and everything it links to that's relevant, then
read prompts/04_experiment_loop.md and run one loop.`

Open `vault/` as a vault in Obsidian (File → Open folder as vault) to see
the graph view. Pull before opening so you see your teammate's notes.
