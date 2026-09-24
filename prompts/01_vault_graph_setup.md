# 01 — Obsidian Vault / Graph Engineering Setup

Run standalone, or as Subagent A from `00_orchestrator.md`. Idempotent —
safe to re-run; it should skip files that already exist and only add
what's missing.

## Goal

Create `vault/` at the repo root as an Obsidian-openable vault that
becomes the durable memory for this project. Every future coding agent
session — yours or your teammate's — opens `vault/00_INDEX.md` first and
gets full context without re-reading chat history.

## Structure to create

```
vault/
  00_INDEX.md              # entry point, links to everything below
  facts/                   # stable ground truth (data, metric, rules)
  hypotheses/              # one note per idea to test, status tracked
  experiments/             # one note per run, auto-linked to its hypothesis
  decisions/               # ADR-style: what we chose and why, with alternatives
  findings/                # conclusions distilled from multiple experiments
  .obsidian/                # leave empty or minimal; don't fight Obsidian's own config
```

## Node conventions (apply to every note you create, now and later)

Each note starts with YAML frontmatter:

```yaml
---
type: hypothesis   # or: fact | experiment | decision | finding
id: H-003
created: <ISO date>
status: open        # open | testing | confirmed | rejected | superseded
links: ["[[F-002 amharic-tokenizer]]", "[[EXP-005]]"]
---
```

Body is free Markdown, but always ends with a `## Links` section listing
outgoing `[[wikilinks]]` — this is what makes Obsidian's graph view useful.
IDs are short and stable (`H-001`, `EXP-001`, `F-001`, `D-001`) so links
don't break when titles change.

## Migration tasks

1. **Read `experiments/LOG.md`, `experiments/HYPOTHESES.md`,
   `experiments/ABLATIONS.md`** (the repo already has these — don't
   duplicate their purpose, convert them). For each existing entry:
   - A hypothesis → `vault/hypotheses/H-XXX-<slug>.md`
   - A logged run → `vault/experiments/EXP-XXX-<slug>.md`
   - An ablation result → fold into the relevant experiment note or a
     `vault/findings/` note if it spans several runs.
   Keep the original files in place (don't delete them yet) but add a
   one-line pointer at the top of each: `> Migrated to vault/, see
   [[00_INDEX]]`.

2. **Seed `vault/facts/`** with these notes, sourced from the competition
   page and the 11th-place writeup (content given to you separately —
   paste it in or reference the files you already have locally):
   - `F-001-competition-metric.md` — ROUGE-1/L weights, LLM-judge weight,
     4-column submission format, AfroLM BertScore as a secondary check.
   - `F-002-data-shape.md` — row counts, columns, 9 subsets, which subsets
     are which language/country.
   - `F-003-reference-approach.md` — the koleshjr 11th-place pipeline,
     verbatim numbers (RougeL 0.4823 / Rouge1 0.5548 / Judge 0.7379 for
     BGE-M3 retrieval-only), their LoRA config, and the sweetlhare router
     idea (closed-pool retrieval vs generative fine-tune per subset).
   - `F-004-rules.md` — open-source only, no paid APIs, seed discipline,
     unlimited practice submissions since the challenge is closed.

3. **Write `vault/00_INDEX.md`** with:
   - One-paragraph project summary and current goal (beat 11th place
     locally, on our own held-out eval).
   - A table of contents linking every note in `facts/`.
   - An "Open hypotheses" section auto-listing every `hypotheses/*.md`
     with `status: open` or `status: testing`.
   - A "Current best" section: which experiment ID currently has the best
     weighted local score, updated by hand after each experiment closes.
   - House rules block (copy verbatim):
     > Any agent working in this repo: read this file and every note it
     > links to before writing code. When you finish a unit of work, add
     > or update a vault note and a link back here. Never leave a result
     > only in chat or terminal output — if it's not in the vault, it
     > didn't happen.

## Verification

After creating the vault, open it (or list files) and confirm:
- `00_INDEX.md` has no dead links (every `[[link]]` resolves to a file).
- Every file in `hypotheses/`, `experiments/`, `decisions/`, `findings/`
  has valid frontmatter and is linked from at least one other note.

Report the final file tree of `vault/` when done.
