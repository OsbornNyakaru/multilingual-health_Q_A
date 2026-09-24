# 00 — Orchestrator Prompt

Paste this into your local coding agent (Claude Code) at the root of
`afro-health-qa`. Run it first, once. It sets up the operating model that
every later prompt (01–04) assumes: a graph-engineered memory in
`vault/` and a loop-engineered experiment cycle. It then dispatches
subagents for the rest of the setup.

---

## Context (do not skip — read before acting)

We are competing (post-close, practice mode — unlimited submissions, no
daily cap) in Zindi's **Multilingual Health Question Answering in
Low-Resource African Languages Challenge**.

- Languages: Akan, Amharic, Luganda, Swahili, English — 9 language/country
  subsets (e.g. `Aka_Gha`, `Eng_Uga`, `Swa_Ken`).
- Data: `Train.csv` (~29,815 rows), `Val.csv` (~6,686 rows), `Test.csv`
  (2,618 rows). Columns: `ID,input,output,subset` (Test has no `output`).
- Metric: weighted mean of ROUGE-1 F1 (0.37), ROUGE-L F1 (0.37), and an
  LLM-as-a-Judge 1–5 score normalized to [0,1] (0.26). Top solutions are
  also checked with AfroLM BertScore F1, so don't over-optimize lexical
  overlap at the expense of meaning.
- Submission file: exactly 4 columns — `ID,TargetRLF1,TargetR1F1,TargetLLM`
  — all three score columns hold the **same generated answer text**.
- Rules that still matter even though the challenge is closed and we're
  free-practicing: open-source/open-weight tools and pretrained models
  only, no paid APIs, always set the seed.

**The reference approach** (11th place, `koleshjr/multilingual_qa_training`)
found that a **BGE-M3 retrieval-only baseline** already scored
ROUGE-L F1 0.4823 / ROUGE-1 F1 0.5548 / LLM-Judge 0.7379 — meaning many
test questions are near-duplicates of train/val questions. Their winning
move was **RAG-enriched fine-tuning**: for each training row, retrieve k=3
similar labelled examples (excluding itself), put them in the prompt as
context, and fine-tune `Sunbird/Sunflower-32B` with Unsloth + LoRA
(rank 64, alpha 64, dropout 0.5, 3 epochs, lr 2e-4, batch 4, AdamW-8bit) to
still produce the *original* ground-truth answer — teaching the model to
use context rather than copy it. Validation retrieves from train only
(honesty); test retrieves from train+val (both have known answers). They
also noted a competitor (`sweetlhare`) went further: split subsets into
**closed-pool** (paraphrase-heavy → retrieve the answer directly) vs
**generative** (→ fine-tuned LLM), i.e. a router. That router is the
clearest untried lever to beat 11th place.

**Our repo already has scaffolding**: `configs/` (Hydra), `docs/`,
`experiments/{LOG.md,HYPOTHESES.md,ABLATIONS.md}`, `scripts/`,
`src/afro_health_qa/` (importable package), a zero-shot Aya-Expanse-8B
baseline, reproducibility infra (seeding, hashing, pytest), and submission
hedging via `select_final.py`.

**Where things run**: code is written *here, locally*, and pushed to
GitHub (`OsbornNyakaru/multilingual-health_Q_A`, `main`). All GPU work
runs on **molab** (marimo's cloud notebooks), which opens notebooks
straight from GitHub and re-syncs on push. molab compute: 4 CPU, 32 GiB
RAM, RTX Pro 6000 Blackwell (96 GB VRAM) — enough for 32B LoRA
fine-tuning. molab storage is not durable, so artifacts (RAG datasets,
LoRA adapters) go to private Hugging Face Hub repos. See
`03_marimo_molab_notebooks.md`. Don't try to run GPU jobs locally.

**Your job right now is NOT to fine-tune anything.** It is to build the
scaffolding described below, then hand off to prompts 01–04.

---

## Two working principles you must enforce from now on

### Graph engineering (`vault/`)
An Obsidian-compatible vault of Markdown notes, one node per fact,
hypothesis, experiment or decision, linked with `[[wikilinks]]`. This
replaces "context living only in chat history." Every subagent you spawn,
now or later, must:
1. Read `vault/00_INDEX.md` and any node it links to that's relevant,
   **before** doing any work.
2. Write or update the relevant node(s) **after** finishing, with a link
   back to `00_INDEX.md`.
No exceptions — an agent that skips this breaks continuity for the next
session.

### Loop engineering (the experiment cycle)
Every experiment, from here on, follows this exact loop and nothing else:
`Hypothesis → single change → run → score locally (per-language) → log to
vault → keep or revert → (only if it beats current best by a real margin)
submit`. This is defined fully in `04_experiment_loop.md` — set up the
mechanism now, use it starting in prompt 02.

---

## Tasks — do these in order, each as its own subagent

### Subagent A — Vault bootstrap
Run the instructions in `01_vault_graph_setup.md` verbatim. This migrates
`experiments/LOG.md`, `HYPOTHESES.md`, `ABLATIONS.md` into the vault graph
and creates the index and node templates.

### Subagent B — Repo audit note
Read `README.md`, `configs/`, `src/afro_health_qa/`, `scripts/`, and any
existing results/checkpoints. Write a single vault node
`vault/facts/repo-state.md` summarizing: what's implemented, what's a
stub, which config groups exist, what the current baseline score is (if
any run has happened), and what's missing versus the reference approach
(RAG context generation, LoRA fine-tune script, vLLM inference, per-subset
router). Link it from `00_INDEX.md`.

### Subagent C — Metric replica
Implement (or verify, if it already exists in `src/afro_health_qa/`) a
local scorer that reproduces the competition metric as closely as
possible: ROUGE-1 F1 and ROUGE-L F1 via the `rouge-score` library
(weights 0.37/0.37), plus a stand-in for the LLM-as-a-Judge component
(weight 0.26) using a local open-weight judge model with a 1–5 rubric
(factual accuracy, completeness, language appropriateness), normalized to
[0,1]. Report all three **and the weighted total**, broken down **by
`subset`**, not just overall — this is how we'll tell if Amharic is
dragging the score down. Write findings (including any ROUGE-tokenizer
concerns for non-Latin scripts, especially Amharic/Ge'ez) to
`vault/facts/metric-replica.md`.

Do not proceed to fine-tuning (prompt 02) until Subagents A–C are done and
their vault nodes exist. Report back with a one-paragraph summary and the
list of vault files created.
