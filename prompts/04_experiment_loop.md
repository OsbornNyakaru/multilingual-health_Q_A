# 04 — Experiment Loop (Loop Engineering + Divide-and-Conquer)

Reusable. Run it every time you want to push the score up, once `02` has
a baseline and `03` notebooks work. Each run of this prompt works through
**one or more hypotheses**, each tested with a single change. Read
`vault/00_INDEX.md`, all `hypotheses/` notes with `status: open|testing`,
and the "Current best" experiment note before doing anything.

## The loop (non-negotiable)

```
pick hypothesis → state expected effect → ONE change → run (molab)
  → score locally, per subset → evaluator check → log EXP note
  → keep (new best) or revert → update INDEX → next hypothesis
```

- **One change per experiment.** If two things change, the result tells
  you nothing.
- **Eval protocol is fixed.** Split `Val.csv` once, stratified by `subset`
  and with a fixed seed, into `val_tune` (50%) and `val_holdout` (50%).
  Tune thresholds, k and decoding on `val_tune`. Report and compare **on
  `val_holdout` only**. Save the split IDs to `data/splits/` and record
  its hash in `vault/facts/F-005-eval-protocol.md`. Never change it
  mid-project. If you have to, start a new baseline and say so in a
  decision note.
- **"Real improvement"** = weighted-score gain on `val_holdout` whose 95%
  paired-bootstrap interval (1,000 resamples over rows) excludes zero.
  Anything smaller counts as noise, gets logged as "no effect", and is
  reverted.
- **Submissions.** The challenge is closed and practice submissions are
  unlimited, but local `val_holdout` is still the primary signal. The
  public leaderboard scores only ~30% of test. Submit each new local best,
  log the Zindi score in the same EXP note, and track how well local and
  Zindi scores agree in `vault/findings/local-vs-lb-correlation.md`. If
  they diverge, fixing the local eval becomes the top hypothesis.

## Divide-and-conquer patterns to use

### Orchestrator–workers (how you run this prompt)
You are the orchestrator. Don't do the experiments yourself. For each
hypothesis, spawn a **worker subagent** with a self-contained brief
containing:
- the hypothesis note path
- the exact config delta
- which notebook to run and which cell to trigger
- where to write the EXP note

Workers that don't share state (e.g. building a reranker vs. writing a
length-analysis script) can run in parallel. Anything that needs the GPU
is serialized, because there's one molab GPU.

### Prompt chaining (inside each worker)
`implement change in src/ → unit test (pytest) → push → run on molab →
collect scores → write EXP note`. Each step's output is the next step's
input. If a step fails, the worker stops and writes the failure to the
EXP note (`status: failed`, reason) instead of guessing past it.

### Routing (in the model itself, and in how work is assigned)
- **Model-level router** (the main lever over 11th place, see H-001/H-002
  below): each test row is routed to *retrieve-and-copy* or
  *generate-with-RAG* based on subset and retrieval confidence.
- **Work-level routing**: data or retrieval hypotheses go to a retrieval
  worker, training hypotheses to a training worker, and decoding or
  post-processing hypotheses to an inference worker. Each worker type
  only touches its own module in `src/afro_health_qa/`.

### Evaluator–optimizer (the gate before "keep")
Before marking any experiment as the new best, spawn a **separate
evaluator subagent** that has *not* seen the worker's reasoning. Give it
only the prediction CSV, the `val_holdout` split, and the scoring code.
It independently:
- recomputes the scores
- checks for leakage: no val row retrieved into its own context, no test
  IDs in training data
- spot-reads 5 random predictions per subset for language correctness
  (answer in the same language as the question)
- confirms the bootstrap interval

Only an evaluator "PASS" allows updating "Current best" in the index.

## Hypothesis queue (seed these as H-notes if they don't already exist)

Ordered by expected value per GPU-hour. Re-prioritize after every result.

| id | hypothesis | single change | why we think so |
|---|---|---|---|
| H-001 | Per-subset router beats uniform RAG-FT | For subsets where retrieval-only ≥ RAG-FT on `val_tune`, answer by retrieval copy; else generate | sweetlhare's winning split; retrieval-only was already strong |
| H-002 | Row-level confidence routing beats subset routing | If top-1 cosine ≥ τ (tuned per subset on `val_tune`), copy; else generate | some subsets are mixed; near-duplicates exist everywhere |
| H-003 | Same-subset retrieval helps | Restrict neighbours to the same `subset` (same language/country) | cross-language neighbours may add noise to the prompt |
| H-004 | Reranking improves neighbour quality | BGE-M3 top-20 → `BAAI/bge-reranker-v2-m3` → top-k | cheap, usually a clear gain for retrieval |
| H-005 | Hybrid retrieval | BGE-M3 dense + its sparse/lexical scores (or BM25), fused | ROUGE rewards lexical overlap |
| H-006 | Length calibration | Compare predicted vs reference token length per subset; adjust `max_tokens` / stopping, or add length guidance to the prompt | ROUGE F1 penalizes both over- and under-length |
| H-007 | MBR decoding | Sample N=8 candidates, pick the one with the highest mean ROUGE against the other candidates and the retrieved answers | a known ROUGE booster; cheap with vLLM |
| H-008 | LoRA dropout 0.5 is too high | dropout 0.05 | 0.5 is unusual for LoRA; may be underfitting |
| H-009 | k sweep beyond 3 | k ∈ {4, 5} (watch token length) | the reference only tried 1–3 |
| H-010 | Base model choice | Iterate fast on a smaller model (e.g. Sunflower-14B or a similar Qwen-3 size), then scale the winner; try Gemma-family / AfriqueGemma (the reference author's open question) and our existing Aya-Expanse-8B | model family was the author's biggest untested variable |
| H-011 | Amharic scoring behaviour | Check whether `rouge-score`'s default tokenizer drops Ge'ez script; if so, measure Amharic under both tokenizations and decide where effort goes | could be a silent zero-weight subset |
| H-012 | Final model on train+val | Retrain the locked recipe on train+val (no `val_holdout` scores after this — decision note required) | more data for the final submission only |

## Per-experiment note template (`vault/experiments/EXP-XXX-<slug>.md`)

```markdown
---
type: experiment
id: EXP-XXX
hypothesis: "[[H-00Y-...]]"
parent: "[[EXP-<current best>]]"
git_sha: <sha>
created: <date>
status: kept | reverted | failed
evaluator: PASS | FAIL | n/a
---
## Change
<the single change, as a config diff>
## Scores (val_holdout)
| subset | R1 | RL | judge | weighted | Δ vs parent |
...
Overall weighted: X.XXXX (95% CI of Δ: [a, b])
## Zindi (if submitted)
public: ... | private: ...
## Takeaway
<one or two sentences; update the hypothesis status>
## Links
[[00_INDEX]] · [[H-00Y]] · [[EXP-parent]]
```

## End of each loop run

1. Update every touched H-note's `status`.
2. Update `00_INDEX.md` → "Current best" and "Open hypotheses".
3. Add new hypotheses the results suggest (with reasoning) to the queue.
4. Commit vault + code changes with a message naming the EXP IDs.
5. Report: what was tested, what was kept, the new best score, and the
   next three hypotheses in priority order.
