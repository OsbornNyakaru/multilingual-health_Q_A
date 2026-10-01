# Sprint Plan — final push to 2026-06-21

This replaces the stale `docs/PLAYBOOK.md` 11-week schedule. The competition
**closes 2026-06-21**. As of writing there are ~15 days left, not 11 weeks. The
phased augmentation/ensemble marathon is no longer affordable. This is a sprint:
maximise iteration speed and trust the leaderboard.

## The three facts that should drive every decision

1. **74% of the score is ROUGE** — ROUGE-1 F1 (0.37) + ROUGE-L F1 (0.37). The
   LLM-judge is only 0.26. AfroLM-BertScore is **not on the leaderboard** (it is
   a host-side secondary check on top solutions only). Optimise lexical overlap
   with the reference answers first; do not spend effort on AfroLM.
2. **English is the majority of the competition.** ~61% of train rows and ~56%
   of test rows are English (`Eng_Uga` alone is 25%). This is mostly an English
   health-QA task wearing a multilingual costume. Win English first.
3. **ROUGE is F1, so answer length is a free lever.** Reference lengths vary
   wildly by subset (Amharic ~20 words, Eng_Eth ~24, Akan ~106, Eng_Uga ~95).
   Match generated length to the per-subset reference median.

## Non-negotiables (carried from CLAUDE.md / LESSONS.md)

- **Keep a held-out slice you NEVER train on.** `data/processed/held_out.csv`
  already exists — protect it. Do not fold it (or all of `Val.csv`) into
  training, or you lose your only honest local signal. This was the EY-2026
  collapse cause.
- **Local truth = ROUGE-1/ROUGE-L F1 on the held-out slice, per subset**, using
  the exact `rouge-score` library. This is perfectly replicable offline and is
  74% of the board. Compute it for real — never print estimated/placeholder
  scores.
- **Every submission gets a logged row in `experiments/LOG.md` BEFORE you
  submit.** No exceptions. (This has already slipped once — the May 12
  submissions were never logged.)
- **Submission = exactly 4 columns**: `ID, TargetRLF1, TargetR1F1, TargetLLM`,
  three target columns identical, plain UTF-8 (no BOM), CSV not XLSX. The
  patched `submission/format.py` enforces this — mirror `SampleSubmission.csv`.

## Day-by-day

### Days 1–2 — Honest baseline on the board
- Lock the held-out slice; build the per-subset ROUGE evaluator and confirm it
  runs on held-out predictions.
- Get ONE real submission up: a strong instruct model, zero/few-shot, in-language
  prompts ("answer in the same language"), greedy or beam=4, with per-subset
  length caps. Establish the local→public-LB correlation. You are blind until
  this exists. **Lock it as your floor.**

### Days 3–9 — One fine-tune, iterate on cheap ROUGE levers
- A single QLoRA fine-tune on the provided `Train.csv` pairs only. Do NOT chase
  external data (Sunbird SALT, GhanaNLP, MedMCQA) — too slow, high variance, and
  ROUGE rewards matching THESE reference answers, not generic health text.
- Iterate on the things that move ROUGE cheaply and measurably on held-out:
  prompt phrasing, per-subset length calibration, light output post-processing
  (strip preambles like "Sure, here is...").
- Submit deliberately: ≥ +0.003 held-out evidence over current best, one LB slot
  per clearly different idea. Log every one.

### Days 10–13 — Per-subset error analysis
- Break held-out ROUGE down by the 8 subsets. Your worst subsets are your
  biggest point reservoir. Likely weak: Amharic (tiny, 61 test rows), Akan
  (long answers, hard to match). Only now consider targeted help for the
  weakest 1–2 subsets if a clear, measured gap shows.
- Ensembling only earns its place if a SECOND model genuinely disagrees and
  improves held-out ROUGE — and if you ensemble, **select candidates by
  ROUGE-to-reference proxy, never by AfroLM similarity** (which scores 0).

### Days 14–15 — Lock and audit
- Use `scripts/select_final.py` to pick the final two: one "best public" and one
  "best held-out", to hedge the public→private shakeup. Final pick math uses
  honest held-out OOF, never raw public LB (Principle 9).
- Run `scripts/verify_reproducibility.py`. You are aiming top-10, which means a
  72-hour code audit — seed everything, pin deps, no custom packages in the
  final notebook.

## Things to stop doing
- Building infrastructure instead of improving the score. The scaffold is done;
  freeze it.
- Optimising or reporting AfroLM-BertScore as if it were the target.
- Printing projected/estimated scores. Measure on held-out or say "unknown".
- Submitting without a logged hypothesis.
