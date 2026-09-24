# PREFLIGHT.md — Day-1 Checklist (Generative-NLP)

Run through this **before** writing any generation code. Items marked **HARD
STOP** mean: do not write a model line until done. Every item maps to a real
mistake.

---

## A. Rules and metric understanding
- [ ] **HARD STOP** — `RULES.md` exists with every "may not" clause verbatim and
      the source URL (open-source-only, no paid APIs, team size, submission caps).
- [ ] **HARD STOP** — The EXACT metric and weights are copied into
      `prepare.py` METRIC_WEIGHTS. Confirm which metrics are on the leaderboard
      vs which are host-side secondary checks (e.g. a semantic-similarity metric
      applied only to top solutions → weight 0).
- [ ] Submission schema copied from `SampleSubmission.csv` into
      `prepare.py` SUBMISSION_COLUMNS. Note whether multi-metric requires
      identical target columns.
- [ ] Public/private split ratio recorded (Zindi often 50/50 or 30/70).
- [ ] Daily + total submission caps recorded.
- [ ] Top-N code-review / reproducibility obligation recorded (timeframe).

## B. Data understanding (the leverage hides here)
- [ ] Row counts per subset (language/locale) for BOTH train and test. Know
      which subset is the **majority** of the score — that is where most points
      live, and it is often not the exotic one.
- [ ] Reference-answer length distribution per subset (median, p10, p90). This
      sets your per-subset length bounds.
- [ ] Confirm train/val/test have no ID overlap (no leakage).
- [ ] Confirm the answer language matches the question language per subset.

## C. Harness sanity
- [ ] **HARD STOP** — `python prepare.py` runs and writes
      `data/processed/work_train.csv` and `held_out.csv`.
- [ ] **HARD STOP** — `python tools/synth_sanity.py` passes (DRY_RUN end-to-end).
- [ ] Sanity-check the metric: perfect predictions score the ROUGE ceiling,
      empty predictions score ~0. (See the synth test output.)
- [ ] `results.tsv` exists with its header. Git initialised; baseline committed.

## D. Generalization instrumentation
- [ ] You will score EVERY experiment on `held_out.csv` per subset, never on
      data the experiment trained or selected on.
- [ ] You commit to NOT folding `Val.csv` or `held_out.csv` into training.
- [ ] You will track the held-out → public-LB transfer ratio after the first
      real submission.

## E. Final-submission strategy
- [ ] Deadline known in your local timezone.
- [ ] "Lock the floor" plan: first above-baseline submission becomes a final pick.
- [ ] Final pair = one best-public + one best-held-out, to hedge the shakeup.

## F. Day-1 agent prompts (AGENT_PLAYBOOK.md)
- [ ] Rules-text deep audit.
- [ ] Competition forum mining (host clarifications, baseline notebooks).
- [ ] Model survey: which openly-available multilingual/instruct models fit the
      languages and the compute budget (verify each HF id actually exists).

## G. Things you do NOT do on day 1
- Do NOT fine-tune before a zero/few-shot baseline is on the LB.
- Do NOT ensemble before a single model is LB-validated.
- Do NOT optimise an unscored metric.
- Do NOT modify `prepare.py` after CONFIG is frozen.
- Do NOT submit before the held-out score and submission schema both check out.

---

## Sign-off
Once every HARD STOP is ticked, write the date + initials and commit this file.

Preflight completed: ________ (date) by ________ (initials)
