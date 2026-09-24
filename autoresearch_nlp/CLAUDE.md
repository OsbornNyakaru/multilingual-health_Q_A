# CLAUDE.md — Agent Entry Point (Generative-NLP track)

You are an autoresearch agent on a **generative NLP** competition: the task is
to read a prompt (a question / source text) and GENERATE a text answer that is
scored against a reference by ROUGE and/or an LLM-judge (not a classification
label). This is a different game from the tabular track in the parent folder —
there is no feature matrix, no out-of-fold probability, no threshold, and no
"probe leakage". Do not import tabular instincts here.

## First read these in order

1. **README.md** — what this track is and the three files that matter.
2. **PREFLIGHT.md** — day-1 hard-stop checklist. Do not write generation code
   until every HARD STOP is ticked.
3. **program.md** — the experiment loop (autoresearch pattern, ROUGE-scored).
4. **AGENT_PLAYBOOK.md** — ready-to-run sub-agent prompts for NLP tasks.
5. **LESSONS.md** — hard-won lessons. The afro-health-qa case at the top is the
   most important; read it twice.

Then look at `data/raw/`, read the competition rules, and confirm the EXACT
metric and weights on the competition page before touching `prepare.py`'s CONFIG.

## The three files that matter (mirrors karpathy/autoresearch)

- **`prepare.py`** — FROZEN harness: data prep, the held-out split, and the
  exact scoring metric. You never edit it after the day-1 CONFIG is filled.
- **`train.py`** — the SINGLE file you edit. Prompts, model, decoding, length
  bounds, post-processing, optional fine-tuning. One change per experiment.
- **`program.md`** — the loop you execute. The human edits this over time; you
  follow it.

## Operating principles (do not deviate)

### P1 — Know your metric before your model
Most generative-NLP comps are scored mostly by **ROUGE** (lexical overlap with
the reference). Confirm the weights on the competition page and put them in
`prepare.py`. If ROUGE dominates, your job is to match the reference answers'
words and length — not to write the "best" answer. Optimise the metric, not
your taste.

### P2 — Frozen harness, single editable file
`prepare.py` defines the metric and the split and is never modified mid-run.
You edit `train.py` only. This guarantees comparable results across experiments.

### P3 — One change per experiment
Each commit changes exactly ONE lever: prompt, OR decoding, OR length bounds, OR
adapter, OR post-processing. Bundling makes ablation impossible.

### P4 — Never train on the held-out slice
`prepare.make_splits()` reserves `held_out.csv`. It is your only honest local
signal. Do not fine-tune on it, do not draw few-shot examples from it. Folding
your validation set into training is how you go blind (see LESSONS.md).

### P5 — Local truth = ROUGE on held-out, per subset
Always compute ROUGE-1/ROUGE-L F1 on the held-out slice with the frozen harness,
broken down per subset. It is replicable offline and is the bulk of the score.
Never print an estimated score where a measured one belongs.

### P6 — Length is a free lever
ROUGE is F1, so generated length materially changes the score. Calibrate output
length per subset to the reference median (`tools/length_calibrate.py`).

### P7 — Trust LB, watch CV-LB transfer
The public LB is ground truth for generalization. Track your held-out → public
transfer ratio. If held-out jumps but LB doesn't, your held-out is unrepresentative
(often: you leaked Val into training, or a subset is mis-weighted).

### P8 — Lock the floor early
The moment one submission beats the host baseline, lock it as a final pick. Keep
experimenting on top; your downside is protected.

### P9 — Submit deliberately
Each LB slot is a controlled experiment. Submit only changes with held-out
evidence of a real gain over your current best. Three regressions in a row = stop
for the day and rethink the mechanism, not the parameter.

### P10 — Reproducibility for the top-N audit
Hosts audit top finishers. Seed everything, pin deps, no custom packages in the
final notebook, and make sure a fresh run reproduces your submission.

## Submission rhythm
Day 1: scaffold + frozen CONFIG + honest baseline submission (lock the floor).
Day 2-N: per day = a few held-out experiments, one LB submission of the best new
variant. Final day: lock final pair 6h before deadline (hedge best-public vs
best-held-out), run the reproducibility check.

## Anti-patterns (carried from real runs)
- Optimising a metric the host does NOT score (e.g. ranking ensemble candidates
  by an unscored semantic-similarity metric).
- Mixing languages within a generation batch so per-subset length caps misapply.
- `max_seq_length` so small it truncates long reference answers during fine-tune.
- One English fallback string inserted into every language's predictions.
- Printing projected/placeholder scores instead of measuring on held-out.
