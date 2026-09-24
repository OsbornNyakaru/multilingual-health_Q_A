# LESSONS.md — Generative-NLP

Append a post-mortem after every competition's private reveal. The most valuable
entry is always "the one thing that would have changed the outcome." Codify each
lesson back into this track so the next competition starts with the fix in place.

---

## Case study — afro-health-qa (Zindi Multilingual Health QA, 2026)

The competition that this track was built from. Lessons, in order of impact:

### 1. We optimised the wrong metric for weeks
The repo hard-coded a scoring formula of ROUGE-1 0.25 / ROUGE-L 0.25 /
AfroLM-BertScore 0.30 / Judge 0.20. The ACTUAL leaderboard was ROUGE-1 0.37 /
ROUGE-L 0.37 / Judge 0.26, with AfroLM-BertScore **not on the board at all**
(host-side secondary check only). Every model-selection decision was made against
a metric that scored zero on the leaderboard.
**Fix codified:** `prepare.py` METRIC_WEIGHTS must be copied verbatim from the
rules page on day 1 (PREFLIGHT A). Unscored metrics get weight 0.

### 2. We misread which language was the majority
We framed the task as "Luganda, Kiswahili, Akan, Amharic" — the exotic languages.
In reality English was ~61% of train and ~56% of test rows (one English locale
alone was 25%). More than half the score came from English, which we treated as
an afterthought.
**Fix codified:** PREFLIGHT B requires counting rows per subset for train AND
test and explicitly naming the majority subset before modelling.

### 3. We ignored length as a lever
Reference answers ranged from ~20 words (Amharic) to ~106 (Akan). A single output
length tanks ROUGE F1 at both ends. We left this on the table.
**Fix codified:** `tools/length_calibrate.py` + per-subset LENGTH_BOUNDS in
train.py; Principle 3.

### 4. A notebook folded Val into training and then printed fake scores
An exp004 notebook concatenated all of Val.csv into the training set "because the
answers are gold", which destroyed the only honest local signal — then printed
hardcoded score estimates ("ROUGE-1: 0.63-0.70", "Target LB 0.768095") that were
never measured.
**Fix codified:** P4/P5 — never train on held-out; never print an estimate where
a measurement belongs. `prepare.make_splits()` reserves held_out.csv and the loop
scores on it.

### 5. The ensemble selected by an unscored metric
The ensemble picked between two models by AfroLM-BertScore similarity to a
centroid — a metric worth 0 on the board — while ignoring ROUGE (74%).
**Fix codified:** Principle 7 — rank ensemble candidates by the SCORED metric.

### 6. Engineering bugs that quietly cost score
- Batches sorted only by prompt length mixed languages, so per-subset length caps
  applied to the wrong rows. Fix: batch by subset (done in train.py `generate`).
- `max_seq_length=256` truncated the longest reference answers during fine-tune.
  Fix: set it from the per-subset length distribution.
- One English fallback string ("Information not available.") was inserted into
  every language's predictions, tanking non-English ROUGE.
- Submission written as `.xlsx` and with a stale 5th column; the host wants exactly
  4 CSV columns with identical target values. Fix: `prepare.build_submission` +
  `validate_submission`.

### The one thing that would have changed the outcome
Reading the metric weights off the rules page into the harness on day 1. Every
other error compounded from optimising the wrong objective. Instrumentation, not
modelling, was the gap.

---

## Post-mortem template (fill in after each private reveal)
```
Competition:
Final public / private rank and score:
Held-out → public transfer ratio (was it stable?):
What worked (ranked by LB impact):
What didn't (and why we thought it would):
Biggest time sink with no payoff:
The ONE thing that would have changed the outcome:
Fix to codify into this track:
```
