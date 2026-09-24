# program.md — The Autoresearch Loop (Generative-NLP)

Mirror of [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
adapted for text-generation competitions scored by ROUGE / LLM-judge.

## Goal
Maximise the public-leaderboard score within the rules. Protect the private
leaderboard (which decides prizes).

## Hard rules (do not violate)
1. Read the competition rules BEFORE writing code. Capture every "you may not"
   clause verbatim into `RULES.md` with the source URL.
2. Do not modify `prepare.py`. It is the frozen evaluation harness. The only
   day-1 exception is filling its CONFIG block to match the competition, once.
3. Do not train, fine-tune, or draw few-shot examples from `held_out.csv`.
4. Do not submit anything you cannot reproduce. Seed everything.
5. Mirror `SampleSubmission.csv` exactly. For multi-metric comps the target
   columns must be identical per row.

## Setup (first time only)
1. Run through `PREFLIGHT.md`. Every HARD STOP must be ticked.
2. Drop the competition CSVs into `data/raw/` and set `prepare.py` CONFIG
   (schema columns, submission columns, METRIC_WEIGHTS) to match the rules page.
3. Build the split + confirm the harness:
   ```bash
   python prepare.py                 # writes work_train.csv + held_out.csv
   python tools/synth_sanity.py      # end-to-end CPU smoke test (DRY_RUN)
   ```
4. Initialise `results.tsv` with the header:
   ```
   commit	combined	rouge1_f1	rougeL_f1	judge	status	description
   ```

## The experiment loop
```
LOOP:
  1. Look at git state (branch, last commit).
  2. Read the tail of results.tsv to see what's been tried.
  3. Form ONE hypothesis (not a bundle). State the expected effect.
  4. Edit ONE lever in train.py CONFIG (prompt | decoding | length bounds |
     adapter | post-processing). Set RUN_NAME.
  5. git commit -am "<short description>"
  6. Run: python train.py > run.log 2>&1
  7. grep "^combined:\|^rouge1_f1:\|^rougeL_f1:" run.log
  8. If empty / traceback: tail -n 60 run.log, fix or abandon.
  9. Read the per_subset block. A gain concentrated in one subset, or a
     regression hidden inside an average, changes what you try next.
  10. Append the result to results.tsv.
  11. If combined improved by > the noise band: keep the commit, advance.
  12. If equal or worse: git reset --hard HEAD~1, try a different lever.
```

## Noise band
ROUGE on a few-hundred-row public slice has σ ≈ 0.005-0.010. Only submit changes
with held-out evidence of ≥ +0.003 combined over your current best. Smaller lifts
are indistinguishable from noise — do not spend LB slots on them.

## What "one lever" means here
Examples of single, ablatable changes:
- Prompt wording / system instruction (in-language vs English).
- Few-shot: 0 → 2 in-language exemplars (drawn from work_train, never held_out).
- Decoding: greedy → beam=4; length_penalty 1.0 → 0.8.
- Per-subset length bounds (from tools/length_calibrate.py).
- Post-processing: strip preambles, trim trailing boilerplate.
- A fine-tuned adapter (its own experiment; characterise the base first).

## Submission file
`train.py` writes `submissions/<RUN_NAME>.csv` and validates the schema. Upload
to the LB only when held-out combined improves by ≥ +0.003 over your LB-best.

## CV-LB divergence
If held-out says +X but the LB says -Y, your held-out is unrepresentative. The
usual causes: (a) you trained on Val/held-out; (b) a subset is weighted wrong in
your head vs the host's aggregation; (c) the public slice is a different subset
mix than your held-out. Diagnose before submitting again — run the
"Held-out representativeness" prompt in AGENT_PLAYBOOK.md.

## Output format
`train.py` ends every run with this greppable block (from prepare.report):
```
---
name: <run_name>
combined: 0.612345
rouge1_f1: 0.640000
rougeL_f1: 0.585000
judge: 0.000000
n: 2088
per_subset:
  ...
```
Optimise `combined`. When the judge is unavailable locally it is 0; the ROUGE
component still ranks experiments correctly because it is most of the weight.

## Stop conditions
- Three consecutive experiments fail to beat HEAD by > +0.003 → stop tweaking
  parameters; change the mechanism (different base model, fine-tune vs prompt,
  retrieval, different language handling).
- Ten consecutive crashes → run the DRY_RUN sanity test to confirm the harness,
  then resume.
- 6 hours before deadline → lock the final pair. One "best public" and one "best
  held-out" to hedge the public→private shakeup. No more experiments without a
  documented positive-EV reason.

## After the private reveal
Within 24h, write the post-mortem and append it to LESSONS.md. The single most
valuable artifact is the answer to "the one thing that would have changed the
outcome" — codify it into this track so the next competition starts with the fix.
