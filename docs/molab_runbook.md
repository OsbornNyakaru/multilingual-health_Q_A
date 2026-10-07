# molab runbook: the experiment runner

how osborn and vera run experiments on molab. the experiments themselves are written and queued
from the mac (by osborn or claude code); molab only runs them. the full loop is in
`autoresearch_nlp/program.md`.

## the idea in five lines

- every experiment is a small spec file: one change compared with the current best.
- the spec goes into a queue in our private hugging face repo `nyakaruosborn/afro-health-qa-runs`.
- the runner notebook on molab takes specs from the queue, runs them on the gpu, and uploads the results.
- on the mac, `scripts/exp.py pull` scores the results, decides keep or drop for each language, and updates `experiments/RESULTS.md`.
- `experiments/RESULTS.md` is the record. feed it to the note-taker; notion mirrors it.

## one-time setup

- hugging face token for molab: a **fine-grained** token with
  - read on `nyakaruosborn/afro-health-qa-data`
  - write on `nyakaruosborn/afro-health-qa-runs`
  - read access to public gated repos (for gemma and other licence-gated models; accept each model's licence on its hugging face page first)
- open the runner from github in molab and save it to your workspace:
  `https://github.com/OsbornNyakaru/multilingual-health_Q_A/blob/main/notebooks/molab_runner.py`
- request access to `Sunbird/Sunflower-32B` on hugging face (needed later; approval is manual).

## every molab session (3 steps)

1. turn on the gpu (rtx pro 6000).
2. paste the token into the password box at the top. check the lines under it:
   - "signed in as nyakaruosborn"
   - "gpu: rtx pro 6000, ~96 gb"
   - "data: ready"
3. press **run queue** with **keep watching** on. leave the tab open.
   - it runs every queued spec, one after another, and shows a live log.
   - with keep watching on, it checks for new specs every minute and stops after the idle time you set (default 30 minutes).

that's it. you don't type run names, pick models or copy scores: the spec carries all of that.

## what you see in the log

- `▶ exp008_…  (3 pending)`: a run started, and how many are waiting.
- `held_out: 2,088 rows, pool 27,727` and `val: 6,686 rows, pool 29,815`: each run is tested on two sets.
- `held_out: combined 0.3085 (r1 …, rl …)`: a quick score. the official verdict comes from `exp.py pull` on the mac.
- `■ exp008_…: ok`: finished and uploaded. other endings:
  - `crash`: the error is uploaded. claude code reads it, fixes the code and queues a new spec.
  - `timeout`: the run hit its time limit.

## if the session dies

- molab shuts down after 90 idle minutes and after 12 hours.
- reopen the runner, paste the token, press run queue. generation runs pick up from their last checkpoint (saved every 5 minutes); short runs simply restart.

## on the mac (osborn or claude code)

- `python scripts/exp.py new --from best --set embedder='"BAAI/bge-m3"' --hyp H-011 --desc "dense retrieval"`
- `git add -A && git commit -m "…" && git push`
- `python scripts/exp.py submit`
- `python scripts/exp.py pull --wait` (waits for molab, then updates `experiments/RESULTS.md`, `BEST.json` and the vault)
- `python scripts/exp.py status` shows queued, running and finished runs.
- `python scripts/exp.py submission` builds the composite zindi csv once every best run has test predictions.

## reading experiments/RESULTS.md

- **current best (per subset)**: for each language/country, the run that scores best, with its held-out and val scores. a run takes over a subset if it wins by ≥ 0.003 on **val** and doesn't lose more than 0.003 on held-out.
- **composite test-mix**: the overall score of that per-subset best, weighted by how much of the test set each subset makes up (eng_uga 28%, aka_gha 19%, eng_gha 19%, lug_uga 14%, swa_ken 9%, eng_ken 6%, amh_eth 2%, eng_eth 2%).
- **runs**: every experiment, newest first, with what changed and which subsets it won.
- scores are 0.37 × rouge-1 + 0.37 × rouge-l, so the maximum is 0.74 (the ai-judge part isn't measured locally).

## the interactive notebook

- `notebooks/molab_afro_health_qa.py` is still there for poking at a model by hand. results from it don't count unless they're re-run through the runner.

## when things go wrong

- "not signed in": paste the token again.
- "data: missing": the token can't read `afro-health-qa-data`.
- crash on upload / 403: the token can't write to `afro-health-qa-runs`.
- "prepare.py hash … refusing to run": someone changed the frozen scoring code. stop and tell osborn.
- "held-out fingerprint … split drifted": the data or split changed. stop and tell osborn.
- gated repo / 401: accept that model's licence on hugging face with the same account.
- out of memory: claude code lowers the batch size in a new spec.
