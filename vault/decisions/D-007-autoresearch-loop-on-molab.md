---
type: decision
id: D-007
created: 2026-10-01
status: confirmed
links: ["[[D-005-held-out-protocol]]", "[[D-006-data-hosting-private-hf]]", "[[H-011-closed-pool-vs-generative-router]]", "[[EXP-007-ret-tfidfchar-nhall-vall]]", "[[00_INDEX]]"]
---
# D-007 Experiments run through an autoresearch loop: spec → HF queue → frozen molab runner → local verdict

**Decision:** Experiments are no longer run by hand in the interactive notebook. Each one is a spec (`experiments/specs/EXP-NNN.json`, one change vs its parent) queued in the private HF dataset `nyakaruosborn/afro-health-qa-runs`. The frozen runner `notebooks/molab_runner.py` fetches the repo at the spec's git SHA, refuses to run if `prepare.py` doesn't match its pinned sha256 or the held-out fingerprint isn't `a8026f24ea3d`, runs `autoresearch_nlp/experiment.py` (the one editable file), and uploads predictions plus `result.json`. Locally, `scripts/exp.py pull` scores, decides and writes `experiments/RESULTS.md`, `BEST.json`, `results.jsonl` and a vault note. The full procedure is in `autoresearch_nlp/program.md`; the human steps are in `docs/molab_runbook.md`.

**Evaluation protocol (extends [[D-005-held-out-protocol]]):**
- Every run is scored on held-out (pool = work_train) **and** on Val (pool = Train). Val has no exact duplicate questions, like Test, while held-out has 41.6% exact repeats in Eng_Eth; Val is the better test proxy.
- The summary is the test-mix weighted average over subsets (weights = subset share of Test.csv).
- Adoption is per subset: a run takes over a subset only if it wins by ≥ +0.003 on both sets, and only full-set runs are eligible. The composite of per-subset bests is the current best ([[H-011-closed-pool-vs-generative-router]] done as bookkeeping).

**Why:** No hand-typed run names (the checkpoint-resume trap), no copying scores out of a dying molab session, results that persist (HF) and land in the repo automatically, and per-subset decisions, because the 2026-10-01 strategy review showed four subsets (~58% of Test) are closed-pool: about 90% of Val answers there already exist verbatim in Train.

**Verified 2026-10-01:** full loop run end to end with [[EXP-007-ret-tfidfchar-nhall-vall]]. The runner was started locally (CPU) against the real queue; code was fetched by SHA, both checks passed, results were uploaded, pulled and recorded.

**Alternatives rejected:** manual notebook runs (error-prone, nothing persists); Claude driving the molab kernel via marimo pairing (per-session token, long blocking calls untested; possible later add-on); GitHub raw files as the queue (cached up to 5 minutes, rate-limited).

**Open risks (test on molab):** whether a long watching loop counts as activity against the 90-minute idle shutdown; Blackwell support of the auto-installed torch; HF commit rate during checkpoints.

## Links
- [[D-005-held-out-protocol]]
- [[D-006-data-hosting-private-hf]]
- [[H-011-closed-pool-vs-generative-router]]
- [[EXP-007-ret-tfidfchar-nhall-vall]]
- [[00_INDEX]]
