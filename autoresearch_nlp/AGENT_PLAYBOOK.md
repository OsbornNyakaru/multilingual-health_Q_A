# AGENT_PLAYBOOK.md — Sub-Agent Prompts (Generative-NLP)

Ready-to-run prompts for specialist research tasks. Paste into a sub-agent.
Each states the goal, constraints, deliverable format, and a word cap. Vague
prompts produce vague output. Demand citations; cap word counts; ask for an EV
estimate where a recommendation is involved.

---

## 1. Rules-text deep audit
```
Read the competition rules at <URL> line by line. Produce RULES.md content:
- Every "you may not" clause, verbatim, with the source URL.
- Every "you may" clause (permitted leverage often hides here: external open
  data, pretrained models, augmentation).
- The EXACT scoring metric and weights, quoted. Flag which metrics are on the
  leaderboard vs host-side secondary checks.
- Submission schema (columns, identical-target requirement, file type, encoding).
- Caps (daily/total submissions, team size) and the top-N code-review duty.
Cap 1200 words. Quote and cite every clause.
```

## 2. Competition forum mining
```
Exhaustively read the competition's discussion forum at <URL>. Report:
- Every host clarification on data, allowed models/tools, and metric edge cases.
- Any official/starter baseline notebook and the score it reaches.
- Common pitfalls competitors report (submission format, language handling).
- The most recent rule change/clarification, quoted.
Cap 1200 words. Cite each item with the thread URL.
```

## 3. Model survey for the languages + compute budget
```
We must answer health questions in <LANGUAGES> and are scored mostly by ROUGE.
Compute budget: <e.g. single T4/L4/A100, Colab/Kaggle>. Open-source only.
Find openly-available models that fit:
- Multilingual/instruct backbones that cover these languages (verify each HF id
  actually exists and is openly licensed — do NOT invent ids).
- Their size vs our VRAM (can it run in 4-bit on the budget GPU?).
- Evidence they handle the target languages (benchmarks, model card, papers).
Rank by expected fit. Flag any id you could not verify.
Cap 1000 words. Cite model cards / benchmark sources.
```

## 4. ROUGE-aware answer-shaping research
```
The metric is ~74% ROUGE F1 against reference answers. Research how to maximise
ROUGE on generative QA without overfitting:
- Length calibration effects on ROUGE-1/ROUGE-L F1 (precision/recall balance).
- Whether mimicking reference phrasing/structure helps vs free paraphrase.
- Decoding choices (beam vs sampling, length_penalty, no_repeat_ngram) and their
  documented ROUGE deltas.
- Post-processing that reliably lifts ROUGE (preamble stripping, dedup).
Give concrete, testable train.py changes, one lever each, with expected sign of
effect. Cap 1000 words. Cite sources; mark estimates as estimates.
```

## 5. Fine-tuning plan (QLoRA / seq2seq)
```
Plan a single fine-tune on the provided train pairs only. Specify:
- Base model + why (must fit the compute budget in 4-bit).
- LoRA config (r, alpha, target modules) and why.
- max_seq_length chosen so the LONGEST reference answers are NOT truncated
  (check the per-subset length distribution first).
- Prompt/answer formatting identical to inference.
- Epochs/LR/batch given the GPU, with an OOM-avoidance note for the budget GPU.
- The ONE held-out metric you will read to decide keep/revert.
Cap 900 words. No code dumps longer than the train loop.
```

## 6. Held-out representativeness (CV-LB divergence)
```
Our held-out combined is <X> but public LB is <Y> (gap <Z>). Diagnose:
- Is held-out trained/selected on by any step? (leak check)
- Does the held-out per-subset mix match the test per-subset mix?
- Is any subset weighted differently than we assume in the host aggregation?
- Recompute an honest held-out that mirrors the test subset proportions.
Predict the true public score and recommend the fix.
Cap 800 words. Show the per-subset math.
```

## 7. Final-pair selection
```
With <N> submissions left and <K> LB-tested in hand, choose the final TWO.
Use HELD-OUT combined (not raw public LB) as each candidate's mean; public LB on
a small slice is noisy. Pick one "best public" and one "best held-out" to hedge
the public→private shakeup. Account for per-subset stability (a candidate strong
only on the majority subset is fragile if the private mix differs).
Output: ranked table (held-out, public, per-subset spread), recommended pair,
explicit justification, and a worst-case floor. Cap 700 words.
```

## 8. Honest stop decision
```
Today's submissions + scores: <LIST>. Current best: <S>. Slots left: <N>.
Deadline: <T>. Estimate P(next beats best) x E[gain | beats] vs the value of
holding the slot. If EV < ~0.0005 combined, recommend STOP. Cite today's
regression count as the base rate. Cap 400 words. Decision: SWING or STOP.
```

## How to use these
Replace `<...>` placeholders with specifics. For research that needs no code,
run 2-4 agents in parallel on different angles, then synthesise. Always: specific
goal, citations required, word cap, EV estimate for recommendations.
