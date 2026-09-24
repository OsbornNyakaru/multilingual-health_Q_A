# RULES.md — Multilingual Health QA (Zindi, ITU/HASH)

Source: https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge
Captured 2026-06-07. **Re-read the page before final submissions** — Zindi can amend rules.

## Task
Generate a fluent, accurate health answer in the SAME language as the question.
Topics: maternal, sexual and reproductive health (MSRH). Open to all.

## Metric (leaderboard)
Weighted mean of THREE metrics:
- ROUGE-1 F1 — weight **0.37**
- ROUGE-L F1 — weight **0.37**
- LLM-as-a-Judge (1-5, normalised to [0,1]) — weight **0.26**

ROUGE via the `rouge-score` library. **AfroLM-BertScore F1 is NOT on the
leaderboard** — it is applied by the host to TOP solutions only, as a secondary
check. (Encoded in prepare.py METRIC_WEIGHTS; AfroLM weight = 0.)

## Submission format
Exactly 4 columns: `ID, TargetRLF1, TargetR1F1, TargetLLM`. The three target
columns must hold IDENTICAL values per row. CSV (not xlsx), plain UTF-8.

## Splits + caps
- Public LB ≈ 50% of test; Private LB ≈ 50% (revealed at close). (Note: the page
  states 50/50 in one section and ~30/70 in another — treat the private set as
  the majority and do not overfit the public slice either way.)
- 5 submissions/day, 50 total. Choose 2 final submissions before close; else the
  2 best public are auto-selected.
- Team size ≤ 4.

## "May not" / disqualifiers
- Open-source languages and tools ONLY. No paid services, no free trials needing
  a credit card. No AutoML.
- No multiple accounts; no sharing code privately outside a team. Shared code must
  be posted publicly on the platform.
- Data under CC-BY-SA 4.0. Solution must not infringe third-party rights; winners
  assign copyright of winning code to Zindi.
- Data leaks or anything compromising solution value → disqualification.

## "May" / permitted leverage
- Pretrained models allowed IF openly available to everyone.
- Freely + operationally available external datasets allowed (available within
  one month of acquisition). The provided data is the core.

## Top-N obligation
Top 10 on private LB get an email at close requesting model + code + report; 72h
to submit (the page also mentions 48h elsewhere — assume the shorter). Code must
reproduce the score and run; no custom packages in the submission notebook; seed
everything.

## Timeline
Closes 2026-06-21. Private LB revealed at close.

## Implications baked into this instance
1. 74% of the score is ROUGE → match reference wording + length, per subset.
2. English is ~56% of the test set → win the majority first.
3. Length is a free lever (LENGTH_BOUNDS in train.py).
4. The submission writer enforces the 4-col identical-target schema.
5. Model licensing matters for the top-10 audit → see MODEL_DECISION.md.
