# PRINCIPLES.md — Core principles for generative-NLP competitions

## 1. The metric is the target, not your taste
A fluent, factually perfect answer that shares few words with the reference loses
on a ROUGE-weighted metric. Read the weights first; optimise what is scored.

## 2. Find the majority subset
In multilingual comps the score is dominated by whichever language/locale has the
most test rows — often the high-resource one (e.g. English), not the exotic one.
Win the majority first; it is the largest and usually the easiest reservoir.

## 3. Length is a lever, not an afterthought
ROUGE is F1. Too short kills recall; too long kills precision. Calibrate output
length per subset to the reference median. This is one of the cheapest, most
reliable gains available.

## 4. The first submission is honest and early
Ship a clean baseline to the LB before optimising. It anchors your held-out → LB
transfer ratio and gives you a known floor to lock.

## 5. Protect one honest held-out
Never train, fine-tune, few-shot, or select on the slice you score on. The moment
you fold validation into training you are flying blind and your local numbers
become fiction.

## 6. Prompts and decoding before fine-tuning
In-language prompts, length bounds, and decoding choices are minutes of work and
often move the score more than a day-long fine-tune. Exhaust the cheap levers
before the expensive ones.

## 7. Ensembles need genuine disagreement and the right selector
Two models help only if their errors decorrelate AND you pick between them by the
SCORED metric. Selecting candidates by an unscored similarity metric is a common,
silent score-killer.

## 8. Every choice flows to the test predictions
Prompt, length cap, post-processing, decoding seed — each changes what you submit.
Track one lever at a time so a regression is attributable.

## 9. Compute is cheap; LB slots are not
Run several held-out experiments per LB submission. Each slot is a controlled
experiment — submit the best held-out candidate, not the most recent edit.

## 10. The deadline is a feature
Lock the final pair 6 hours out. Panic submissions in the last hour have negative
expected value; measured submissions made earlier do not.
