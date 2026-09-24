# COMPETITION_INTEL.md — grounded findings

Sourced from the official starter notebook, the two webinar PDFs, and direct
measurement on the local data. Updated 2026-06-07. Facts are marked; inferences
are labelled as such.

## 1. The official starter notebook = the public-LB floor

`multilingual_health_qa_starter_notebook.ipynb` ships TWO baselines that most of
the ~440 active competitors will start from:

- **Baseline 1 — TF-IDF retrieval.** Char-wb n-grams (3-5), per-subset
  NearestNeighbors (cosine), returns the nearest TRAINING answer verbatim. No
  model. (FACT — from the notebook.)
- **Baseline 2 — multilingual seq2seq LLM.** `google/mt5-small|base` or
  `facebook/nllb-200-distilled-600M`, zero-shot then optionally fine-tuned
  (3 epochs, lr 5e-5, full fine-tune, bf16). (FACT.)

Two weaknesses baked into the starter that you can exploit (FACT, from reading
the code):
- `build_prompt()` returns the **raw question** — the language-conditioning line
  is commented out. So the default does NOT tell the model which language to
  answer in. Uncommenting / adding the language name is a free improvement.
- ROUGE is scored with a **whitespace tokenizer** (not the rouge-score default).
  Our harness now matches this (`_WhitespaceTokenizer` in prepare.py).

## 2. The metric uses whitespace-tokenized ROUGE (important for Amharic)

The starter scores ROUGE with `WhitespaceTokenizer` and `use_stemmer=False`,
noting it is "safe for non-English scripts." The rouge-score DEFAULT tokenizer
lowercases and regex-splits, which can gut Amharic Ge'ez. (FACT that the starter
does this; INFERENCE that the host grader does too — likely, but not 100%
confirmed. If you can confirm on the forum, do.) Our `prepare.py` now uses
whitespace tokenization so local scores track the LB.

## 3. Retrieval is a genuinely strong baseline — measured

Measured on our stratified held-out slice (fit retrieval on work_train, predict
held_out), using a CRUDE token-overlap nearest-neighbour (weaker than the
notebook's char-ngram TF-IDF, so treat as a LOWER bound), whitespace ROUGE:

```
subset       n  exact%   R1     RL    comb*  simJ
Aka_Gha    312   0.0%  0.294  0.174  0.173  0.42
Amh_Eth    129   0.0%  0.117  0.110  0.084  0.31
Eng_Eth    274  41.6%  0.569  0.555  0.416  0.76
Eng_Gha    311   0.0%  0.249  0.165  0.153  0.42
Eng_Ken    146   0.0%  0.459  0.406  0.320  0.46
Eng_Uga    534   2.8%  0.472  0.426  0.332  0.65
Lug_Uga    237   0.0%  0.443  0.416  0.318  0.43
Swa_Ken    145   0.0%  0.544  0.505  0.388  0.49
ALL       2088   6.2%  0.404  0.350  0.279
```

`comb*` = ROUGE-only (0.37·R1 + 0.37·RL); the judge (0.26) adds on top on the
real LB. So pure retrieval is worth ~0.28 ROUGE-only here, and the notebook's
char-ngram version will beat this crude one. (FACT — measured.)

Read this carefully:
- **`Eng_Eth` has 41.6% identical questions** between held-out and train, mean
  similarity 0.76 → retrieval almost solves it (R1 0.569). `Eng_Uga` similarity
  0.65. The dataset has heavy question repetition/paraphrase. (FACT.)
- **Amharic is the hard floor** (0.084) — low overlap, short answers. Akan is
  also weak. These two subsets are where a real model must earn its keep.
- A small zero-shot mT5 scores near 0 on ROUGE (it paraphrases, low lexical
  overlap). So on this metric, **retrieval beats zero-shot generation**, and
  likely beats a lightly fine-tuned small model on the high-overlap subsets.

## 4. A rules-legal external dataset exists

The HASH webinar (Dr Elizabeth Oseku) states HASH published the **HASH
Crowdsourced Dataset**: ~5,488 STI/sexual-health Q&A pairs, long+short answer
formats, health-worker validated, open-source on **Harvard Dataverse**. (FACT —
from the PDF.) The rules permit freely/openly-available external data. This is
same-consortium, same-domain English data → a legitimate augmentation lever for
the English subset (56% of the score). ACTION: locate it on Harvard Dataverse,
verify the exact license, check for overlap with the competition test set before
using.

## 5. Recommended strategy (inference, grounded in the above)

The structure points to a **retrieval + generation hybrid**, selected per row by
the scored metric:

1. For each test question, retrieve the nearest train question (char-ngram
   TF-IDF, per subset) and its answer; record the similarity.
2. If similarity ≥ a tuned threshold → use the retrieved answer (it is real,
   fluent, correct-language reference text → high ROUGE). This wins the
   high-overlap subsets (much of English).
3. Else → use a fine-tuned generative model (mT5/NLLB or a stronger licensed
   multilingual model) → for Amharic, Akan, and low-similarity rows.
4. Calibrate length per subset (LENGTH_BOUNDS already wired) and answer
   in-language (prompts already wired).
5. Select the per-row threshold on the held-out slice by ROUGE — NOT by AfroLM.

This is the cheapest path above the starter floor in the time remaining, and it
directly exploits the measured question-duplication structure. Pure fine-tuning
alone leaves the easy retrieval points on the table.

## 6. Still unread
The discussions threads (JS-rendered, sign-in gated) — paste them in or connect
Claude-in-Chrome and I'll fold in anything actionable (host clarifications,
shared baselines, leaderboard scores to calibrate the floor against).
