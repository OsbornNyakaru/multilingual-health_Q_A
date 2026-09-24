---
type: fact
id: F-006
created: 2026-09-24
status: confirmed
links: ["[[F-001-competition-metric]]", "[[H-013-whitespace-tokenizer-matches-grader]]", "[[D-001-metric-weights-and-tokenizer]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[FND-001-retrieval-strength-by-subset]]", "[[00_INDEX]]"]
---
# F-006 Local metric replica and per-subset scorer

The local copy of the leaderboard metric ([[F-001-competition-metric]]), measured and tested on CPU on 2026-09-24. **The ROUGE part matches the starter notebook exactly. The judge part is a stand-in, not the grader.**

```
total      = 0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·judge    (NaN when no judge ran)
rouge_only = 0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1                 (judge counted as 0, so a strict lower bound)
```

## Code (in `src/afro_health_qa/evaluation/`)
| File | What it does |
|------|--------------|
| `combined.py` | The single source for the weights: `W_R1=0.37, W_RL=0.37, W_JUDGE=0.26`, and `W_AFROLM_BS=0.0`. The file used to say 0.25/0.25/0.30/0.20, which was wrong. `combined_score()` returns NaN when there is no judge, instead of a silent 0. |
| `rouge.py` | `score_rouge` / `score_rouge_per_example` take `tokenizer=` with one of `whitespace` (the default), `unicode`, `default`, or `default_stem`. |
| `judge.py` | An injectable judge: rubric/prompt constants, `parse_judge_score`, `normalize_judge_score`, `run_judge`, plus lazy `TransformersJudge` / `VLLMJudge` backends and `make_judge`. |
| `scorer.py` | Loads predictions and references, joins them, scores per row, builds a per-`subset` table with an `ALL` row, compares tokenizers, and has a CLI. |
| `tests/test_metric_replica.py` | 26 CPU tests that run in about 1 second. |

Other fixes: `configs/base.yaml` `scoring_weights` changed to 0.37/0.37/0.0/0.26, with a new `rouge_tokenizer: whitespace` key. No code reads these keys; I checked with grep. `local_eval.combined_score` (called by `scripts/run_exp001.py`) now delegates to `combined.py`. `scripts/retrieval_baseline_bge.py` picks up the new defaults (whitespace, 0.37) without any change.

## API usage
```bash
# CPU, ROUGE only (judge = NaN)
PYTHONPATH=src python -m afro_health_qa.evaluation.scorer \
  --predictions preds.csv --references data/processed/held_out.csv \
  [--pred-col prediction] [--tokenizer whitespace] [--compare-tokenizers] [--strict] \
  [--out table.csv] [--rows-out rows.csv]
# molab, with the judge stand-in
... --judge vllm --judge-model google/gemma-3-27b-it [--judge-max-per-subset 60]
```
```python
from afro_health_qa.evaluation.scorer import evaluate, score_rows, summarize
table, rows = evaluate("preds.csv", "data/processed/held_out.csv", judge_fn=None)  # or any prompts->texts callable
```
- Predictions can be `ID` plus one answer column (auto-detected from prediction/answer/response/generated), or a 4-column submission. For a submission the scorer uses `TargetR1F1`→R1, `TargetRLF1`→RL and `TargetLLM`→judge, the same way the platform scores it.
- References must have `ID, input, output, subset`, so Train, Val or `held_out.csv` all work. Test.csv has no answers and is rejected.
- `ALL` is the row-weighted mean, as on the leaderboard. When the judge is sampled per subset, `ALL`'s judge value is the subset-size-weighted mean of the subset means.

## Tokenizer finding (the important part)
**Evidence 1 (unit):** rouge-score's `DefaultTokenizer` lowercases the text and replaces everything outside `[a-z0-9]` with spaces. For `የእርግዝና ምርመራ በሆስፒታል ማድረግ ይመከራል። …` it returns `[]`, so any Amharic pair scores 0.0. Whitespace and `unicode` tokenization both give 8 tokens and R1 = 0.667 on the test pair. The default tokenizer also damages Akan: `Ɔkwan bɛn` becomes `['kwan','b','n']`. The ɛ/ɔ letters break words into short ASCII fragments that match easily, which *inflates* Aka_Gha ROUGE.

**Evidence 2 (starter notebook):** `data/multilingual_data/multilingual_health_qa_starter_notebook.ipynb`, cell 13, uses `WhitespaceTokenizer.tokenize = str(text).strip().split()` with `use_stemmer=False`. That makes it case-sensitive, with punctuation left attached. The `whitespace` option copies this exactly.

**Evidence 3 (legacy numbers):** I rescored EXP-004's per-row file (`docs/competition_report/data/bge_retrieval_heldout.csv`, 1,491 rows) under all four tokenizers. The published 0.5604 / 0.4914 / **0.3892** reproduces *exactly* only under `default_stem`, which is rouge-score's default tokenizer plus the Porter stemmer. That was the old `score_rouge`. So [[EXP-004-bge-m3-retrieval-heldout]] was **not** scored with the whitespace tokenizer as it claims, and Amh_Eth's 0.016 is a tokenizer artifact. Only Latin letters and digits inside Amharic answers were ever scored. 18.8% of Val's Amh_Eth answers contain any.

ROUGE-only partial total, BGE-M3 predictions (EXP-004 file, scored against `data/raw/Train.csv` outputs for the same IDs):

| subset | n | whitespace (starter) | unicode | default | default_stem (EXP-004) |
|---|--:|--:|--:|--:|--:|
| Aka_Gha | 223 | 0.1706 | 0.1928 | 0.2272 | 0.2281 |
| Amh_Eth | 92 | **0.1151** | 0.1299 | 0.0121 | **0.0121** |
| Eng_Eth | 196 | 0.4379 | 0.4548 | 0.4548 | 0.4599 |
| Eng_Gha | 222 | 0.1798 | 0.2083 | 0.2083 | 0.2234 |
| Eng_Ken | 104 | 0.5821 | 0.5886 | 0.5886 | 0.5912 |
| Eng_Uga | 381 | 0.5470 | 0.5589 | 0.5589 | 0.5622 |
| Lug_Uga | 169 | 0.2795 | 0.3019 | 0.3019 | 0.3019 |
| Swa_Ken | 104 | 0.5881 | 0.5943 | 0.5942 | 0.5942 |
| **ALL** | 1491 | **0.3700** | 0.3872 | 0.3851 | **0.3892** |

Takeaways:
- On Amh_Eth, BGE-M3 R1/RL go from 0.016/0.016 to **0.162/0.149** under whitespace. Amharic is weak, but it is not at 0.
- Whitespace scores are 0.005–0.03 lower on every Latin-script subset than `default`/`unicode`. The cause is case sensitivity and punctuation that stays attached ("test." ≠ "test"). The effect is largest on Aka_Gha and Eng_Gha, where answers are long and varied. On pure-ASCII text, `unicode` and `default` without stemming are identical.
- The leaderboard number that EXP-004 implies under the starter tokenizer is **0.3700**, not 0.3892.

**Which one the grader uses (my belief):** `whitespace`, with moderate confidence. It is the host's own starter code, the organisers included a comment calling it "safe for non-English scripts", and a grader built on the default tokenizer would give every Amharic answer 0, which a host running an Amharic track would have noticed. None of this proves what the server runs, so [[H-013-whitespace-tokenizer-matches-grader]] stays open until one submission's per-component LB scores are compared with local ones. For that comparison, report `--compare-tokenizers` and see which column matches. **Use `whitespace` for every decision** ([[D-001-metric-weights-and-tokenizer]]), and never compare numbers scored with different tokenizers.

Practical consequence for generation: because whitespace ROUGE is case- and punctuation-sensitive, output casing and punctuation style that match the references are worth real points.

## Judge design and caveats
- `judge_fn: Callable[[list[str]], list[str]]` maps prompts to raw completions. It can be a fake in tests, `VLLMJudge` or `TransformersJudge` on molab, or `None` for no judge. The backends import torch/vllm and download weights only on their first call, so the module imports and runs on CPU without them.
- The rubric (`JUDGE_RUBRIC` in `judge.py`) scores accuracy, completeness and language appropriateness (wrong language = 1), each 1–5, plus an `overall` 1–5. The judge is asked for JSON. The parser uses `overall` if present, otherwise the mean of the three criteria, otherwise "Score: N" / "N/5" / a single bare digit. Out-of-range or ambiguous output becomes **NaN and is counted** as a parse failure. It is no longer silently turned into 3. Normalization is `(s−1)/4`.
- Closed-API model ids are refused. The old judge defaults, Aya-Expanse (rejected in [[D-002-reject-aya-expanse]]) and Phi-3.5 (weak on Amharic/Akan), were replaced.
- **Suggested molab judge: `google/gemma-3-27b-it`** in bf16 via vLLM (it fits in the 96 GB RTX Pro 6000). It has broad multilingual coverage including Amharic and Swahili, open weights, and is not one of our generator candidates, which avoids self-preference. Alternative: `Qwen/Qwen2.5-32B-Instruct`. Avoid judging with the same model family that generated the answers.
- **Caveats:** Zindi's judge model and prompt are undisclosed, so the stand-in's absolute level will not match the leaderboard's 0.26 term. F-003's 0.7379 for BGE-M3 retrieval came from the host judge, not ours. Use the judge only to *rank* candidates, and preferably calibrate it once against a leaderboard submission. It adds noise: judge 60–100 rows per subset and treat judge deltas under about 0.02 as noise. It is also slow, so sample with `--judge-max-per-subset` during iteration.

## Sanity check 2: CPU char-n-gram TF-IDF retrieval on the current split
This checks the current [[D-005-held-out-protocol]] split with no leakage. For each `data/processed/held_out.csv` question (2,088 rows), I took the answer of the nearest question in `work_train.csv` within the same subset, using TF-IDF over char 3–5-grams in pure Python. That took about 2.5 minutes, and the script was scratch and not committed. Scored with `whitespace`, no judge:

| subset | n | R1 | RL | rouge_only |
|---|--:|--:|--:|--:|
| Aka_Gha | 312 | 0.3061 | 0.1839 | 0.1813 |
| Amh_Eth | 129 | 0.1291 | 0.1201 | 0.0922 |
| Eng_Eth | 274 | 0.5975 | 0.5863 | 0.4380 |
| Eng_Gha | 311 | 0.2539 | 0.1682 | 0.1562 |
| Eng_Ken | 146 | 0.5752 | 0.5334 | 0.4102 |
| Eng_Uga | 534 | 0.5243 | 0.4782 | 0.3709 |
| Lug_Uga | 237 | 0.4848 | 0.4586 | 0.3491 |
| Swa_Ken | 145 | 0.6019 | 0.5706 | 0.4338 |
| **ALL** | 2088 | 0.4411 | 0.3882 | **0.3068** |

Under the other tokenizers, ALL is 0.3251 (unicode), 0.3238 (default) and 0.3284 (default_stem). This is the first whitespace-scored retrieval number on the current split. BGE-M3 still needs a re-run on this split (GPU/molab) to become the loop's baseline.

## Links
- [[00_INDEX]]
- [[F-001-competition-metric]]
- [[H-013-whitespace-tokenizer-matches-grader]]
- [[D-001-metric-weights-and-tokenizer]]
- [[D-005-held-out-protocol]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[FND-001-retrieval-strength-by-subset]]
- [[F-003-reference-approach]]
