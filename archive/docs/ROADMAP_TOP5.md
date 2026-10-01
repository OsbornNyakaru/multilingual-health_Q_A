# Roadmap to Top-5 — Afro-Health-QA (Practice Submissions, Autoresearch-Driven)

> Zindi *Multilingual Health QA in Low-Resource African Languages*. The competition closed
> 2026-06-21, but Zindi still **scores post-deadline submissions** (you see the real
> ROUGE-1 / ROUGE-L / LLM-Judge breakdown; they don't rank or win prizes). Goal: practice,
> submit real entries, and climb toward top-5 — run through the repo's **karpathy/autoresearch**
> loop, with **Colab Pro (A100)** for fine-tuning and a **local retrieval submission first**.

---

## The decisive finding (triangulated by a 15-agent recon)

**The entire pipeline is already built** — eval harness, data splits, BGE-M3 retrieval, QLoRA
training (`run.py` + 3 model configs), inference (`generate / rerank / postprocess / ensemble`),
submission writer/validator, and the autoresearch loop. **The gap to a top result is execution +
compute, not missing code.**

## Measured numbers (the spine — all on the 1,491-row held-out set)

| Subset | n | R1 | RL | mean cos | Verdict |
|---|---:|---:|---:|---:|---|
| Eng_Ken | 104 | 0.812 | 0.786 | 0.90 | retrieval wins |
| Swa_Ken | 104 | 0.816 | 0.790 | 0.90 | retrieval wins |
| Eng_Uga | 381 | 0.778 | 0.742 | 0.92 | retrieval wins |
| Eng_Eth | 196 | 0.636 | 0.607 | 0.93 | retrieval wins (35.7% verbatim dupes) |
| Lug_Uga | 169 | 0.433 | 0.383 | 0.85 | generation candidate |
| Aka_Gha | 223 | 0.395 | 0.221 | 0.82 | generation needed |
| Eng_Gha | 222 | 0.372 | 0.232 | 0.86 | generation needed |
| Amh_Eth | 92 | 0.016 | 0.016 | 0.81 | **hard floor — generation only** |
| **ALL** | **1491** | **0.560** | **0.491** | 0.88 | ROUGE-only = **0.389** |

- **BGE-M3 retrieval matches/edges 11th place** (Koleshjr: R1 0.5548 / RL 0.4823 / combined 0.576).
- **Router projection (measured arithmetic):** routing weak subsets to a generator at R1≈0.55,
  leaving strong subsets on retrieval, lifts weighted held-out **R1 0.560 → 0.643 (+0.083)** with a
  monotonic safety guarantee. Per-query confidence threshold **T≈0.85** on top-1 cosine.
- **Truncation is NOT a free win (measured):** truncating *retrieval* predictions to gold-median lost
  **−9.8%** (7/8 subsets worse). Length control belongs to **generation only**.
- **Reranker** (`BAAI/bge-reranker-v2-m3` over top-10): est. **+0.04–0.06 R1**.
- **Top-5 vs 11th (~0.04)** is *compounding* (routing + base model + native prompts + augmentation +
  decoding + small ensemble), **not one secret**. (Real top-5 writeups aren't public yet — synthesized.)

## Compute reality (decisive)

Local machine: **no GPU, ~8 GB RAM, ~7 GB free disk** → can't fine-tune or run 7-9B inference; but
**BGE-M3 retrieval runs locally on CPU (cached, proven).** So **Track 1 is local; Track 2 is Colab A100.**

---

## Track 1 — Local retrieval submission TODAY (no GPU; "lock the floor")

`scripts/retrieval_test_submission.py` (NEW), reusing `retrieval_baseline_bge.py` internals +
`submission/format.py` writer:

1. Build the **full labelled pool** = `train_core` + `held_out` + `Val` ≈ **36,499** rows (test is
   disjoint, so this is not leakage; larger pool ⇒ better neighbours).
2. Encode pool with **BAAI/bge-m3** (cached); encode `Test.csv` (2,618 rows); nearest neighbour
   **within the same `subset`**; prediction = neighbour's `output`.
3. Write 4-col `ID,TargetRLF1,TargetR1F1,TargetLLM` (targets identical); validate with `validate.py`.
4. **Upload to Zindi → first real practice score (~0.57).**

*Optional same-day local experiments (CPU, one lever each, logged):* swap retriever to
`Davlan/afro-xlmr-base` for weak subsets; `bge-reranker-v2-m3` over top-10. Keep ≥ +0.003 held-out gains.

## Track 2 — Colab Pro A100 fine-tune (autoresearch loop)

- **Data builder** `scripts/build_rag_dataset.py` (NEW) → `train_rag_enriched.jsonl`: k=3 same-subset
  neighbours per row (cached embeddings), **exclude self, never use held_out**; per-language system
  instructions + Q/A labels from `autoresearch_nlp/train.py`; 80/20 stratified split for early-stop.
- **Model:** Aya-Expanse-8B (wired: `configs/models/aya_expanse_8b.yaml` + `qlora_default.yaml`) via
  `training/run.py`. Llama-3.1-8B = ensemble upgrade only if Aya plateaus.
- **Hybrid router:** per-subset (strong→retrieve, weak→generate) + per-query T≈0.85. Projected
  held-out R1 ≈ **0.643**; can never score below pure retrieval.
- **Generation levers (generated output only):** per-subset length bounds (`tools/length_calibrate.py`),
  beam search, `bge-reranker` pick, preamble stripping (`postprocess.py`). One ablatable change each.

---

## What Andrej would do (the recommended path)

`autoresearch_nlp/` **is** karpathy/autoresearch mirrored for ROUGE-scored generation — use it as-is:

1. **Lock the floor first** (Track 1 submission) — full pipeline end-to-end before modelling.
2. **Verify training on a tiny overfit batch** (a few rows, loss→~0) *before* spending A100 hours.
3. **Frozen `prepare.py`, single editable `train.py`, ONE lever per commit**; append every run to
   `results.tsv` / `LOG.md`. Drive from the `H-01…H-10` backlog (H-01 floor → H-04 QLoRA → H-10
   native prompts → router).
4. **Become one with the data** — read per-subset failures (esp. Amharic/Akan); don't optimise an average.
5. **Measure on held-out, never estimate**; submit only ≥ +0.003 held-out gains; track held-out→LB
   transfer; **hedge a final best-public / best-held-out pair**.

## Honesty / process guardrails

- **Verify metric weights on the live page first** (`LESSONS.md` #1). Confirmed in code: 0.37 / 0.37 /
  0.26, AfroLM = 0.
- Keep **measured ROUGE** separate from the **estimated LLM-judge** (local Aya judge is an unvalidated
  proxy; Zindi's judge is undisclosed).
- Respect measured negatives: **no truncation on retrieval**; top-5 intel is synthesized, not leaked.

## Files

- **Reuse (don't rebuild):** `scripts/retrieval_baseline_bge.py`; `src/afro_health_qa/{evaluation/rouge.py,
  evaluation/combined.py, submission/format.py, submission/validate.py, training/run.py, models/prompts.py,
  inference/*}`; `configs/{models/aya_expanse_8b.yaml, training/qlora_default.yaml}`;
  `autoresearch_nlp/{prepare.py, train.py, program.md, tools/length_calibrate.py}`;
  `data/raw/{Test,Val,SampleSubmission}.csv`; `data/processed/{train_core,held_out}.csv`.
- **New:** `scripts/retrieval_test_submission.py`, `scripts/build_rag_dataset.py`,
  `notebooks/colab_rag_finetune.ipynb`.

## Verification

- **Track 1:** `python scripts/retrieval_test_submission.py` → 2,618-row 4-col CSV passes `validate.py`;
  upload returns ≈0.57.
- **Dataset:** 28k JSONL lines, each ≤3 retrieval IDs, **no self-match, no held-out IDs**, within max_seq_len.
- **Track 2:** a run prints the greppable `combined / rouge1_f1 / rougeL_f1 / per_subset` block; router run
  shows held-out R1 → ~0.64; each accepted change logged in `results.tsv` before its Zindi submission.
