# 02 — Reproduce the 11th-Place Approach

Run after `00_orchestrator.md` has finished (vault exists, metric replica
works). Read `vault/00_INDEX.md`, `vault/facts/F-003-reference-approach.md`,
and `vault/facts/metric-replica.md` before starting.

## Goal

Get our own numbers, on our own held-out validation split, for the two
reference points from the 11th-place solution — then we know exactly what
"matching 11th place" means locally before trying to beat it.

Reference repo (read it, adapt it — don't just copy-paste scripts into our
tree): `https://github.com/koleshjr/multilingual_qa_training`. Key files:
`generate_rag_context_datasets.py`, `main.py`, `vllm_inference.py`,
`config.py`, `utils.py`.

Implement everything **inside our package** `src/afro_health_qa/` with
settings in **our Hydra `configs/`**, not as loose scripts, so later
experiments are one config change away. You write and unit-test code
locally. GPU steps are run through the molab notebooks from prompt 03,
so if those don't exist yet, build `notebooks/00_setup_data.py` and
`01_retrieval.py` first.

**Use the fixed eval protocol from `04_experiment_loop.md`**: split
`Val.csv` once into `val_tune` / `val_holdout` (stratified by subset,
fixed seed), and report every score below on `val_holdout`.

---

## Step 1 — Retrieval-only baseline (cheap, do first)

Subagent task:
1. Embed all `input` questions with `BAAI/bge-m3` (sentence-transformers).
   Cache embeddings under `.cache/embeddings/`, keyed by model name and
   data-file hash, so reruns are free.
2. For each validation row, retrieve the single nearest **train** question
   (cosine similarity) and use its `output` verbatim as the prediction.
3. Score with our metric replica, overall and per `subset`.
4. Also run a TF-IDF nearest-neighbour version as a sanity baseline.

Write `vault/experiments/EXP-001-retrieval-bge-m3-k1.md` (and one for
TF-IDF) with: config, overall + per-subset scores, and a comparison line
against the 11th-place leaderboard numbers. Our local numbers won't match
theirs exactly (different split, judge stand-in) — that's expected; what
matters is we now have a comparable local floor.

**Record per-subset scores prominently** — if some subsets score very high
on retrieval-only (e.g. near-duplicate paraphrases), note them as
candidates for the "closed-pool" bucket in a future router (see prompt 04).

---

## Step 2 — Build RAG-enriched datasets

Subagent task: port `generate_rag_context_datasets.py` logic into
`src/afro_health_qa/rag.py` (or similar) with a matching config group.
Produce, for k ∈ {1, 2, 3}:

- **Train**: retrieve k neighbours from train, **excluding the row itself**.
- **Val**: retrieve k neighbours from **train only** — no val-to-val
  leakage, this is what keeps validation honest.
- **Test**: retrieve k neighbours from **train + val**.

Prompt format (keep identical to the reference so results are comparable):

```
Given a health-related question in a low-resource African language, provide a fluent, accurate, safe, and contextually appropriate answer in the same language as the question.

Retrieved examples may contain useful wording or facts. Use them only when relevant.

Example 1:
Question: ...
Answer: ...

Question: {current question}

Answer:
```

Save to `data/rag_context/{Train,Val,Test}_rag_bge_m3_k{k}.csv` with
columns `retrieved_context`, `retrieved_neighbor_ids`,
`retrieved_neighbor_subsets`, `formatted_prompt`, `rag_context_source`.
Print token-length stats of `formatted_prompt` — flag anything near the
model's max sequence length.

---

## Step 3 — RAG-enriched LoRA fine-tune

Subagent task: implement training via **Unsloth + TRL `SFTTrainer` + LoRA**
in our package. Start with the reference config exactly:

| setting | value |
|---|---|
| base model | `Sunbird/Sunflower-32B` (Qwen-3 family) |
| max seq len | 4096 |
| LoRA r / alpha / dropout | 64 / 64 / 0.5 |
| epochs | 3 |
| lr | 2e-4 |
| batch size | 4 |
| scheduler | linear |
| optimizer | adamw_8bit |
| k (retrieved examples) | 3 |

Chat format: `system: "You are a helpful assistant!"`,
`user: formatted_prompt`, `assistant: output`. Use
`train_on_responses_only` so loss is only on the answer.

**Critical difference from the reference repo**: their `main.py` merges
train+val for the final model. For *this* step, train on **train only** and
evaluate on **val**, so the score is honest. Only merge train+val later,
for a final submission model, once the recipe is locked (log that as a
separate decision note).

Set every seed (Python, NumPy, torch, transformers, vLLM). Training runs
on molab's RTX Pro 6000 Blackwell (96 GB) via `notebooks/02_train.py`
(see prompt 03). 32B LoRA in bf16 should fit; if it doesn't, fall back to
4-bit loading and record that in the vault. Push the adapter to a private
HF Hub repo at every checkpoint, because molab storage isn't durable.

**Iteration speed tip**: a 32B, 3-epoch run is slow. Validate the whole
pipeline end to end first on a smaller model and a 10% sample. Then run
the full reference config once to get the true comparison point.

---

## Step 4 — vLLM inference and scoring

Subagent task: port `vllm_inference.py` logic. Load base model + LoRA
adapter with vLLM, **greedy decoding** (the reference's active setting),
generate on val, clean outputs (strip template artifacts, trailing
whitespace), score with our metric replica, overall and per subset.

Write `vault/experiments/EXP-00X-rag-ft-sunflower32b-k3.md` with the full
config, scores, and a direct comparison against EXP-001 (retrieval-only).
Update `vault/00_INDEX.md` "Current best".

Then generate a test submission CSV: columns `ID,TargetRLF1,TargetR1F1,TargetLLM`,
same cleaned answer in all three. Save under `submissions/` with a
metadata sidecar (the repo's existing convention).

---

## Done when

- [ ] EXP notes exist for retrieval-only (BGE-M3 + TF-IDF) and RAG-FT k=3.
- [ ] Per-subset score table exists in each note.
- [ ] `vault/00_INDEX.md` "Current best" points to the winning experiment.
- [ ] A valid submission CSV exists and passes a column/row-count check
      against `SampleSubmission.csv`.
