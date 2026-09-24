---
type: fact
id: F-005
created: 2026-09-24
status: confirmed
links: ["[[F-001-competition-metric]]", "[[F-003-reference-approach]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[H-011-closed-pool-vs-generative-router]]", "[[H-012-rag-enriched-finetune]]", "[[H-013-whitespace-tokenizer-matches-grader]]", "[[00_INDEX]]"]
---
# F-005 Repo state audit (2026-09-24)

Subagent B audit. I opened every module in `src/afro_health_qa/`, every config, `scripts/`, `notebooks/`, `mac_local_execution/`, `autoresearch_nlp/` and `submissions/`. Nothing was modified, trained or committed. Git state: one commit (`00d444b`), and most real work is untracked (see [[Repo Layout]], [[Repo Recovery 2026-09-18]]).

## TL;DR
- **Three working pipelines exist, and none of them is the `src/` package:** (1) `autoresearch_nlp/prepare.py` is the correct harness (0.37/0.37/0.26 weights, whitespace ROUGE, 4-column submission); (2) `notebooks/molab_afro_health_qa.py` is a marimo port of it that adds a LoRA cell and an HF push; (3) the `scripts/retrieval_*.py` retrieval scripts.
- **`src/afro_health_qa/` is broken scaffold.** Training and generation import `afro_health_qa.models.{registry,qlora,prompts}`, but that subpackage **does not exist** (`ModuleNotFoundError`). The evaluation code uses the old wrong weights, 0.25/0.25/0.30/0.20.
- **The current best score was measured with the wrong ROUGE tokenizer.** The measurement is in the baseline section below.
- **Tests: none.** There is no `tests/` directory. `pytest` collects 0 items and exits with code 5.

## Implemented vs stub, by path

### `src/afro_health_qa/` (tracked, April scaffold, restored 2026-09-18)
| Module | State | Notes |
|---|---|---|
| `seeding.py` | implemented | seeds 5 RNGs; `pass` at l.51 is a benign `except` |
| `data/load.py` | implemented | maps real columns `input/output/subset` → internal `Question/Response/Language` |
| `data/split.py` | implemented | generic stratified split; configured by `configs/data.yaml` (85/10/5, seed 42), **not** [[D-005-held-out-protocol]] |
| `data/runtime.py`, `language_id.py`, `tokeniser_audit.py`, `length_bounds.py`, `augment.py` | implemented | `length_bounds.py` is an auto-generated table for exp001 |
| `data/sources/sunbird_salt.py`, `ghananlp_khaya.py`, `medmcqa_translated.py`, `synthetic_qa.py` | **stub** | `raise NotImplementedError` (+ TODO) |
| `data/sources/who_factsheets.py` | stub-ish | only passes through a hand-curated parquet that doesn't exist |
| `evaluation/rouge.py` | implemented, **wrong for this comp** | `RougeScorer(use_stemmer=True)` with the default tokenizer, which drops Ge'ez; contradicts [[F-001-competition-metric]] |
| `evaluation/combined.py` | implemented, **wrong weights** | `W_R1=0.25, W_RL=0.25, W_AFROLM_BS=0.30, W_JUDGE=0.20`; CLI reads `data/processed/val.csv` and the column `TargetBert` |
| `evaluation/local_eval.py` | implemented, mixed | has a whitespace tokenizer and `use_stemmer=False` (correct), but `combined_score` is also 0.25/0.25/0.30/0.20 |
| `evaluation/judge.py` | implemented | open-model judge with default `CohereForAI/aya-expanse-8b` (rejected, [[D-002-reject-aya-expanse]]); needs a GPU |
| `evaluation/afrolm_bertscore.py`, `per_language.py` | implemented | AfroLM has weight 0 on the leaderboard ([[D-001-metric-weights-and-tokenizer]]) |
| `inference/generate.py` | **broken** | HF `model.generate` loop, imports the missing `afro_health_qa.models.prompts` |
| `inference/rerank.py`, `ensemble.py`, `postprocess.py` | implemented | BM25/AfroLM reranker; `ensemble.combine(mode="language_routed")` routes model→language, but it is **not** a retrieval-vs-generation router |
| `training/run.py`, `trainer.py` | **broken** | import the missing `afro_health_qa.models.registry/qlora/prompts`; `--dry-run` only parses configs |
| `training/callbacks.py` | implemented | per-language callback logs only mean prediction length (a proxy) |
| `submission/format.py`, `validate.py` | implemented | read the schema from `SampleSubmission.csv` (4 columns); the default fallback is the old 5 columns with `TargetBert` |

### Other code
| Path | State | Notes |
|---|---|---|
| `autoresearch_nlp/prepare.py` | **implemented, correct** | frozen harness ([[Autoresearch Harness]]); `tools/synth_sanity.py` passed on CPU today (perfect=0.7400, 4 columns validated) |
| `autoresearch_nlp/train.py` | dry-run stub | `DRY_RUN = True`, `MODEL_ID = CohereLabs/aya-expanse-8b`; the stub output `submissions/exp001_zeroshot_baseline.csv` is all "information" ([[EXP-000-scaffold]]) |
| `autoresearch_nlp/results.tsv` | empty | header only |
| `notebooks/molab_afro_health_qa.py` | implemented, never run on GPU | modes `dry_run`/`zero_shot`/`few_shot`; weights 0.37/0.37/0.26, whitespace ROUGE, judge=0 locally; checkpointing; optional LoRA cell (PEFT, r=16, α=2r, bf16) and HF Hub push. No retrieval mode, no RAG, no router, no vLLM ([[Molab Notebook Plan]]) |
| `scripts/retrieval_baseline_bge.py` | implemented, **stale** | BGE-M3 held-out retrieval → `docs/competition_report/data/bge_retrieval_scores.json`; scores with `src` `score_rouge` (default tokenizer), reads the missing `data/processed/train_core.csv`, hard-coded Windows `CACHE_DIR` |
| `scripts/retrieval_test_submission.py` | implemented, stale | produced [[EXP-006-e5-small-retrieval-test-submission]]; same Windows cache path |
| `scripts/build_rag_dataset.py` | **partial** | see the gap table |
| `scripts/run_baseline.sh` / `make baseline` | **broken** | runs `notebooks/02_baseline_zeroshot.ipynb`, which doesn't exist |
| `scripts/verify_reproducibility.py` | broken | `--tiny` needs `tests/fixtures/tiny_predictions.csv` (missing) |
| `scripts/select_final.py` | implemented | reads submission JSON sidecars |
| `scripts/run_exp001.py`, `local_multilingual_qa.py`, `notebooks/exp00*_*`, `uploaded_nb_code.py`, `Copy_of_notebook*.ipynb` | historical | May-era Kaggle/Colab code ([[EXP-001-aya-expanse-oom]], [[EXP-002-afriquellama-fewshot-colab]]) |
| `scripts/build_report_*.py` | implemented | competition-report PDF/charts only |
| `mac_local_execution/` | README only | describes an MLX/AfriqueLlama pipeline, but no code is present |
| `models/`, `*.safetensors` | none | no adapters or checkpoints exist anywhere |
| `submissions/` | 4 files + 1 sidecar | `final_checkpoint.csv`, `submission_ready.csv`, `submission_20260512_0241.xlsx` (not a valid format), `20260624_095953_e5-small_retrieval_test.csv(+.json)`; no local or LB score recorded for any of them |

Docs and files referenced by README/Makefile but missing: `docs/SCORING.md`, `docs/DECISIONS.md`, `docs/KNOWN_ISSUES.md`, `notebooks/99_final_submission.ipynb`, `tests/`.

## Config groups (`configs/`)
These are plain YAML loaded with `yaml.safe_load`. Nothing imports Hydra, although `hydra-core==1.3.2` is pinned and the README calls the configs "Hydra-composable".

| Group | Options |
|---|---|
| root | `base.yaml`, `data.yaml` |
| `models/` | `aya_expanse_8b`, `gemma2_9b` (**untracked**: `.gitignore` rule `models/` also matches `configs/models/`, so GitHub lacks them while `training/*.yaml` references them) |
| `training/` | `qlora_default` (r32 α64 drop 0.05, 3 ep, lr 2e-4), `qlora_extended` (4 ep), `qlora_high_rank` (r64 α128, lr 1.5e-4) — all on `aya_expanse_8b` |
| `decoding/` | `greedy`, `beam5`, `beam_rerank` (8 beams / 4 groups, AfroLM rerank, BM25 pool) |

Inconsistencies:
- `base.yaml` `scoring_weights` 0.25/0.25/0.30 AfroLM/0.20 vs the real **0.37/0.37/0.26** ([[F-001-competition-metric]]). The same wrong weights are hard-coded in `src/.../evaluation/combined.py` and `local_eval.py`.
- `base.yaml` columns are `Question/Language/Response` plus a `TargetBert` target. The real columns are `input/output/subset` with 4 submission columns ([[F-002-data-shape]]).
- `base.yaml` `languages` = lug/swa/aka/amh, which omits English (~56% of Test). Its `seed: 42` differs from the harness seed 1234.
- `data.yaml` split is 85/10/5 on `Language` with seed 42. The protocol is 7% held-out, seed 1234 ([[D-005-held-out-protocol]]).
- Every training config targets Aya-Expanse, which was rejected ([[D-002-reject-aya-expanse]]). `data.yaml` `synthetic_qa.generator_model` is also Aya.
- No config matches the reference LoRA recipe (r64, α64, dropout 0.5, 3 ep, lr 2e-4, bs 4; [[F-003-reference-approach]]).
- `pyproject.toml` `requires-python >=3.11` (loosened locally) with torch 2.4.1 pins. These pins can't install on the Mac's Python 3.14; they target the GPU box.

## Current baseline score
From the experiment notes only. Every score was measured with the judge term = 0 (ROUGE-only).
- Current best: [[EXP-004-bge-m3-retrieval-heldout]], BGE-M3 retrieval, 1,491-row older held-out: R1 0.5604 / RL 0.4914 / ROUGE-only 0.3892 (`docs/competition_report/data/bge_retrieval_scores.json`).
- Current 2,088-row split: token-overlap retrieval only, ROUGE-only 0.279 ([[EXP-003-token-overlap-retrieval]]).
- No generative run has ever been scored. No public LB score is recorded anywhere.

**New measurement, 2026-09-24 (this audit, CPU).** I re-scored the saved per-row predictions `docs/competition_report/data/bge_retrieval_heldout.csv` with `rouge-score`:

| tokenizer | R1 | RL | ROUGE-only (0.37·R1+0.37·RL) | Amh_Eth R1 |
|---|---:|---:|---:|---:|
| default + `use_stemmer=True` (what `src/.../rouge.py` does) | 0.5604 | 0.4914 | 0.3892 | 0.0163 |
| whitespace, no stemmer (F-001 / `prepare.py`) | 0.5237 | 0.4764 | 0.3700 | 0.1620 |

EXP-004's numbers reproduce exactly with the **default** tokenizer, not the "LB-matching" whitespace one its JSON claims. Under the harness tokenizer the same predictions score **ROUGE-only 0.3700**. Amharic goes up (0.016 → 0.162) and Aka_Gha / Eng_Gha go down. This explains the "suspicious" Amharic number in EXP-004 and bears directly on [[H-013-whitespace-tokenizer-matches-grader]]. The JSON's `weights: 0.37` also doesn't match the current `combined.py` (0.25), so the June version of `combined.py` was probably lost in the 2026-09-18 restore (inference).

## Gap analysis vs the reference approach ([[F-003-reference-approach]])
| Component | Status | Where / what's missing |
|---|---|---|
| BGE-M3 retrieval baseline | **present (stale)** | `scripts/retrieval_baseline_bge.py`, `retrieval_test_submission.py`; needs a repo-relative cache path, the current `work_train.csv`/`held_out.csv` split, and whitespace ROUGE via `prepare.py` |
| RAG context generation | **partial** | `scripts/build_rag_dataset.py`: k=3 same-subset neighbours, excludes self, never touches held-out, prompts match `autoresearch_nlp/train.py`, writes prompt/completion JSONL. Gaps: builds **train-side only**, with no builder for held-out/test prompts (the reference uses train for val and train+val for test); depends on the missing `train_core.csv` and a Windows-only embedding cache; only `data/processed/val_rag_enriched.jsonl` (1,416 rows) is on disk, and `train_rag_enriched.jsonl` is missing; RAG was never wired into the molab notebook |
| LoRA fine-tune script | **partial** | molab notebook cell 10 (plain PEFT, bf16, r=16 default, dropout 0.05, completion-only labels) works on paper but was never run. It trains on plain prompts, not RAG JSONL. `src/.../training/run.py` is broken (missing `models` package). No Unsloth, and nothing configured for `Sunbird/Sunflower-32B` |
| vLLM inference | **missing** | no `vllm` anywhere; all generation is HF `model.generate` (notebook, `train.py`, `src/.../generate.py`) |
| Per-subset router (closed-pool vs generative) | **missing** | only the idea exists ([[H-011-closed-pool-vs-generative-router]], `scripts/build_report_pdf.py` text). `inference/ensemble.py` `language_routed` routes between models, not retrieval-vs-LLM |
| HF Hub artifact push | **partial** | molab notebook: `upload_file` of submission and `upload_folder` of adapter to `HF_RUNS_REPO` (dataset repo), gated by a switch, never exercised; data pull via `HF_DATA_REPO` |
| molab notebooks | **partial** | one notebook `notebooks/molab_afro_health_qa.py` (no GPU run recorded in `results.tsv` or the vault); no retrieval, RAG or router cells |

## Next actions (prioritised)
1. **Fix the scoring source of truth.** Retire or repair `src/.../evaluation/rouge.py` + `combined.py` (whitespace tokenizer, 0.37/0.37/0.26) so no script can import the wrong metric. Settle [[H-013-whitespace-tokenizer-matches-grader]]. The metric-replica note is pending (Subagent C).
2. **Re-baseline retrieval on the current 2,088-row split** with `prepare.py` scoring, a repo-relative embedding cache and `work_train` as the pool. That number becomes the loop's floor ([[D-003-retrieval-first-baseline]]). Log it as a new EXP.
3. **Router on held-out:** per-subset, and optionally per-row by retrieval cosine threshold, choosing retrieval vs generation → [[H-011-closed-pool-vs-generative-router]]. The retrieval half can be tuned on CPU now.
4. **Finish RAG data:** make `build_rag_dataset.py` read `work_train.csv`, use a portable cache, and add held-out/test prompt builders (held-out retrieves from work_train; test from train+val). Wire into the molab LoRA cell with the reference hyper-parameters → [[H-012-rag-enriched-finetune]]; base model per [[D-004-base-model-for-molab]].
5. **Add vLLM inference** (optionally LoRA) to the molab notebook for held-out and test throughput.
6. Housekeeping: add a minimal `tests/` (metric replica, submission validation, split determinism); fix `.gitignore` so `configs/models/` is tracked; drop or fix `make baseline` and the stale Aya configs; commit untracked work (pending the owner's OK).

## Links
- [[00_INDEX]]
- [[F-001-competition-metric]]
- [[F-002-data-shape]]
- [[F-003-reference-approach]]
- [[D-001-metric-weights-and-tokenizer]]
- [[D-003-retrieval-first-baseline]]
- [[D-004-base-model-for-molab]]
- [[D-005-held-out-protocol]]
- [[EXP-003-token-overlap-retrieval]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[EXP-006-e5-small-retrieval-test-submission]]
- [[H-011-closed-pool-vs-generative-router]]
- [[H-012-rag-enriched-finetune]]
- [[H-013-whitespace-tokenizer-matches-grader]]
- Legacy: [[Repo Layout]], [[Autoresearch Harness]], [[Molab Notebook Plan]], [[Repo Recovery 2026-09-18]]
