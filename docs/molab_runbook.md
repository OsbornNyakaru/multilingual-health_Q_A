# molab runbook: afro health qa notebook

a step-by-step guide for osborn and vera to running `notebooks/molab_afro_health_qa.py` on molab.
one change per run. write down every result.

## 0. the golden rules

- **one change per run.** change a single setting, run, record, then decide.
- **new run name every run.** the notebook saves progress under the run name. if you reuse a name, it quietly reloads the old answers instead of generating new ones. the only exception: you are resuming a run that got cut off, with nothing else changed.
- **record before you move on.** molab deletes its files when the session ends. copy the scores into the notion log straight away.
- **keep a change only if it wins by ≥ 0.003** on the full 2,088-row held-out set. smaller gaps are noise (about ±0.005).
- **never click "generate test submission" until a full held-out run (rows = 0) says the setting is worth it.**

## 1. what the notebook does, in one breath

- it loads the competition data, splits off 2,088 training questions as a private exam ("held-out"), gets a model to answer them, and scores those answers the way zindi does.
- the score is `combined = 0.37 × rouge-1 + 0.37 × rouge-l`. the ai-judge part (0.26) is not measured here, so the best possible score on this screen is **0.74**, not 1.
- the bar to beat is roughly **0.37**, the score of simply copying the answer of the most similar training question (retrieval). it was measured on an older split; a re-run on this split is still pending.

## 2. the settings cell (section 1), line by line

| setting | what it means | what to put |
|---|---|---|
| run name | label for this run. names the progress file, the results line and the submission file | a new name every run (see naming below) |
| mode | `dry_run` = no model, fake answers, tests the plumbing. `zero_shot` = model answers cold. `few_shot` = model first sees a few example q&a pairs from the same language | follow the stage you're on |
| precision | `auto` loads the model at full quality (bf16) on molab's 96 gb gpu | always `auto`. use `4bit` only if you get an out-of-memory error |
| base model | which model writes the answers | follow the stage you're on |
| …or type any hf id | overrides the dropdown with any model on hugging face | leave blank unless testing a new model |
| lora adapter dir | fine-tuned add-on weights | blank until stage 8 |
| num_beams | 1 = fast, picks the likeliest word each step. 3–5 = tries several answers in parallel, slower | 1, except in the beams stage |
| inference batch size | questions answered at once | 32. drop to 16 or 8 on out-of-memory |
| few-shot examples per subset | how many example q&a pairs go in the prompt (few_shot only) | 2, except in the k stage |
| few-shot answer cap | example answers are cut to this many characters | 150, except in the cap stage |
| held-out rows to score | how many of the 2,088 questions to score. 0 = all | 200 smoke test, 500 screening, 0 for anything you'd keep |
| checkpoint every n batches | how often progress is saved, so a crash can resume | 10 |
| push to hf hub | uploads the submission at the end | off for now |

- rows 200 and 500 always pick the same questions (fixed random seed), so runs with the same row count are fair to compare.
- 500-row scores are rougher. use them only to rank models, then confirm on all rows (0).

## 3. naming runs

- pattern: `exp<number>_<model>_<mode>[_k<k>][_c<cap>][_b<beams>]_n<rows>`
- models: `afrique`, `qwen`, `gemma`, `llama`, `aya`. modes: `zs` (zero_shot), `fs` (few_shot). rows: `n200`, `n500`, `nall`.
- examples: `exp008_afrique_zs_n500`, `exp020_qwen_fs_k3_nall`, `exp025_qwen_fs_k2_b3_nall`
- dry runs don't get a number: `dry_01`, `dry_02`.
- numbers continue from the vault (exp000–exp006 exist). the next free number is **exp007**. always use the next free one, even if a run failed.

## 4. the models

| short name | hf id (type exactly) | notes |
|---|---|---|
| afrique | `McGill-NLP/AfriqueLlama-8B` | llama 3.1 adapted to african languages. open, no approval needed |
| qwen | `Qwen/Qwen2.5-7B-Instruct` | strong general model. open, no approval needed |
| gemma | `google/gemma-2-9b-it` | licence-gated. osborn's account already has access. needs `HF_TOKEN` set on molab |
| llama | `meta-llama/Llama-3.1-8B-Instruct` | licence-gated, **no access yet**. request it on the model's hf page, approval is manual. skip until approved |
| aya | `CohereLabs/aya-expanse-8b` | reference only. non-commercial licence and weak coverage of our languages, so it can't be a final pick (vault decision d-002) |

## 5. before every session

- open the notebook from github in molab (saved copy in your workspace, not the temporary preview).
- turn on the gpu (rtx pro 6000).
- set the environment variables: `HF_DATA_REPO=nyakaruosborn/afro-health-qa-data` and `HF_TOKEN=<read-only token>`. without the token, gemma and aya won't download.
- check these cells as they run:
  - **section 2 (environment):** gpu shows **rtx pro 6000, ~96 gb**. "none" means the gpu is off.
  - **section 3 (data):** "downloaded from hf dataset" and no missing files.
  - **section 4 (harness):** green "harness parity … **ok**". red mismatch = stop, the split changed.
  - **section 5 (prompting):** few-shot example table (only fills in few_shot mode).
  - **section 6 (model):** says which model and precision loaded.

## 6. how cells run

- marimo runs cells by itself. change a setting and everything that depends on it re-runs.
- the slow steps wait for a button:
  - **evaluate held-out** (section 8): answers and scores the held-out questions.
  - **generate test submission** (section 9): answers the 2,618 test questions and writes the zindi file.
  - **train lora adapter** (section 10): fine-tuning, stage 8 only.
- after changing any setting, click **evaluate held-out** again. the old result does not update by itself.
- if something looks stuck or stale: notebook menu → restart, then let it run from the top.

## 7. what to read after "evaluate held-out"

- top line: run name, mode, model, rows, seconds taken, **refusal rate** (share of answers like "i'm sorry, i can't"; should be near 0%).
- three numbers: **combined (no judge)**, rouge-1 f1, rouge-l f1.
- the per-subset table: one row per language/country (`Aka_Gha`, `Amh_Eth`, `Eng_Eth`, `Eng_Gha`, `Eng_Ken`, `Eng_Uga`, `Lug_Uga`, `Swa_Ken`). this is where you see which language drags the score down.
- "greppable report": open it and copy the whole block into notion. it has every number.
- "sample predictions": read 5–10. note if answers are in the wrong language, cut off, rambling, or refusals.
- the last cell appends a line to `results.tsv`. it disappears when molab shuts down, so notion is the real record.

## 8. the run plan, stage by stage

### stage 1: dry run (plumbing check, once per new notebook version)
- mode `dry_run`, rows `0`, run name `dry_01`, other settings default.
- click **evaluate held-out**.
- pass = the score table appears (scores will be near zero, every answer is the word "information") and the parity box is green.
- optional: click **generate test submission**; pass = green "submission written, 2,618 rows".

### stage 2: gpu smoke test
- `exp007_afrique_zs_n200`: mode `zero_shot`, model afrique, rows `200`.
- pass = gpu used, model loads, answers look like answers in the right language, refusal rate low.

### stage 3: zero-shot model screen (rows 500)
- one run per model, everything else default:
  - `exp008_afrique_zs_n500`
  - `exp009_qwen_zs_n500`
  - `exp010_gemma_zs_n500`
  - `exp011_llama_zs_n500` (only once access is approved)
  - `exp012_aya_zs_n500` (reference only)

### stage 4: few-shot model screen (rows 500, k = 2, cap = 150)
- same models, mode `few_shot`:
  - `exp013_afrique_fs_k2_n500`
  - `exp014_qwen_fs_k2_n500`
  - `exp015_gemma_fs_k2_n500`
  - `exp016_llama_fs_k2_n500`
  - `exp017_aya_fs_k2_n500`

### stage 5: confirm the top two on all rows
- pick the two best (model + mode) combos from stages 3–4. aya can't be a final pick.
- re-run each with rows `0`: e.g. `exp018_<model>_<mode>_nall`, `exp019_…_nall`.
- the higher one is the **current best**. from here on every run uses rows `0` and changes one thing from the current best.

### stage 6: tune few-shot (best model, mode few_shot, rows 0)
- k (examples): try 1, 3, 4 (2 is done). one run each.
- cap (example length): with the best k, try 80 and 300.
- after each run: if it beats the current best by ≥ 0.003 it becomes the new current best, otherwise drop it.

### stage 7: beams (current best, rows 0)
- num_beams 3, then 5. slower. if it runs out of memory, set batch size to 16.
- keep only if it wins by ≥ 0.003 and the extra time is acceptable.

### stage 8: fine-tuning (later)
- section 10, lora. only after stages 1–7. settings and plan to be agreed first; it takes 1–3 hours per epoch.
- output lands in `models/<run name>/`. put that path into **lora adapter dir**, then run a new held-out evaluation with a new run name.

### stage 9: test submission (only for a new current best)
- keep every setting and the run name exactly as in the winning held-out run.
- click **generate test submission**. the file is `submissions/<run name>.csv` with a `.csv.json` summary next to it.
- download both before the session ends (molab files don't survive), then upload the csv to zindi and record the public score.

## 9. molab limits

- idle for 90 minutes = shut down. any session = 12 hours max.
- a cut-off run resumes when you click evaluate again **with the same run name and settings**.
- everything in the session's files is temporary. notion and downloads are the record.

## 10. when things go wrong

- "no cuda gpu" → turn the gpu on, restart.
- "waiting for data" → env vars not set, or upload the 4 csvs in the sidebar.
- gated repo / 401 / 403 → `HF_TOKEN` missing, or no licence access for that model.
- out of memory → batch size 16 (then 8). for beams, lower the batch first.
- red parity mismatch → stop and tell osborn; scores can't be compared.
- anything else → copy the full error into notion and ask.
