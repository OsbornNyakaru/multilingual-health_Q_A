# Step-by-Step Guide: Running exp001 on Kaggle

## Why Kaggle Over Colab Free?

| Feature | Kaggle Free | Colab Free |
|---------|-------------|------------|
| GPU hours/week | **30 hours** | ~6h before disconnect |
| RAM | **30 GB** | 12.7 GB |
| Session stability | **High** | Frequently disconnects |
| VRAM (T4) | 16 GB | 16 GB |
| Max session length | **12 hours** | ~6-8 hours |

**Verdict**: Kaggle is strongly recommended. The full pipeline (Val+Test generation) takes ~8-12 hours on T4 — Kaggle handles this in one session.

---

## Step 0: Setup on Kaggle

### 0a. Create a Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Name it: `afro-health-qa`
3. Upload these 4 files from `data/raw/`:
   - `Train.csv`
   - `Val.csv`
   - `Test.csv`
   - `SampleSubmission.csv`
4. Click **Create** — the dataset path will be `/kaggle/input/afro-health-qa/`

### 0b. Create a Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **Settings** (gear icon):
   - **Accelerator**: `GPU T4 x2` (or `GPU T4 x1`)
   - **Internet**: `ON` (required for downloading models from HuggingFace)
   - **Persistence**: `Files only`
3. Click **Add Data** → search for your `afro-health-qa` dataset → **Add**

### 0c. Install Dependencies (first cell)
```python
!pip install -q torch transformers accelerate bitsandbytes rouge-score \
    scikit-learn pandas numpy tqdm sentencepiece protobuf fast-langdetect
```

---

## Step 1: Data Audit (~30 seconds)

**What happens**: Loads all 4 CSV files, detects column names (`ID`, `input`, `output`, `subset`), maps language codes, checks for nulls, Amharic script issues, and overlaps.

**Key outputs**:
- Column mapping: `id_col=ID, question_col=input, lang_col=subset, answer_col=output`
- SampleSubmission has **4 columns** (not 5): `ID, TargetRLF1, TargetR1F1, TargetLLM`
- 8 language subsets including English variants

**What to watch for**: Any `[FLAG]` messages about distribution gaps or missing data.

---

## Step 2: Length Bounds (~5 seconds)

**What happens**: Computes p10/p90 answer lengths per language from Train, compares with Val, builds `LENGTH_BOUNDS` dict.

**Key outputs**: A `LENGTH_BOUNDS` dictionary mapping each canonical language to `{min_new_tokens, max_new_tokens}`.

**What to watch for**: Notes about Train/Val p90 divergence > 20 tokens.

---

## Step 3: Held-Out Split (~10 seconds)

**What happens**: Carves 5% of Train as `held_out.csv` (sealed — never touch until final week). Remaining 95% is `train_core.csv`.

**Key outputs**: `train_core.csv` and `held_out.csv` saved to working directory.

---

## Step 4: Evaluation Calibration (~5-10 minutes)

**What happens**: 
1. Loads AfroLM encoder model for BertScore
2. Runs two sanity checks on 10 Val rows:
   - **Check A**: prediction = reference → score should be ≥ 0.95
   - **Check B**: wrong-language predictions → score should be ≤ 0.20
3. Unloads AfroLM to free GPU memory

**Note**: LLM Judge is SKIPPED on free GPU. We use the 3-metric proxy formula instead: `R1×0.3125 + RL×0.3125 + AfroLM×0.375`.

**What to watch for**: Both calibration checks must pass (with proxy scoring, thresholds are slightly relaxed).

---

## Step 5: Generate Predictions (~8-12 hours total)

This is the longest step. Budget your time.

### 5a. Load Model (~5-10 minutes)
Attempts to load `CohereForAI/aya-expanse-8b` in 4-bit quantization. Falls back through the model chain if OOM.

### 5b. Generate Val Predictions (~5-7 hours)
- 6,686 rows with beam=5 decoding
- **Checkpoints every 50 rows** — if session disconnects, re-run and it resumes
- Saves to `predictions_val_partial.csv`

### 5f. Score on Val (~15-20 minutes)
- Reloads AfroLM, computes ROUGE + BertScore
- **Go/No-Go gate**: combined score must be ≥ 0.50

### 5g. Generate Test Predictions (~2-3 hours)
- 2,618 rows with beam=5
- Same checkpointing system

**⏱️ Total time estimate**: 8-12 hours on T4. Well within Kaggle's 12-hour limit.

---

## Step 6: Build Submission (~1 minute)

**What happens**: 
1. Reads SampleSubmission format (4 columns)
2. Builds submission DataFrame — all target columns identical
3. Runs 14-point validation checklist
4. Saves as `submissions/YYYY-MM-DD_HHMM_exp001_aya_zeroshot.csv`

**What to watch for**: All 14 checks must show `[PASS]`.

---

## Step 7: Experiment Log (~instant)

Writes the experiment entry to `experiments/LOG.md`.

---

## Step 8: Diagnostics (~instant)

**What happens**:
1. Shows weakest sub-metric per language
2. Prints 3 sample predictions per language (read these carefully!)
3. Recommends exp002 action based on scores

---

## After the Run

1. **Download** the submission CSV from Kaggle's Output tab
2. **Upload** to Zindi: [Competition page](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge) → My Submissions → Upload
3. **Paste the public score** back so we can plan exp002

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Session disconnects mid-generation | Re-run — checkpointing resumes from last saved row |
| OOM loading Aya-8B | The fallback chain will try smaller models automatically |
| AfroLM fails to load | Falls back to `multilingual-e5-base` (scores will differ from Zindi) |
| Check A/B calibration fails | Check AfroLM loaded correctly; try restarting runtime |
| `bitsandbytes` install fails | Run `!pip install bitsandbytes==0.43.0` specifically |
| Internet disabled on Kaggle | Go to Settings → Enable Internet |

---

## Time Budget

| Step | Estimated Time |
|------|---------------|
| Steps 1-3 (data processing) | < 1 minute |
| Step 4 (calibration) | 5-10 minutes |
| Step 5 (val + test generation) | 8-12 hours |
| Steps 6-8 (submission + diagnostics) | < 5 minutes |
| **Total** | **~8-12 hours** |

This fits within Kaggle's 12-hour session limit and 30 hours/week GPU quota.
