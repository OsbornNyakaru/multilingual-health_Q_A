# %% [markdown]
# # AfroHealth QA — exp002 (Few-Shot Prompting)
# **Platform**: Google Colab Free (T4 GPU)
# **Purpose**: Improve zero-shot score (0.417) by injecting 2 training examples into the prompt.

# %% — Cell 1: Setup and Model Loading
import os
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──
from google.colab import drive
drive.mount("/content/drive")
RAW_DIR = Path("/content/drive/MyDrive/afro-health-qa/data/raw")
PROCESSED_DIR = Path("/content/working/data/processed")
SUBMISSIONS_DIR = Path("/content/working/submissions")

for d in [PROCESSED_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load Data ──
# We need Test.csv for submission and train_core.csv for few-shot examples
print("Loading data...")
test = pd.read_csv(RAW_DIR / "Test.csv", dtype=str)
sample = pd.read_csv(RAW_DIR / "SampleSubmission.csv", dtype=str)

# If train_core.csv is saved in your Drive, load it from there. 
# Otherwise, we load the raw Train.csv to sample examples.
try:
    train = pd.read_csv(PROCESSED_DIR / "train_core.csv", dtype=str)
except FileNotFoundError:
    print("train_core.csv not found in working dir, loading raw Train.csv from Drive...")
    train = pd.read_csv(RAW_DIR / "Train.csv", dtype=str)

id_col = "ID"
if "input" in test.columns:
    question_col = "input"
    lang_col = "subset"
    answer_col = "output" if "output" in train.columns else "Target"
else:
    question_col = "Question"
    lang_col = "Language"
    answer_col = "Target"

# ── Language Mapping ──
def canonical_prompt_language(lang_code): 
    return "eng" if lang_code == "eng" else lang_code

LANG_CODE_MAP = {}
for raw in sorted({str(v) for v in train[lang_col].unique()}):
    key = raw.strip(); lower = key.lower()
    if lower.startswith("swa") or "swahili" in lower: LANG_CODE_MAP[key] = "swa"
    elif lower.startswith("lug") or "luganda" in lower: LANG_CODE_MAP[key] = "lug"
    elif lower.startswith("aka") or "twi" in lower or "akan" in lower: LANG_CODE_MAP[key] = "aka"
    elif lower.startswith("amh") or "amharic" in lower: LANG_CODE_MAP[key] = "amh"
    elif lower.startswith("eng") or "english" in lower: LANG_CODE_MAP[key] = "eng"

# ── Extract Few-Shot Examples ──
print("Extracting few-shot examples from training data...")
FEW_SHOT_EXAMPLES = {}
for raw_lang in train[lang_col].unique():
    canon_lang = LANG_CODE_MAP[str(raw_lang)]
    # Filter for reasonable length examples (not too long to save context window)
    lang_data = train[train[lang_col] == raw_lang].dropna(subset=[answer_col])
    lang_data["ans_len"] = lang_data[answer_col].astype(str).str.len()
    # Pick 2 examples that are between 50 and 200 characters long (ideal concise answers)
    ideal_examples = lang_data[(lang_data["ans_len"] >= 50) & (lang_data["ans_len"] <= 300)]
    
    if len(ideal_examples) >= 2:
        sampled = ideal_examples.sample(2, random_state=42)
    else:
        sampled = lang_data.sample(min(2, len(lang_data)), random_state=42)
        
    FEW_SHOT_EXAMPLES[canon_lang] = [
        (row[question_col], row[answer_col]) for _, row in sampled.iterrows()
    ]

# ── Prompts ──
SYSTEM_INSTRUCTIONS = {
    "swa": "Wewe ni daktari mtaalamu. Jibu swali la afya hapa chini kwa Kiswahili. Toa jibu kamili na la moja kwa moja. Usiseme huwezi kujibu.",
    "lug": "Oli omusawo omukugu. Ddamu ekibuuzo kino eky'obulamu mu Luganda. Wa eddamu erijjuvu era ery'obwesigwa. Togamba nti toyinza kuddamu.",
    "aka": "Woyɛ dɔkota nimdeɛ. Bua afutuo asɛmmisa yi so wɔ Akan kasa mu. Ma mmuaeɛ a ɛyɛ pɛpɛɛpɛ. Nnka sɛ wontumi mmua so.",
    "amh": "እርስዎ ልምድ ያለው ሐኪም ነዎት። ይህን የጤና ጥያቄ በአማርኛ ይመልሱ። ሙሉ እና ቀጥተኛ መልስ ይስጡ። መመለስ አልችልም አይበሉ።",
    "eng": "You are an expert doctor. Answer the following health question in English. Give a complete and direct answer. Never refuse to answer.",
}

PROMPTS = {
    "swa": "Swali: {question}\nJibu:",
    "lug": "Ekibuuzo: {question}\nEky'okuddamu:",
    "aka": "Asɛmmisa: {question}\nMmuaeɛ:",
    "amh": "ጥያቄ: {question}\nመልስ:",
    "eng": "Question: {question}\nAnswer:",
}

ANSWER_MARKERS = {
    "swa": "Jibu:", "lug": "Eky'okuddamu:", "aka": "Mmuaeɛ:",
    "amh": "መልስ:", "eng": "Answer:",
}

REFUSAL_PATTERNS = [
    "i apologize", "i'm sorry", "i cannot", "i can't",
    "i don't have enough context", "i'm not able to",
    "i am not able to", "i do not have enough",
    "it appears to be in", "the text appears",
    "i'm unable to", "i am unable to",
    "as an ai", "as a language model",
    "i don't understand", "i do not understand",
]

def is_refusal(text):
    lower = text.lower().strip()
    if len(lower) < 5: return True
    return any(pat in lower for pat in REFUSAL_PATTERNS)

def strip_prompt_artefacts(text, lang):
    marker = ANSWER_MARKERS.get(lang, "")
    if marker and marker in text:
        text = text.split(marker, 1)[-1]
    text = text.strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text

def build_prompt(question, lang, tokenizer, force_answer=False):
    sys_msg = SYSTEM_INSTRUCTIONS[lang]
    
    # ── Inject Few-Shot Examples ──
    examples_text = ""
    for ex_q, ex_a in FEW_SHOT_EXAMPLES[lang]:
        examples_text += PROMPTS[lang].format(question=ex_q) + " " + str(ex_a).strip() + "\n\n"
        
    user_content = examples_text + PROMPTS[lang].format(question=question)

    if force_answer:
        force_suffix = {
            "swa": " Jibu moja kwa moja bila kusita:",
            "lug": " Ddamu ku biseera bino awatali kwebuuza:",
            "aka": " Bua ntɛm ara:",
            "amh": " አሁን በቀጥታ ይመልሱ:",
            "eng": " Answer directly now:",
        }
        user_content = user_content + force_suffix.get(lang, " Answer directly:")

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            merged = sys_msg + "\n\n" + user_content
            messages = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return sys_msg + "\n\n" + user_content

# %% — Cell 2: Load Model
print("\n" + "=" * 70)
print("LOADING MODEL (McGill-NLP/AfriqueLlama-8B)")
print("=" * 70)

# ── Aggressive Memory Cleanup ──
# If you ran previous cells, old models might still be taking up VRAM.
import gc
for var_name in ['model', 'gen_model', 'tokenizer', 'gen_tokenizer', '_afrolm_bundle']:
    if var_name in globals():
        del globals()[var_name]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

assert torch.cuda.is_available(), "No CUDA GPU available! Switch runtime to T4 GPU."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "McGill-NLP/AfriqueLlama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model.eval()
device = next(model.parameters()).device
print(f"✅ Loaded: {model_id}")

# %% — Cell 3: Generate Predictions
print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS (FEW-SHOT)")
print("=" * 70)

def single_generate(prompt, bounds, use_sampling=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536) # Increased max_length for few-shot
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = {
        "min_new_tokens": bounds["min_new_tokens"],
        "max_new_tokens": bounds["max_new_tokens"],
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if use_sampling:
        gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    else:
        gen_kwargs.update({"num_beams": 1, "do_sample": False})

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

predictions = []
checkpoint_path = PROCESSED_DIR / "predictions_test_exp002_partial.csv"
start_idx = 0

if checkpoint_path.exists():
    partial = pd.read_csv(checkpoint_path, dtype=str).fillna("")
    predictions = partial["prediction"].tolist()
    start_idx = len(predictions)
    print(f"Resuming from checkpoint at row {start_idx}")

# Use fixed bounds to save compute time (max 80 tokens)
bounds = {"min_new_tokens": 1, "max_new_tokens": 80}

for i in tqdm(range(start_idx, len(test)), desc="Generating [Test]"):
    row = test.iloc[i]
    lang = canonical_prompt_language(LANG_CODE_MAP[str(row[lang_col])])
    q = str(row[question_col])

    prompt = build_prompt(q, lang, tokenizer, force_answer=False)
    answer = single_generate(prompt, bounds, use_sampling=False)
    answer = strip_prompt_artefacts(answer, lang)

    if is_refusal(answer):
        prompt_forced = build_prompt(q, lang, tokenizer, force_answer=True)
        answer = single_generate(prompt_forced, bounds, use_sampling=True)
        answer = strip_prompt_artefacts(answer, lang)

    if is_refusal(answer):
        answer = q

    predictions.append(answer)

    # Checkpoint every 10 rows
    if (i + 1) % 10 == 0:
        pd.DataFrame({
            "id": test[id_col].astype(str).tolist()[:len(predictions)],
            "prediction": predictions
        }).to_csv(checkpoint_path, index=False, encoding="utf-8")

# %% — Cell 4: Save Submission
print("\n" + "=" * 70)
print("BUILDING SUBMISSION")
print("=" * 70)

sub_columns = tuple(sample.columns.tolist())
submission = pd.DataFrame({sub_columns[0]: test[id_col].astype(str).values})
for tc in sub_columns[1:]:
    submission[tc] = predictions

submission_path = SUBMISSIONS_DIR / "submission_exp002_few_shot.csv"
submission.to_csv(submission_path, index=False, encoding="utf-8")
print(f"✅ Submission saved to: {submission_path}")

# For Colab:
from google.colab import files
try:
    files.download(str(submission_path))
except Exception as e:
    print(f"Please download the file manually from the left sidebar: {submission_path}")
