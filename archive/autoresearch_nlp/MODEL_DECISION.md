# MODEL_DECISION.md — choosing the base model (day-1 decision)

The base model is a real decision with three constraints: **license** (the
top-10 audit + Zindi's "openly available / usable" rule), **language coverage**
(4 African languages + English), and **compute** (whatever GPU you run on).
`train.py` ships with a placeholder so the loop runs in DRY_RUN; pick deliberately
before spending GPU time. Run AGENT_PLAYBOOK.md prompt #3 to do this properly.

## Verified facts (2026-06-07)

- **CohereLabs/aya-expanse-8b** — the id in train.py. Verified to exist on HF.
  Caveats: license is **CC-BY-NC** (non-commercial) with an acceptable-use policy,
  which is a risk for a prize competition whose winning code must be openly usable;
  and its documented language list (23 languages) **does not include Swahili,
  Luganda, Akan, or Amharic**. It is strong on English (your 56% majority) but a
  weak fit for the four African subsets. Treat as an English-only baseline at best.
  Source: https://huggingface.co/CohereLabs/aya-expanse-8b

I have NOT verified the following as fits — check each before use (license,
language coverage, that the HF id resolves):

- General permissive multilingual instruct models (e.g. Gemma-2 / Llama-3.1
  families) — broader licenses that are usually competition-acceptable, but
  confirm African-language quality on your held-out before trusting them.
- African-language-focused models (Masakhane / community seq2seq, mT5/byT5
  fine-tunes, etc.) — may handle Luganda/Akan/Amharic better but verify they are
  instruction-capable and openly licensed.

The competition itself only mentions AfroLM (an *encoder*, used for the secondary
BertScore check) — it is not a generation model, so do not use it to answer.

## How to decide (don't guess)
1. Run the model-survey agent prompt (AGENT_PLAYBOOK.md #3) with your real compute
   budget. Demand: HF id that resolves, license, evidence on the 4 languages.
2. Shortlist 2-3. For each, run a zero-shot pass in DRY_RUN=False on a SMALL
   held-out sample and read per-subset ROUGE.
3. Pick the best license-clean option per subset. It is fine to use one model for
   English and another for the African subsets if that wins held-out — selection is
   per-row and the submission is just text.

## Compute reality
Your exp004 OOM'd loading an 8B 4-bit model on a 15.6 GB T4. On a T4, either use a
smaller base (≤3-4B) or a single model with conservative batch size; 8B 4-bit
inference is borderline and 8B QLoRA fine-tuning needs more headroom (L4/A100).
