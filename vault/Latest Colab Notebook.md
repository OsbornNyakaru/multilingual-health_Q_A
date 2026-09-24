---
tags: [notebooks]
---
# Latest Colab Notebook (`resources/multilingual_qa.ipynb`)

The most recent inference notebook (identical copy also inside the Zindi zip folder in `resources/`). Colab T4, Google Drive mounted at `/content/drive/MyDrive/afro-health-qa/data/raw`.

## Pipeline
1. `pip install transformers accelerate bitsandbytes pandas tqdm`.
2. Load Test + SampleSubmission + Train (few-shot pool). Column detection handles both the real schema and the old boilerplate.
3. Two few-shot examples per language (answers 50–300 chars, `random_state=42`), per-language system prompts (doctor persona, "never refuse"), refusal detection patterns, `strip_prompt_artefacts()` post-processing.
4. Model: **McGill-NLP/AfriqueLlama-8B**, 4-bit NF4, `device_map=auto`.
5. "Turbo engine": left padding fix, cleaned generation_config (greedy, no temperature), prompts trimmed to ~140 tokens, few-shot answers capped at 150 chars, prompts sorted by length for tight batching, `max_new_tokens 80`, `max_length 400`, batch 8, checkpoint every 25 batches to `/content/working/submissions/final_checkpoint.csv`, VRAM flush every 20 batches. Restores original ID order at the end.
6. Cell 5 is 66 KB of accidental `h` keystrokes; ignore.

## Lessons carried into [[Molab Notebook Plan]]
- Left padding + cleaned generation config are mandatory for batched decoder-only generation.
- Sort by prompt length before batching.
- Checkpoint the submission incrementally; molab kills idle sessions after 90 min and all sessions at 12 h ([[Molab Platform]]).
- It never scored held-out locally, so no ROUGE number exists for this run. The [[Autoresearch Harness]] fixes that.

Related: [[Model Decisions]].
