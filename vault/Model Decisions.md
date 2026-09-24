---
tags: [models]
---
> Migrated to vault/ graph, see [[00_INDEX]]

# Model Decisions

What was tried or considered, and the verdicts recorded in the repo.

| Model | Where | Verdict |
|-------|-------|---------|
| CohereLabs/aya-expanse-8b | `train.py` default, configs | Exists on HF but CC-BY-NC licence (risk for prize comps) and language list does not cover swa/lug/aka/amh. English-only baseline at best. exp004 OOM'd loading it 4-bit on a 15.6 GB T4. |
| McGill-NLP/AfriqueLlama-8B | [[Latest Colab Notebook]] | Actually run. Llama-3 chat template. 4-bit NF4 on T4 was very slow (130 s/batch before the "turbo" prompt trimming, ~8–12 s after). Produced `submissions/final_checkpoint.csv`. |
| Gemma-2-9B | `configs/models/gemma2_9b.yaml` | Config only. |
| e5-small retrieval | `scripts/retrieval_test_submission.py` | Retrieval-as-answer baseline, `submissions/20260624_*_e5-small_retrieval_test.csv`. |
| BGE retrieval | `scripts/retrieval_baseline_bge.py`, report data | Retrieval scores on held-out in `docs/competition_report/data/`. |
| AfroLM | evaluation only | Encoder for BERTScore, not a generator. |

## What changes on Molab
The RTX Pro 6000 (96 GB VRAM) means an 8B model fits in **bf16 with no bitsandbytes**, batch sizes of 32+, and LoRA/QLoRA fine-tuning becomes realistic. Prioritise: (1) AfriqueLlama-8B bf16 zero/few-shot to reproduce the old baseline fast, (2) LoRA SFT on work_train + Val, (3) larger permissive models (Gemma-2-9B/27B, Llama-3.1-8B, Qwen2.5) for per-subset selection.

Related: [[Autoresearch Harness]], [[Molab Platform]].
