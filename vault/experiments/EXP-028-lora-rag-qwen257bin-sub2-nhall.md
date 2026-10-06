---
type: experiment
id: EXP-028
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-028 LoRA SMOKE (re-run of EXP-027, which hit the old runner): RAG-enriched LoRA on Qwen2.5-7B, 10% of work_train, 1 epoch, k=3; eval held-out Aka_Gha + Eng_Gha

- run: `exp028_lora_rag_qwen257bin_sub2_nhall_5b098e` · git `23858d8` · status **crash** · 0.3 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp027_lora_rag_qwen257bin_sub2_nhall_26988d` · changed: —
- config: `{"mode": "lora_rag", "few_shot_k": 3, "lora_epochs": 1, "lora_data_frac": 0.1}` (non-default keys)
- eval: {'held_out': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**error**

```
Traceback (most recent call last):
  File "/tmp/marimo_304/__marimo__cell_BYtC_.py", line 80, in run_one
    answers, meta = experiment.run(spec["config"], eval_df.drop(columns=["output"], errors="ignore"), pool_df, Ctx())
                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/23858d885c78cd40819397b83d6c8f5599849f7e/multilingual-health_Q_A-23858d885c78cd40819397b83d6c8f5599849f7e/autoresearch_nlp/experiment.py", line 470, in run
    raise RuntimeError("lora_rag needs the runner with the setup step (2026-10-06): reopen notebooks/molab_runner.py on molab")
RuntimeError: lora_rag needs the runner with the setup step (2026-10-06): reopen notebooks/molab_runner.py on molab

```

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
