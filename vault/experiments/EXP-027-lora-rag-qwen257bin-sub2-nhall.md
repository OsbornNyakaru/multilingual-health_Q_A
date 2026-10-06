---
type: experiment
id: EXP-027
created: 2026-10-06
status: rejected
links: ["[[H-012-rag-enriched-finetune]]", "[[00_INDEX]]"]
---
# EXP-027 LoRA SMOKE: RAG-enriched LoRA on Qwen2.5-7B, 10% of work_train, 1 epoch, k=3 (reference r64/a64/drop0.5/lr2e-4); eval held-out Aka_Gha + Eng_Gha

- run: `exp027_lora_rag_qwen257bin_sub2_nhall_26988d` · git `abbac19` · status **crash** · 0.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp025_rag_bgem3_qwen257bin_k2_sub2_nh40_cc19d5` · changed: few_shot_k, lora_data_frac, lora_epochs, mode
- config: `{"mode": "lora_rag", "few_shot_k": 3, "lora_epochs": 1, "lora_data_frac": 0.1}` (non-default keys)
- eval: {'held_out': 0} · subsets: ['Aka_Gha', 'Eng_Gha']
- adopted for subsets: **none**

**error**

```
Traceback (most recent call last):
  File "/tmp/marimo_216/__marimo__cell_BYtC_.py", line 80, in run_one
    answers, meta = experiment.run(spec["config"], eval_df.drop(columns=["output"], errors="ignore"), pool_df, Ctx())
                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/abbac19b2d3751d00d6d267ca6b9b6578f32fb71/multilingual-health_Q_A-abbac19b2d3751d00d6d267ca6b9b6578f32fb71/autoresearch_nlp/experiment.py", line 470, in run
    raise RuntimeError("lora_rag needs the runner with the setup step (2026-10-06): reopen notebooks/molab_runner.py on molab")
RuntimeError: lora_rag needs the runner with the setup step (2026-10-06): reopen notebooks/molab_runner.py on molab

```

## Links
- [[H-012-rag-enriched-finetune]]
- [[00_INDEX]]
