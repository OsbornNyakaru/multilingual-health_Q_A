---
type: experiment
id: EXP-084
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-084 13-candidate vLLM generation with EXP-083's Qwen3.8-27B adapter (max_num_seqs 256; EXP-083's own generation hits the Mamba-cache limit)

- run: `exp084_rag_bgem3_qwen3827b_k3_sub3_nhall-vall_e80aaa` · git `5f9645c` · status **crash** · 0.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: adapter, model_id, vllm_max_seqs
- config: `{"mode": "rag_few_shot", "model_id": "Qwen/Qwen3.8-27B", "precision": "bf16", "adapter": "run:exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "vllm_max_seqs": 256, "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
Traceback (most recent call last):
  File "/tmp/marimo_29222/__marimo__cell_BYtC_.py", line 104, in run_one
    answers, meta = experiment.run(spec["config"], eval_df.drop(columns=["output"], errors="ignore"), pool_df,
                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   make_ctx(set_name, resume, partial))
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/5f9645c2416948ee482471b1f902cb4ac235c4bb/multilingual-health_Q_A-5f9645c2416948ee482471b1f902cb4ac235c4bb/autoresearch_nlp/experiment.py", line 1083, in run
    cfg["adapter"] = resolve_adapter(cfg["adapter"], ctx)
                     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/5f9645c2416948ee482471b1f902cb4ac235c4bb/multilingual-health_Q_A-5f9645c2416948ee482471b1f902cb4ac235c4bb/autoresearch_nlp/experiment.py", line 644, in resolve_adapter
    return resolve_artifact(adapter, ctx, "adapter", "adapter_config.json")
  File "/marimo/runner_work/code/5f9645c2416948ee482471b1f902cb4ac235c4bb/multilingual-health_Q_A-5f9645c2416948ee482471b1f902cb4ac235c4bb/autoresearch_nlp/experiment.py", line 635, in resolve_artifact
    raise FileNotFoundError(f"no {folder} uploaded for run {rid}")
FileNotFoundError: no adapter uploaded for run exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
