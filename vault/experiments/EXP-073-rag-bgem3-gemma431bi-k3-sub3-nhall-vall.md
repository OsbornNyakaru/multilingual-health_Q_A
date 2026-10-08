---
type: experiment
id: EXP-073
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-073 EXP-071 rerun after a kernel restart: 13-candidate MBR on all held-out + Val via vLLM

- run: `exp073_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_b384d0` · git `025fa6c` · status **crash** · 1.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp071_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_6ce1fb` · changed: —
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
name, resume, partial))
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/multilingual-health_Q_A-025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/autoresearch_nlp/experiment.py", line 1149, in run
    generate(cfg, eval_df, pickers[mode], ctx, answers, pool_df, meta=meta)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/multilingual-health_Q_A-025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/autoresearch_nlp/experiment.py", line 384, in generate
    return generate_vllm(cfg, rows, examples_for, ctx, answers, pool_df, meta)
  File "/marimo/runner_work/code/025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/multilingual-health_Q_A-025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/autoresearch_nlp/experiment.py", line 532, in generate_vllm
    mem = free_gpu_for_vllm(cfg, ctx)
  File "/marimo/runner_work/code/025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/multilingual-health_Q_A-025fa6cc7c7bf6e64ea0aebd9b9b280c4c0d5c92/autoresearch_nlp/experiment.py", line 509, in free_gpu_for_vllm
    raise RuntimeError(f"only {free / 2**30:.1f} GiB of GPU memory free for vLLM; something in the kernel still "
                       "holds a model. Restart the molab kernel and press Run queue.")
RuntimeError: only 22.1 GiB of GPU memory free for vLLM; something in the kernel still holds a model. Restart the molab kernel and press Run queue.

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
