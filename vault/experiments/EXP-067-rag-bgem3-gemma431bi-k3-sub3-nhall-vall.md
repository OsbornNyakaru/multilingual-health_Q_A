---
type: experiment
id: EXP-067
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-067 EXP-065 (13-candidate MBR on EXP-061's adapter) on all held-out + Val rows, generated with vLLM

- run: `exp067_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_de17f7` · git `523f3d9` · status **crash** · 5.9 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp065_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_7b6d72` · changed: gen_engine
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
lingual-health_Q_A-523f3d9ff0dec26e30cc3caa3c8541f5b522111d/autoresearch_nlp/experiment.py", line 1110, in run
    generate(cfg, eval_df, pickers[mode], ctx, answers, pool_df, meta=meta)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/523f3d9ff0dec26e30cc3caa3c8541f5b522111d/multilingual-health_Q_A-523f3d9ff0dec26e30cc3caa3c8541f5b522111d/autoresearch_nlp/experiment.py", line 384, in generate
    return generate_vllm(cfg, rows, examples_for, ctx, answers, pool_df, meta)
  File "/marimo/runner_work/code/523f3d9ff0dec26e30cc3caa3c8541f5b522111d/multilingual-health_Q_A-523f3d9ff0dec26e30cc3caa3c8541f5b522111d/autoresearch_nlp/experiment.py", line 510, in generate_vllm
    py = ensure_vllm(cfg, ctx)
  File "/marimo/runner_work/code/523f3d9ff0dec26e30cc3caa3c8541f5b522111d/multilingual-health_Q_A-523f3d9ff0dec26e30cc3caa3c8541f5b522111d/autoresearch_nlp/experiment.py", line 460, in ensure_vllm
    subprocess.run(uv + ["venv", "--seed", "--python", "3.12", str(env)], check=True)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/subprocess.py", line 577, in run
    raise CalledProcessError(retcode, process.args,
                             output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['/tmp/uv-venv/bin/python', '-m', 'uv', 'venv', '--seed', '--python', '3.12', '/marimo/runner_work/vllm-0.31.0']' returned non-zero exit status 2.

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
