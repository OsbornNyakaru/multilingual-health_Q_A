---
type: experiment
id: EXP-083
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-083 Second generator for cross-model MBR: Qwen3.8-27B QLoRA (r64, 1 epoch, same recipe as EXP-061) + 13 candidates via vLLM on Aka_Gha/Eng_Gha/Amh_Eth

- run: `exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b` · git `3c94bdd` · status **crash** · 11216.0 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: adapter, mode, model_id
- config: `{"mode": "lora_rag", "model_id": "Qwen/Qwen3.8-27B", "precision": "bf16", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
rn cls(
           ^^^^
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 111, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 126, in make_client
    return SyncMPClient(
           ^^^^^^^^^^^^^
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/tracing/otel.py", line 175, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 917, in __init__
    super().__init__(
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 674, in __init__
    with launch_core_engines(
         ^^^^^^^^^^^^^^^^^^^^
  File "/home/marimo/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1268, in launch_core_engines
    wait_for_engine_startup(
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1348, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}


```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
