---
type: experiment
id: EXP-066
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-066 vLLM smoke test: EXP-065's 13-candidate MBR on 150 Val rows, served by vLLM (base + LoRA) instead of HF generate

- run: `exp066_rag_bgem3_gemma431bi_k3_sub3_nv150_c76f20` · git `523f3d9` · status **crash** · 102.8 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp065_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_7b6d72` · changed: gen_engine
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'val': 150} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
3caa3c8541f5b522111d/autoresearch_nlp/experiment.py", line 469, in ensure_vllm
    raise RuntimeError("vLLM import failed:\n" + v.stderr[-3000:])
RuntimeError: vLLM import failed:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/__init__.py", line 14, in <module>
    import vllm.env_override  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/env_override.py", line 153, in <module>
    from vllm.utils.torch_utils import is_torch_equal, is_torch_equal_or_newer
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/utils/torch_utils.py", line 21, in <module>
    from vllm.utils.platform_utils import is_pin_memory_available
  File "/marimo/runner_work/vllm-0.31.0/lib/python3.12/site-packages/vllm/utils/platform_utils.py", line 10, in <module>
    import regex as re
  File "/tmp/uv-venv/lib/python3.13/site-packages/regex/__init__.py", line 1, in <module>
    import regex._main
  File "/tmp/uv-venv/lib/python3.13/site-packages/regex/_main.py", line 428, in <module>
    from regex import _regex_core
  File "/tmp/uv-venv/lib/python3.13/site-packages/regex/_regex_core.py", line 21, in <module>
    from regex import _regex
ImportError: cannot import name '_regex' from partially initialized module 'regex' (most likely due to a circular import) (/tmp/uv-venv/lib/python3.13/site-packages/regex/__init__.py)


```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
