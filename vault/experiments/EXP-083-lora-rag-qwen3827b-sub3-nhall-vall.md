---
type: experiment
id: EXP-083
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-083 Second generator for cross-model MBR: Qwen3.8-27B QLoRA (r64, 1 epoch, same recipe as EXP-061) + 13 candidates via vLLM on Aka_Gha/Eng_Gha/Amh_Eth

- run: `exp083_lora_rag_qwen3827b_sub3_nhall-vall_70ce6b` · git `3c94bdd` · status **crash** · 13002.1 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: adapter, mode, model_id
- config: `{"mode": "lora_rag", "model_id": "Qwen/Qwen3.8-27B", "precision": "bf16", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
, line 573, in _committer_loop
    self._process_batch(batch)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/_upload_pipeline.py", line 610, in _process_batch
    self._commit_with_split(ops_to_commit)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/_upload_pipeline.py", line 614, in _commit_with_split
    self._do_commit(ops)
    ~~~~~~~~~~~~~~~^^^^^
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/_upload_pipeline.py", line 655, in _do_commit
    self.last_commit_info = _send_commit(
                            ~~~~~~~~~~~~^
        operations=operations,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
        retry_on_error=True,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/_commit_api.py", line 1068, in _send_commit
    hf_raise_for_status(response, endpoint_name="commit")
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 929, in hf_raise_for_status
    raise _format(BadRequestError, message, response) from e
huggingface_hub.errors.BadRequestError: (Request ID: Root=1-6ac8a27a-0ec4c4ec4d39e81e15b0ae0c;d6a824a0-10b0-4a98-b5a2-dc1be5423556)

Bad request for commit endpoint:
Private storage limit reached for user nyakaruosborn, please upgrade your plan to increase your private storage limit

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
