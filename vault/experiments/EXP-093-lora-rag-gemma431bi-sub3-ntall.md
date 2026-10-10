---
type: experiment
id: EXP-093
created: 2026-10-09
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-093 Test predictions for EXP-077 from a Gemma-4-31B LoRA retrained on Train + Val (the test pool; EXP-061's adapter saw only work_train), same recipe and 13-candidate vLLM MBR; 1st place retrained on all data

- run: `exp093_lora_rag_gemma431bi_sub3_ntall_ec4e7e` · git `3f3b977` · status **crash** · 8950.5 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp077_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_40dd01` · changed: adapter, mode
- config: `{"mode": "lora_rag", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 12, "gen_engine": "vllm", "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'test': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
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
  File "/tmp/uv-venv/lib/python3.13/site-packages/huggingface_hub/utils/_http.py", line 959, in hf_raise_for_status
    raise _format(BadRequestError, message, response) from e
huggingface_hub.errors.BadRequestError: (Request ID: Root=1-6ac97601-2f2ac41f7d47700b5b831f8e;df3279de-7934-40a2-a76a-ce5929736ec2)

Bad request for commit endpoint:
Private storage limit reached for user nyakaruosborn, please upgrade your plan to increase your private storage limit

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
