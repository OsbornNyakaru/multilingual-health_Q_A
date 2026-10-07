---
type: experiment
id: EXP-058
created: 2026-10-07
status: rejected
links: ["[[00_INDEX]]"]
---
# EXP-058 test predictions for EXP-054 (graded-label selector; input to the learned final ranker)

- run: `exp058_ret_bgem3_rrb_sub4_ntall_cc1c32` · git `7f555d3` · status **crash** · 92.4 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp054_ret_bgem3_rrb_sub5_nhall-vall_984c50` · changed: —
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_on": "both", "rerank_train": true, "rerank_train_max_len": 384, "rerank_train_graded": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'test': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken']
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
huggingface_hub.errors.BadRequestError: (Request ID: Root=1-6ac67c7c-7890d2f035e950096e1032eb;7e63f510-798f-42e6-86c2-3e3eef152b91)

Bad request for commit endpoint:
Private storage limit reached for user nyakaruosborn, please upgrade your plan to increase your private storage limit

```

## Links
- [[00_INDEX]]
