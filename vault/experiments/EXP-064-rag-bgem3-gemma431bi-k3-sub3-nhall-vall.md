---
type: experiment
id: EXP-064
created: 2026-10-08
status: rejected
links: ["[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# EXP-064 MBR on EXP-061's adapter: greedy + 4 samples at each of T=0.7/1.0/1.3 (13 candidates), keep the ROUGE medoid; all candidates saved for cross-model MBR

- run: `exp064_rag_bgem3_gemma431bi_k3_sub3_nhall-vall_72c8c7` · git `c1285e8` · status **crash** · 261.9 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1` · changed: adapter, gen_sample_batch, gen_samples, mode
- config: `{"mode": "rag_few_shot", "model_id": "google/gemma-4-31B-it", "precision": "bf16", "adapter": "run:exp061_lora_rag_gemma431bi_sub3_nhall-vall_fb5ca1", "infer_batch": 16, "gen_samples": 4, "gen_sample_batch": 32, "few_shot_k": 3, "lora_epochs": 1, "lora_batch": 2, "lora_grad_acc": 2, "lora_train_subsets": ["Aka_Gha", "Eng_Gha", "Amh_Eth"], "lora_bits": 4, "lora_save_steps": 250}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Aka_Gha', 'Eng_Gha', 'Amh_Eth']
- adopted for subsets: **none**

**error**

```
al/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/tmp/uv-venv/lib/python3.13/site-packages/transformers/models/gemma4/modeling_gemma4.py", line 1252, in forward
    attn_output, attn_weights = attention_interface(
                                ~~~~~~~~~~~~~~~~~~~^
        self,
        ^^^^^
    ...<7 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/tmp/uv-venv/lib/python3.13/site-packages/transformers/integrations/sdpa_attention.py", line 112, in sdpa_attention_forward
    value = repeat_kv(value, module.num_key_value_groups)
  File "/tmp/uv-venv/lib/python3.13/site-packages/transformers/integrations/sdpa_attention.py", line 27, in repeat_kv
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 888.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 479.56 MiB is free. Process 1 has 94.50 GiB memory in use. Of the allocated memory 85.16 GiB is allocated by PyTorch, and 8.66 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

```

## Links
- [[H-014-bigger-generator]]
- [[00_INDEX]]
