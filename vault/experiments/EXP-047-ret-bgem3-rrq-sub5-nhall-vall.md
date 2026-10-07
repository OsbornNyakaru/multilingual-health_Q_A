---
type: experiment
id: EXP-047
created: 2026-10-07
status: rejected
links: ["[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# EXP-047 fine-tuned BGE-M3 retriever (same-answer pairs + hard negatives, all subsets), learned selector retrained on its candidates; target: Luganda recall@50 73% -> ~85%

- run: `exp047_ret_bgem3_rrq_sub5_nhall-vall_da9740` · git `5e64a4a` · status **crash** · 11.6 s on NVIDIA RTX PRO 6000 Blackwell Server Edition
- parent: `exp039_ret_bgem3_rrq_sub5_nhall-vall_e658df` · changed: embedder_train
- config: `{"embedder_train": true, "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_k": 50, "rerank_train": true, "no_repeat_ngram": 3, "diag_k": 50}` (non-default keys)
- eval: {'held_out': 0, 'val': 0} · subsets: ['Eng_Uga', 'Lug_Uga', 'Swa_Ken', 'Eng_Ken', 'Eng_Eth']
- adopted for subsets: **none**

**error**

```
Traceback (most recent call last):
  File "/tmp/marimo_206/__marimo__cell_BYtC_.py", line 93, in run_one
    experiment.setup(spec["config"], {k: (e.drop(columns=["output"], errors="ignore"), p) for k, (e, p) in sets.items()},
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     make_ctx("setup", {}, None))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/5e64a4a514fa46067930d7f6ef40b67f72c1dd85/multilingual-health_Q_A-5e64a4a514fa46067930d7f6ef40b67f72c1dd85/autoresearch_nlp/experiment.py", line 615, in setup
    ctx.cache[_embedder_key(ctx, cfg)] = train_embedder(cfg, sets[name][1], ctx)
                                         ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/marimo/runner_work/code/5e64a4a514fa46067930d7f6ef40b67f72c1dd85/multilingual-health_Q_A-5e64a4a514fa46067930d7f6ef40b67f72c1dd85/autoresearch_nlp/experiment.py", line 461, in train_embedder
    q = embed([t[0] for t in batch])
  File "/marimo/runner_work/code/5e64a4a514fa46067930d7f6ef40b67f72c1dd85/multilingual-health_Q_A-5e64a4a514fa46067930d7f6ef40b67f72c1dd85/autoresearch_nlp/experiment.py", line 451, in embed
    feats = {k: v.to(dev) for k, v in model.tokenize(texts).items()}
                ^^^^
AttributeError: 'str' object has no attribute 'to'

```

## Links
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
