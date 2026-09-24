---
type: experiment
id: EXP-006
created: 2026-09-24
run_date: 2026-06-24
status: confirmed
hypothesis: "[[H-011-closed-pool-vs-generative-router]]"
links: ["[[EXP-004-bge-m3-retrieval-heldout]]", "[[D-003-retrieval-first-baseline]]", "[[00_INDEX]]"]
---
# EXP-006 Retrieval test submission (e5-small, 2026-06-24)

- **File:** `submissions/20260624_095953_e5-small_retrieval_test.csv` + JSON sidecar. Script: `scripts/retrieval_test_submission.py`.
- **Method:** retrieval-top1, no generation. `model_hf_id: intfloat/multilingual-e5-small`, pool = 28,319 (train_core + held_out + Val), mean retrieval cos 0.9231, 2,618 rows.
- **Inconsistency in the sidecar:** `hypothesis` says "full-pool BGE-M3 within-subset nearest-neighbour retrieval" but `model_hf_id` is e5-small (likely a fallback proxy). Its note "Held-out R1=0.560/RL=0.491 measured on the same retriever" matches the BGE-M3 numbers of [[EXP-004-bge-m3-retrieval-heldout]], so the held-out figure may not describe the e5 model. Treat the held-out score of *this exact model* as unknown.
- **Local combined / public LB:** `null` in the sidecar — never recorded.

## Links
- [[00_INDEX]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[D-003-retrieval-first-baseline]]
