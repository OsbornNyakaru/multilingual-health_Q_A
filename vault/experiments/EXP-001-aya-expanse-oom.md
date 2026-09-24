---
type: experiment
id: EXP-001
created: 2026-09-24
run_date: 2026-05 (exact date unknown; "exp004" notebook)
status: rejected
hypothesis: "[[H-001-zero-shot-aya-floor]]"
links: ["[[H-001-zero-shot-aya-floor]]", "[[D-002-reject-aya-expanse]]", "[[00_INDEX]]"]
---
# EXP-001 Aya-Expanse-8B 4-bit on T4 (repo's "exp004")

- **Hypothesis:** [[H-001-zero-shot-aya-floor]].
- **Model:** `CohereLabs/aya-expanse-8b`, 4-bit, Kaggle/Colab T4 (15.6 GB).
- **Result:** OOM while loading. No measured score. (`vault/Model Decisions.md`, `autoresearch_nlp/MODEL_DECISION.md`.)
- **Integrity note:** the same exp004 notebook concatenated all of Val.csv into training and then printed hard-coded "estimates" (ROUGE-1 0.63–0.70, "Target LB 0.768095"). Those were **never measured** and must not be quoted as results (`autoresearch_nlp/LESSONS.md` #4).
- **Verdict:** reverted; led to [[D-002-reject-aya-expanse]] and [[D-005-held-out-protocol]].

## Links
- [[00_INDEX]]
- [[H-001-zero-shot-aya-floor]]
- [[D-002-reject-aya-expanse]]
- [[D-005-held-out-protocol]]
