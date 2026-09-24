---
type: experiment
id: EXP-003
created: 2026-09-24
run_date: 2026-06-07
status: confirmed
hypothesis: "[[H-011-closed-pool-vs-generative-router]]"
links: ["[[FND-001-retrieval-strength-by-subset]]", "[[D-003-retrieval-first-baseline]]", "[[00_INDEX]]"]
---
# EXP-003 Crude token-overlap nearest-neighbour retrieval (held-out 2,088)

- **Question:** how strong is "return the nearest train answer" on this metric?
- **Method:** crude token-overlap nearest neighbour, per subset, fit on `work_train` (27,727), predict `held_out` (2,088; 7% stratified, seed 1234). Whitespace ROUGE. Weaker than the starter's char-ngram TF-IDF, so a **lower bound**.
- **Source:** `autoresearch_nlp/COMPETITION_INTEL.md` §3 (measured).

| subset | n | exact% | R1 | RL | ROUGE-only comb | simJ |
|--------|--:|------:|---:|---:|---:|---:|
| Aka_Gha | 312 | 0.0 | 0.294 | 0.174 | 0.173 | 0.42 |
| Amh_Eth | 129 | 0.0 | 0.117 | 0.110 | 0.084 | 0.31 |
| Eng_Eth | 274 | 41.6 | 0.569 | 0.555 | 0.416 | 0.76 |
| Eng_Gha | 311 | 0.0 | 0.249 | 0.165 | 0.153 | 0.42 |
| Eng_Ken | 146 | 0.0 | 0.459 | 0.406 | 0.320 | 0.46 |
| Eng_Uga | 534 | 2.8 | 0.472 | 0.426 | 0.332 | 0.65 |
| Lug_Uga | 237 | 0.0 | 0.443 | 0.416 | 0.318 | 0.43 |
| Swa_Ken | 145 | 0.0 | 0.544 | 0.505 | 0.388 | 0.49 |
| **ALL** | **2088** | 6.2 | **0.404** | **0.350** | **0.279** | |

ROUGE-only comb = 0.37·R1 + 0.37·RL (judge term not computed). This is the only retrieval number measured on the **current** harness split.

## Links
- [[00_INDEX]]
- [[FND-001-retrieval-strength-by-subset]]
- [[D-003-retrieval-first-baseline]]
- [[EXP-004-bge-m3-retrieval-heldout]]
