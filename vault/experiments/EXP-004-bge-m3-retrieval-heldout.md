---
type: experiment
id: EXP-004
created: 2026-09-24
run_date: 2026-06-24
status: confirmed
hypothesis: "[[H-011-closed-pool-vs-generative-router]]"
links: ["[[EXP-003-token-overlap-retrieval]]", "[[FND-001-retrieval-strength-by-subset]]", "[[F-003-reference-approach]]", "[[00_INDEX]]"]
---
# EXP-004 BGE-M3 within-subset retrieval (held-out 1,491) — CURRENT BEST

- **Method:** `BAAI/bge-m3` embeddings, cosine on normalized vectors; for each held-out question return the answer of the nearest *train* question **within the same subset**. Pool `data/processed/train_core.csv`, queries `data/processed/held_out.csv` (n = 1,491 at the time). Script: `scripts/retrieval_baseline_bge.py`.
- **Source (measured):** `docs/competition_report/data/bge_retrieval_scores.json` (+ per-row `bge_retrieval_heldout.csv`). ROUGE with the LB-matching whitespace tokenizer. Judge term **not** computed.

| subset | n | R1 | RL | ROUGE-only comb | mean cos |
|--------|--:|---:|---:|---:|---:|
| Aka_Gha | 223 | 0.3952 | 0.2212 | 0.2281 | 0.8244 |
| Amh_Eth | 92 | 0.0163 | 0.0163 | 0.0121 | 0.8145 |
| Eng_Eth | 196 | 0.6356 | 0.6073 | 0.4599 | 0.9293 |
| Eng_Gha | 222 | 0.3720 | 0.2319 | 0.2234 | 0.8551 |
| Eng_Ken | 104 | 0.8116 | 0.7864 | 0.5912 | 0.9034 |
| Eng_Uga | 381 | 0.7777 | 0.7419 | 0.5622 | 0.9225 |
| Lug_Uga | 169 | 0.4332 | 0.3827 | 0.3019 | 0.8462 |
| Swa_Ken | 104 | 0.8162 | 0.7899 | 0.5942 | 0.8960 |
| **ALL** | **1491** | **0.5604** | **0.4914** | **0.3892** | 0.8802 |

**2026-09-24 correction ([[metric-replica]]):** the table above reproduces exactly under rouge-score's **default tokenizer + Porter stemmer** (the old `score_rouge`), **not** the whitespace tokenizer claimed above. Rescored from the same per-row file with the starter's whitespace tokenizer: R1 **0.5237**, RL **0.4764**, ROUGE-only **0.3700**. Per subset: Aka_Gha 0.1706, Amh_Eth 0.1151, Eng_Eth 0.4379, Eng_Gha 0.1798, Eng_Ken 0.5821, Eng_Uga 0.5470, Lug_Uga 0.2795, Swa_Ken 0.5881. Use 0.3700 when comparing with any whitespace-scored run.

## Caveats
- The 1,491-row held-out slice is **not** the current harness split (2,088 rows, [[D-005-held-out-protocol]]); `data/processed/held_out.csv` has since been regenerated. Not directly comparable with [[EXP-003-token-overlap-retrieval]]. Re-run on the current split before using it as the loop's baseline.
- Compared with the 11th-place BGE-M3 retrieval (R1 0.5548 / RL 0.4823, [[F-003-reference-approach]]), ours is R1 0.5604 / RL 0.4914 — similar, but on a different evaluation set.
- Amh_Eth ROUGE ≈ 0.016 despite cos 0.81 — suspicious; check tokenizer / script handling (see [[H-013-whitespace-tokenizer-matches-grader]]).

## Links
- [[metric-replica]]
- [[00_INDEX]]
- [[EXP-003-token-overlap-retrieval]]
- [[EXP-005-length-truncation-ablation]]
- [[FND-001-retrieval-strength-by-subset]]
- [[F-003-reference-approach]]
- [[D-003-retrieval-first-baseline]]
