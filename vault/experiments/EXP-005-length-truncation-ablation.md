---
type: experiment
id: EXP-005
created: 2026-09-24
run_date: 2026-06-24
status: rejected
hypothesis: "[[H-003-per-language-length-bounds]]"
links: ["[[H-003-per-language-length-bounds]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[FND-002-length-truncation-on-retrieval]]", "[[00_INDEX]]"]
---
# EXP-005 Per-subset length truncation of retrieval outputs (ablation)

- **Hypothesis:** [[H-003-per-language-length-bounds]] (applied to retrieval predictions).
- **Base run:** [[EXP-004-bge-m3-retrieval-heldout]] (same 1,491 rows; baseline columns match EXP-004 exactly).
- **Single change:** truncate each prediction to a per-subset "optimal" word length.
- **Source (measured):** repo root `length_optimization_summary.csv`, `length_optimization_results.csv`, `length_calibration_full_report.csv`. Score = ROUGE-only combined.

| subset | gold median len | pred len | optimal len | baseline | truncated | Δ |
|--------|---:|---:|---:|---:|---:|---:|
| Aka_Gha | 101.0 | 106.9 | 121 | 0.2281 | 0.2286 | +0.0005 |
| Amh_Eth | 20.0 | 19.2 | 14 | 0.0121 | 0.0134 | +0.0013 |
| Eng_Eth | 25.0 | 24.7 | 30 | 0.4599 | 0.4534 | −0.0065 |
| Eng_Gha | 66.0 | 73.8 | 79 | 0.2234 | 0.2223 | −0.0012 |
| Eng_Ken | 63.5 | 77.5 | 78 | 0.5912 | 0.5397 | −0.0515 |
| Eng_Uga | 73.0 | 94.1 | 95 | 0.5622 | 0.5108 | −0.0514 |
| Lug_Uga | 68.0 | 73.9 | 81 | 0.3019 | 0.2843 | −0.0176 |
| Swa_Ken | 65.5 | 86.3 | 85 | 0.5942 | 0.5374 | −0.0569 |

- **Verdict:** revert. Truncation hurts every high-overlap subset; the only gains (Amh_Eth, Aka_Gha) are within noise (< +0.003). Folded into [[FND-002-length-truncation-on-retrieval]].
- This was the only ablation with results; `experiments/ABLATIONS.md` otherwise holds planned sweeps, now tracked as hypotheses ([[H-002-beam-search-lift]], [[H-003-per-language-length-bounds]], [[H-004-qlora-comp-only-beats-zero-shot]], [[H-005-medmcqa-translated-augmentation]], [[H-010-per-language-prompts]]).

## Links
- [[00_INDEX]]
- [[H-003-per-language-length-bounds]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[FND-002-length-truncation-on-retrieval]]
