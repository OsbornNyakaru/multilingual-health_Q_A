---
type: finding
id: FND-001
created: 2026-09-24
status: confirmed
links: ["[[EXP-003-token-overlap-retrieval]]", "[[EXP-004-bge-m3-retrieval-heldout]]", "[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# FND-001 Retrieval strength splits cleanly by subset

Across two independent retrieval runs ([[EXP-003-token-overlap-retrieval]], crude token overlap on 2,088 rows; [[EXP-004-bge-m3-retrieval-heldout]], BGE-M3 on 1,491 rows), the per-subset ordering is stable:

| Group | Subsets | BGE-M3 R1 | Meaning |
|-------|---------|-----------|---------|
| Closed-pool (retrieval wins) | Swa_Ken, Eng_Ken, Eng_Uga, Eng_Eth | 0.64–0.82 | heavy question paraphrase/duplication (Eng_Eth: 41.6% exact duplicates in EXP-003) |
| Middle | Lug_Uga | 0.43 | generation candidate |
| Generative (retrieval fails) | Aka_Gha, Eng_Gha, Amh_Eth | 0.02–0.40 | low overlap; RL much lower than R1 (long, varied answers) |

- Amharic is the hard floor in both runs (0.084 / 0.012 ROUGE-only comb).
- Better embeddings lift the closed-pool subsets a lot (Eng_Ken R1 0.459 → 0.812), help Aka_Gha/Eng_Gha only moderately (R1 +0.10/+0.12, RL stays ≤ 0.23), and *reduce* Amh_Eth (0.117 → 0.016). The Amharic drop is suspicious; check it before trusting it.
- **2026-09-24 correction ([[metric-replica]]):** the EXP-004 numbers were scored with rouge-score's *default* tokenizer plus the Porter stemmer, not whitespace. That tokenizer erases Ge'ez script, so **Amh_Eth 0.016 / 0.012 is a tokenizer artifact and is invalid.** Under the starter's whitespace tokenizer, BGE-M3 Amh_Eth is R1 0.162 / RL 0.149 / ROUGE-only 0.115. Amharic is still the weakest subset, but it is not near 0, and the "better embeddings reduce Amh_Eth" claim above is unsupported. Whitespace also lowers Aka_Gha (0.228 → 0.171) and Eng_Gha (0.223 → 0.180) because it is case- and punctuation-sensitive. The closed-pool vs generative grouping is unchanged.
- The two runs use different held-out slices, so compare the patterns, not the absolute numbers.

Implication: supports [[H-011-closed-pool-vs-generative-router]] and [[D-003-retrieval-first-baseline]]; generation effort should target Amh_Eth, Aka_Gha, Eng_Gha, Lug_Uga.

## Links
- [[metric-replica]]
- [[00_INDEX]]
- [[EXP-003-token-overlap-retrieval]]
- [[EXP-004-bge-m3-retrieval-heldout]]
- [[EXP-006-e5-small-retrieval-test-submission]]
- [[H-011-closed-pool-vs-generative-router]]
- [[D-003-retrieval-first-baseline]]
- [[F-002-data-shape]]
