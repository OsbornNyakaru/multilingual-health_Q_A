---
type: finding
id: FND-004
created: 2026-10-09
status: open
links: ["[[FND-003-research-synthesis-2026-10-09]]", "[[H-011-closed-pool-vs-generative-router]]", "[[00_INDEX]]"]
---
# FND-004 Many Lug_Uga stock answers are translations of Eng_Uga stock answers

**Check (CPU, Train + Val):** Luganda distinct answers were aligned to English distinct answers on the shared words both languages keep: drug names, acronyms, numbers. The words were weighted by rarity.

- Lug_Uga has 1,205 distinct answers and Eng_Uga 1,976. Only 184 rare words are shared, so this matching reaches 18–23% of Lug rows. That coverage is a limit of the method, not of the parallelism.
- Where it reaches, the pairs are sentence-for-sentence translations, with a median length ratio of 0.91. Examples:
  - the most common Lug answer, chlamydia treatment (30 rows), against the Eng answer (83 rows);
  - BCG vaccine 80% effective;
  - the HIV self-test kit steps;
  - "AIDS in full is…".
- Answer frequencies differ, so Lug_Uga is not simply a translated subset of Eng_Uga rows. Some anchor matches are wrong (for example, PEP matched to a shorter Eng answer). Treat this as parallel at the level of stock answers, not of rows.

**Why it matters:** Lug_Uga carries the biggest selection headroom (+0.024 on the leaderboard if perfect). It loses mainly at finding the answer: gold is in the top-5 for 76% of questions against 88% in Train. English retrieval is much stronger, with gold in the top-5 for 87% on Eng_Uga.

**Next (GPU):** translate-then-retrieve. Machine-translate the Lug questions (pool and queries) to English with NLLB-200 (lug_Latn) or Gemma. Retrieve on the English text within Lug_Uga, then blend with the Luganda similarity, or feed it to the ranker as an input. This needs no answer alignment.

A later step could also use Eng_Uga's larger question pool through aligned answer pairs. That needs a real alignment, from cross-lingual embeddings of the machine-translated answers.

## Links
- [[FND-003-research-synthesis-2026-10-09]]
- [[H-011-closed-pool-vs-generative-router]]
- [[00_INDEX]]
