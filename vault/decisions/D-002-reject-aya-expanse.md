---
type: decision
id: D-002
created: 2026-09-24
status: confirmed
links: ["[[EXP-001-aya-expanse-oom]]", "[[D-004-base-model-for-molab]]", "[[F-004-rules]]", "[[00_INDEX]]"]
---
# D-002 Do not use Aya-Expanse-8B as the base model

**Decision:** Drop `CohereLabs/aya-expanse-8b` (still the default in `autoresearch_nlp/train.py` and `configs/`) as a candidate generator.

**Why:**
- CC-BY-NC licence — a risk under the open-source/prize rules ([[F-004-rules]]).
- Its language list does not cover Swahili, Luganda, Akan or Amharic — English-only baseline at best.
- The 4-bit load OOM'd on a 15.6 GB T4 ([[EXP-001-aya-expanse-oom]]); no measured score ever existed.

**Alternatives:** see [[D-004-base-model-for-molab]] (AfriqueLlama-8B, Gemma-2-9B/27B, Llama-3.1-8B, Qwen2.5, Sunflower-32B).

**Consequences:** supersedes [[H-001-zero-shot-aya-floor]] and [[H-009-gemma-aya-adapter-ensemble]]. `train.py` MODEL_ID must be changed before the first real molab run.

Source: legacy [[Model Decisions]], `autoresearch_nlp/MODEL_DECISION.md`.

## Links
- [[00_INDEX]]
- [[EXP-001-aya-expanse-oom]]
- [[D-004-base-model-for-molab]]
- [[H-001-zero-shot-aya-floor]]
- [[H-009-gemma-aya-adapter-ensemble]]
- [[F-004-rules]]
