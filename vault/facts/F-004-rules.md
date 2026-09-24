---
type: fact
id: F-004
created: 2026-09-24
status: confirmed
links: ["[[F-001-competition-metric]]", "[[D-002-reject-aya-expanse]]", "[[00_INDEX]]"]
---
# F-004 Rules we still follow

The challenge is **closed** (deadline 2026-06-21 per `docs/ROADMAP_TOP5.md`; README window 2026-04-24 → 2026-07-14). We are in **post-close practice mode**: Zindi still scores late submissions (ROUGE-1 / ROUGE-L / judge breakdown) but does not rank them. **Unlimited practice submissions, no daily cap.** (The original rules were 50 total, 5/day, final two picked manually via `scripts/select_final.py`.)

Rules that still bind us:
1. **Open-source / open-weight tools and pretrained models only.** Check licences (e.g. CC-BY-NC models were a prize-eligibility risk; see [[D-002-reject-aya-expanse]]).
2. **No paid APIs** — no hosted LLM calls for generation or judging.
3. **Always set the seed** (five-place seeding: python, numpy, torch, cuda, transformers; `src/afro_health_qa/seeding.py`). Zindi audited top-10 code, hence pinned deps and reproducibility infra.
4. Freely/openly available external data is permitted (e.g. HASH Crowdsourced Dataset on Harvard Dataverse, ~5,488 STI Q&A pairs — licence and test-overlap unverified; `autoresearch_nlp/COMPETITION_INTEL.md` §4).
5. House discipline (not Zindi's, ours): never train or draw few-shot examples from the held-out slice; never print an estimate where a measurement belongs. See [[D-005-held-out-protocol]].

Where things run: code is written locally and pushed to GitHub (`OsbornNyakaru/multilingual-health_Q_A`, `main`); GPU work runs on molab (RTX Pro 6000 Blackwell, 96 GB). No local GPU jobs. See [[Molab Platform]].

Sources migrated: `vault/Competition Overview.md`, `prompts/00_orchestrator.md`, `autoresearch_nlp/LESSONS.md`.

## Links
- [[00_INDEX]]
- [[F-001-competition-metric]]
- [[D-002-reject-aya-expanse]]
- [[D-005-held-out-protocol]]
- [[Molab Platform]]
- Legacy: [[Competition Overview]]
