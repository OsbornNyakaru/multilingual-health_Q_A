---
type: finding
id: FND-003
created: 2026-10-09
status: open
links: ["[[F-007]]", "[[H-014-bigger-generator]]", "[[00_INDEX]]"]
---
# FND-003 Research synthesis: what to try next (2026-10-09)

Three research passes: the winners' code, Kaggle/Zindi grandmaster write-ups, and papers. Full reports: `vault/research/A_winners_gap.md`, `B_grandmaster.md`, `C_papers.md`.

Starting point (Val): on the repeated-answer subsets the gold answer is in our top-5 but we pick another one on 9 pts of Eng_Uga, 15 of Lug_Uga, 7 of Swa_Ken, 8 of Eng_Ken, 13 of Eng_Eth. Perfect choice within the top-5 is worth about +0.03 on the final score.

## Ranked pipeline

| # | item | subsets | cost | est. gain | sources |
|--:|---|---|---|---|---|
| 0 | **EXP-085 probe**: base Gemma-4-31B chooser over the top-5, 5 cyclic shifts, per-candidate scores saved | closed 5 | ~1 GPU-h | signal check | papers (Found in the Middle, FIRST), Eedi 1st |
| 1 | **Ranker features** (CPU): candidate's near-duplicate count in the pool (CanonEcho), mean/max ROUGE vs other candidates, raw scores + per-question z/gap/is-max, one LambdaRank per subset | closed 5 | CPU | +0.004–0.010 | Darius `sel_features.py`, `features.py` |
| 2 | **LLM scores as ranker features**, not as an override (Darius: LLM-as-chooser −0.0055 LB, as feature it helped) | closed 5 | from #0 | +0.002–0.006 | Darius `llm_probe.py`, Eedi, papers |
| 3 | **Fine-tuned listwise chooser** (Eedi recipe): QLoRA r64, shuffled options, CE + soft ROUGE targets, 1–2 example questions per candidate answer; out-of-fold lists only | closed 5 | 6–10 GPU-h | +0.010–0.012 | Eedi 1st, RankZephyr, FIRST |
| 4 | **Copy-style 27–32B generator on all 8 subsets with Train+Val** (top-3 retrieved answers + 5 demos + "copy exact wording", r128/α256, 3 epochs, greedy); as ranker candidate and MBR member (Obiti's single model = 0.729) | all | 20–45 GPU-h | +0.01–0.03 | LanAnh, Obiti, Magic |
| 5 | **Reranker cross-fitting the Darius way** (per subset, folds over Val, ranker trains on Val OOF) + judge-revert of changed picks | closed 5 | 6–10 GPU-h | +0.003–0.006 | Darius `cv_ensemble.py`, `3_judge/judge.py` |
| 6 | **Cross-model MBR** (Qwen3.8 in flight, EXP-083/084); utility + small judge term on top-3; more models over more samples | Aka, Eng_Gha, Amh | in flight | judge ↑ | papers (2410.15021, 2410.02902) |
| 7 | Expected-metric pick over the top-5 (prob-weighted ROUGE) | closed 5 | CPU | +0.002–0.004 | grandmaster |
| 8 | Darius's Gemma recipe (dropout 0, r64/α128, 2 epochs, all subsets): +0.04 on Aka vs ours | Aka | ~6 GPU-h | Aka ↑ | Darius |

Needs a rules check before use: filling the three submission columns (TargetR1F1 / TargetRLF1 / TargetLLM) from different builds (10th place did; +0.001–0.003).

## Corrections to [[F-007]]
- Darius's cross-fitting rotates Val only; each fold model trains on all Train + 80% Val.
- His "three selectors" are three gating variants, each with a per-subset ranker.
- His 108-sample MBR covers Aka_Gha and Eng_Gha only.
