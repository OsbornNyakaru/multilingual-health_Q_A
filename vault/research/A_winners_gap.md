# A. Winners' techniques we have not tried yet (audit, 2026-10-09)

Sources, all read from the code (cloned to scratchpad/clones; nothing executed, no weights or outputs used):
- Darius `DariusTheGeek/Multilingual-Health-QA-ITU-Zindi-Challenge` (code and Solution_Documentation.pdf)
- `yigagilbert/hash-challenge-review`, which holds the code for #1 Magic, #2 LanAnh, #3 Nailiw, #5 Somun, #7 Obiti and #10 David, plus the organisers' scorer.

Our current position: private 0.6830. Darius's public score breaks down as R1 0.7187 / RL 0.6514 / judge 0.8522. Compared with ours, most of the gap is ROUGE (about 0.03 LB). The judge accounts for about 0.004.

## 1. Corrections to our notes

1. **How Darius cross-fits the reranker** (`2_reranker/cv_ensemble.py --fold-val-only`). There is one cross-encoder (CE) per subset, trained on question–question pairs. The label is the ROUGE of the candidate's answer against the gold; settings are 2 epochs, lr 2e-5, batch size 32, max_len 512.
   - Only Val rotates through the 5 folds. Every fold model trains on 100% of Train plus 80% of Val.
   - The LightGBM ranker is trained only on the Val out-of-fold (OOF) scores. The test score is the mean of the 5 fold models.
   - Folding Train and Val together regressed. Mixing feature sources (a Train-only CE for training features, a Train+Val CE for test features) cost **−0.0105 LB**.
2. **"Three selectors" are three gating strategies, not three models.** Each script trains a **separate LambdaRank per subset** (loop over `s` in `v30_direct.py` / `v36_nogen.py`). Our `scripts/ltr.py` trains one pooled model with `subset_code`.
3. **Darius's ROUGE tokenizer.** For features and MBR he used a Unicode-aware, lowercased tokenizer (`\p{L}+|\p{N}+`, `common/metrics.py`), not whitespace. The organisers' scorer (`reurgent…/rouge_*_cached_par.py`) **is** whitespace and case-sensitive, so our replica is right.
4. **Darius regenerates only Aka_Gha and Eng_Gha** with the 108-sample cross-model MBR. Amh_Eth "passes through" an older base. gen7454 trains on all 8 subsets; raft_r1 and mg_2226 train only on the two Ghana subsets. The RAFT data builder is **not in the repo**, so its recipe is unknown.
5. **Rank labels in the review repo are public-board ranks** (table dated 2026-06-30; Obiti's notebook says "7th place"; Magic claims private 0.730865 vs the API's 0.729017). We keep the Zindi API podium.
6. **The three target columns are scored independently.** Our F-001 says all three hold the same answer, but that is only a convention. #10 David (`scripts/build_v26.py`) shipped R1/RL columns from one build and TargetLLM from another.
7. **ID hashes carry no signal** (David's "topic family" feature): unique per subset; same-hash Aka F1 0.211 vs 0.207 random. Drop it.
8. **Darius tested an LLM as chooser, and it lost −0.0055 LB.** As a *feature* it helped (`3_select/llm_probe.py`). This affects our running listwise-chooser probe.
9. **The organisers' judge** is gpt-4o-mini, batches of 20, 1–5 scale. It sees the **reference and the prediction** (accuracy, completeness, fluency).

## 2. Gap list

Gains are LB deltas, estimated as subset share × proxy delta × 0.74 (+0.26 × judge delta). GPU hours assume one 80GB GPU.

| # | Technique (where in their code) | Subsets | Est. LB gain | GPU-h | Code | Gain per GPU-h |
|---|---|---|---|---|---|---|
| 1 | **Per-column submission**: R1-medoid → TargetR1F1, RL-medoid → TargetRLF1, judge-oriented pick → TargetLLM (David `build_v26.py`) | all | +0.001–0.003 | 0 | ~40 lines | CPU |
| 2 | **Upgraded LTR features** (Darius `1_pools/sel_features.py`, `3_select/features.py:add_relative_feats`, `verify.py:_echo_one`), detailed below | Eng_Uga, Lug, Swa, Eng_Ken | +0.004–0.010 | 0 | ~200 lines | CPU |
| 3 | **Generated answer as an LTR candidate, gated** (`v36_nogen.py`) | Eng_Eth (49% in bank), Eng_Uga | +0.001–0.003 | 0 if test generations exist | ~60 lines | CPU |
| 4 | **LLM "directness" probe as a feature** (`3_select/llm_probe.py`), not a chooser | closed subsets | +0.002–0.006 | 2–4 | ~90 lines | high |
| 5 | **Fold-Val-only 5-fold CE cross-fit**, with raw CE scores fed to LTR | closed subsets | +0.002–0.004 | 6–10 | ~60 lines | medium-high |
| 6 | **Judge-revert pass** (`3_judge/judge.py`) | changed rows | +0.0005–0.002 | 0.5 | ~50 lines | medium-high |
| 7 | **MBR utility variants** on existing samples: R1+R2 F (Magic `build_ensemble.py`), uni+bigram set-F1 (LanAnh `consensus.py`), chrF 2–6 RRF (Nailiw `lib/rankfuse.py`) | Aka, Eng_Gha, Amh | +0–0.002 | 0 | ~50 lines | CPU |
| 8 | **More samples and members**: 36 per member (n=12 × T 0.7/1.0/1.3, top_p 0.95, `gen_samples.py`) plus early-checkpoint members (Magic's "early-tap") | Aka, Eng_Gha | +0.001–0.003 | 2–4 | config only | medium |
| 9 | **RAG-copy SFT of a 27–32B model on all 8 subsets with Train+Val** (Obiti notebook; LanAnh `pipeline/retrieve.py`, `demo_prep.py`; Magic `build_fewshot_*_k5.py`, `build_v8_k5_fewshot.py`) | all | **+0.010–0.030** | 20–45 | ~250 lines | medium (biggest absolute) |
| 10 | **Darius's gemma recipe**: dropout 0, r64/α128, lr 1e-4, cosine, 2 epochs, all subsets, no RAG, length-anchored prompt (`common/prompt_desc.py`, `train_unsloth.py`) | Aka, Amh | +0.004–0.006 | 15–25 | config | low-medium |
| 11 | **Expanded Luganda pool** (BGE@100 ∪ BM25@50; `lug_features.py`) | Lug | +0.000–0.003 | <1 | ~80 lines | low (recall is not our bottleneck: top-5 already 76%) |
| 12 | **RAFT member** (r32/α64, lr 5e-5, 1 epoch on the merged primary) | Aka, Eng_Gha | unknown | 10 | recipe missing | low |

### Item details

**#2 LTR features.** Add these to `scripts/ltr.py`:
- Raw CE/reranker and cosine scores, alongside the ranks we already use.
- Query-level confidence: `ce_max`, `ce_gap`, `top1cos`, `cos_gap`.
- Within-query relative versions of every signal: z-score, descending rank, gap to max, is-argmax.
- `consensus` (mean ROUGE to the other 19 candidates) and `maxsim_other`.
- **CanonEcho**: for each candidate, the mean, max and count of pool members with ROUGE ≥ 0.8 at depths 22, 50 and the full ~128 pool.
- A second generator's agreement.
- Per-subset LambdaRank with linear `label_gain` 0..10, n_est 500, lr 0.03, 31 leaves, min_child 30, subsample and colsample 0.8.

Darius went 0.705 → 0.7285 with these plus Train-scale CE and CV; we already have Train-scale CE. Our `lfreq` counts exact-text frequency in the bank; echo counts near-duplicates in the pool, a different signal. Gold-pick rates (78% vs 87% Eng_Uga, 61% vs 76% Lug) say the gap is selection.

**#3 Generated answer as candidate.** On Eng_Uga and Eng_Eth, add the generated answer to the pool with `is_gen` and agreement features. Train the ranker; where it picks retrieval, re-pick among retrieval candidates only; where it picks generation, freeze. Darius froze generation on 45 of 744 Eng_Uga rows and 13 of 60 Eng_Eth rows. This only makes sense with a model that writes good closed-subset answers (#9; our 7B EXP-038 is weak).

**#4 LLM probe prompt.** The exact prompt asks: "0–100, how DIRECTLY and SPECIFICALLY does this answer address THIS EXACT question… longer is NOT better". Settings: greedy, max_tokens 6, question and answer each capped at 1400 characters. Combined with CE-ft and echo, Darius reports +0.04 OOF on Swa and +0.027 on Eng_Ken. **Repurpose our running listwise-chooser probe into this per-candidate score.**

**#5 Cross-fit.** The rule: the training-feature model and the test-feature model must come from the same class. Our LTR uses picker ranks partly to dodge calibration. Cross-fitting lets us also feed raw scores.

**#6 Judge revert.** Only for rows where a new selector changed the pick: label both the new and old answer ALIGNED/MISMATCH with a 27–31B model, and revert if new = MISMATCH and old = ALIGNED. This fixes PrEP/PEP-style swaps (27 reverts for Darius). Use it for any LTR change.

**#7 MBR utilities.** Magic's R1+R2 uses the default `rouge_score` tokenizer (ASCII), which drops Ge'ez. For Amh use chrF or a whitespace variant.

**#9 RAG-copy SFT recipe:**
- **Prompt.** Language tag, then this instruction (Magic "v8", best single model at 0.7233 public): "The retrieved contexts are your source of truth — copy or paraphrase their exact phrasing… Reply in the same language and script… no disclaimers". Then the top-3 AfriE5 *answers* as evidence, then K=5 (also 3/4/7) same-subset AfriE5 question–question neighbour Q/A demos (leave-self-out by ID), then the question.
- **Training.** LoRA r128/α256, dropout 0–0.1, all linear layers, lr 1.5–2e-4, cosine, warmup 3–5%, effective batch 64, 3 epochs on Train+Val (about 1711 steps).
- **Checkpoint and decoding.** Keep a checkpoint every 100 steps and ship the 1200–1700 "peak". Decode greedy with `enable_thinking=False`; Magic reports a 0.10–0.25 drop with thinking on.
- **Evidence it works.** Obiti, with one such model and no selection or ensemble, scored **0.72902 private, equal to 1st**. Somun trained a 27B r64 on one 80GB GPU in about 14–20 h to ~2.1 epochs.
- **How to deploy.** Feed it to the LTR as a candidate plus agreement feature (#3) rather than replacing selection. Also add it as a cross-model MBR member for Aka, Eng_Gha and Amh, alongside gemma and Qwen3.8.
- **Gating.** Validate with a Train-only (or train_core) run on Val first, then retrain on Train+Val. Our EXP-083 Qwen3.8-27B was trained on open subsets only, with RAG k=3 and no demo block, so it does not cover this.

## 3. Risks

- **#1:** the rules don't address mixing columns, and none of the top-4 did it. It is fine for post-deadline learning, but confirm before any claim.
- **#2, #4, #5:** leakage. Every OOF feature must come from the same model class as its test counterpart (Darius's −0.0105 lesson).
- **#9:** overfits past ~2.8 epochs (pick checkpoints on held-out data); hybrid Qwen3.5+ needs `max_num_seqs` set (cf. EXP-082); 6–8k-token prompts make QLoRA costly; copying a paraphrased neighbour can hurt the judge on PEP/PrEP-type rows, so keep the judge-revert.
- **#10:** Darius's single LoRA was worse than ours on Eng_Gha (−0.007), so route per subset.

## 4. Priority order (gain per GPU-hour)

1. Per-column submission (#1). CPU.
2. LTR feature upgrade: CanonEcho, relative features, raw scores, per-subset LambdaRank (#2). CPU.
3. LLM directness probe as a feature, replacing the chooser probe (#4). About 3 GPU-h.
4. Fold-Val-only CE cross-fit with consistent calibration (#5), plus the judge revert (#6).
5. RAG-copy 27B SFT on all subsets with Train+Val (#9). Expensive, but it is the largest single lever. It also enables #3 and a stronger cross-model MBR.
