# C. Literature check for the afro-health-qa pipeline

Legend: **ALIGNS** = supports what we or the winners already do. **REFINES** = keep the technique but change how it's done. **CONTRADICTS** = argues against it. [V] = title, authors and arXiv id checked by web search this session. [M] = cited from memory and not re-checked.

## 1. LLM rerankers (the closed-subset selection gap)

| Paper | Claim / evidence | Relation |
|---|---|---|
| Sun et al. 2023, *Is ChatGPT Good at Search? (RankGPT)*, arXiv:2304.09542 [V] | Listwise LLM reranking with instructions is competitive with supervised rerankers. A 440M model distilled from ChatGPT permutations beats a 3B supervised model on BEIR. | ALIGNS with an LLM chooser |
| Pradeep, Sharifymoghaddam, Lin 2023, *RankZephyr*, arXiv:2312.02724 [V] | A fine-tuned 7B listwise reranker matches or beats GPT-4. Training on shuffled input orders makes it robust to the initial ordering and to the number of candidates. | REFINES: train the LoRA chooser on shuffled candidate orders |
| Reddy et al. 2024, *FIRST: single-token listwise reranking*, arXiv:2406.15657 [V] (EMNLP'24) | The ranking can be read from the logits of the first output token. Inference is 50% faster with no loss on BEIR. Training uses a weighted LTR loss that favors the top ranks. | ALIGNS exactly with our option-letter logprob design. Also suggests training the LoRA with a top-weighted ranking loss on the letter logits, not plain CE on the gold letter |
| Tang et al. 2023, *Found in the Middle: Permutation Self-Consistency*, arXiv:2310.07712 [V] (NAACL'24) | Shuffle the list, rerank, and aggregate to the central ranking. Gains of 7–18% for GPT-3.5, 8–16% for LLaMA-2-70B and 34–52% for Mistral. | REFINES: one shuffle is not enough. Average over several orders |
| Zheng et al. 2023, *LLMs Are Not Robust Multiple Choice Selectors (PriDe)*, arXiv:2309.03882 [V] (ICLR'24) | Selection bias comes mainly from a token prior on the option IDs (A/B/C…). PriDe estimates that prior by permuting options on a few samples, then divides it out. Label-free and cheap. | REFINES: debias the letter logprobs |
| Robinson et al. 2022, *Leveraging LLMs for MCQA (MCSB)*, arXiv:2210.12353 [V] | Showing all options and scoring the answer symbol beats cloze scoring and removes the length and tokenization effects of the options, provided the model binds symbols to options well. | ALIGNS with letters over per-answer likelihood. Avoid scoring raw answer likelihood, which is length-biased against long Luganda answers |
| Qin et al. 2023, *Pairwise Ranking Prompting (PRP)*, arXiv:2306.17563 [V] (NAACL'24 Findings) | Pairwise prompting, run in both orders, is the most robust zero-shot formulation for moderate-size open LLMs. | REFINES: cheap tie-breaker for top-2 near-ties |
| Zhuang et al. 2023, *Setwise ranking*, arXiv:2310.09497 [V] | Setwise ("which is most relevant?") keeps most of the pairwise quality at far lower cost. | ALIGNS (top-5 setwise is our design) |
| Schlatt et al. 2024, *Distilling LLMs into Cross-Encoders (Rank-DistiLLM)*, arXiv:2405.07920 [V] | Cross-encoders distilled from LLMs beat ones trained on human labels, but do not reach the LLM teacher. | Low priority: we can afford the LLM at inference |
| Adeyemi, Oladipo, Pradeep, Lin 2023, *Zero-Shot Cross-Lingual Reranking with LLMs for Low-Resource Languages*, arXiv:2312.16159 [V] (ACL'24) | LLM reranking on African languages (Hausa, Somali, Swahili, Yoruba) works, but quality depends heavily on how multilingual the LLM is. English is still best. | CAUTION for Luganda: test Gemma vs Qwen on Luganda. An English-translated view may help |
| Zhang et al. 2025, *Qwen3 Embedding / Qwen3-Reranker*, arXiv:2506.05176 [V] | Pointwise yes/no-logit LLM rerankers (0.6B/4B/8B) are state of the art on multilingual retrieval. | Cheap additional GBDT feature |

## 2. Answer selection / FAQ answer reuse

- Sakata et al. 2019, *FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance*, arXiv:1905.02851 [V, pre-2023]: combining query–question (q-Q) and query–answer (q-A) scores beats either one alone. **REFINES**: make sure the GBDT has a strong q-A feature and not only q-Q. The listwise chooser is a q-A model.
- The 2023+ work found (e.g., arXiv:2311.17502, QAN + LLM knowledge augmentation [V, weak]) is generic community QA and adds little. Answer clustering has no strong paper in this set. In practice the gain comes from merging near-duplicate stored answers inside the top-k so that their votes pool, which is the self-consistency principle (Wang et al. 2022, arXiv:2203.11171 [M]).

## 3. MBR decoding

| Paper | Claim / evidence | Relation |
|---|---|---|
| Suzgun, Melas-Kyriazi, Jurafsky 2022, *Follow the Wisdom of the Crowd*, arXiv:2211.07634 [V] (ACL'23 Findings) | MBR over sampled candidates gives +3–7 ROUGE/BLEU across summarization, data-to-text and MT. | ALIGNS (our 13-sample MBR) |
| Bertsch et al. 2023, *It's MBR All the Way Down*, arXiv:2310.01387 [V] | Self-consistency, output ensembling ("post-ensemble") and reranking are all forms of MBR. | ALIGNS with cross-model MBR |
| Freitag, Ghorbani, Fernandes 2023, *Epsilon Sampling Rocks*, arXiv:2305.09860 [V] (EMNLP'23 Findings) | How candidates are sampled matters a lot. Epsilon sampling (drop tokens with p<ε, about 0.02) beats nucleus, top-k and ancestral sampling for MBR in human evaluation. | REFINES: winners varied temperature (0.7/1.0/1.3). Epsilon sampling at T≈1 is the better pool |
| Kamigaito et al. 2024, *Diversity Explains Inference Scaling Laws (MBR)*, arXiv:2410.15021 [V] (ACL'25) | MBR error = bias + diversity. Gains from more samples come from diversity. | ALIGNS with cross-model and multi-temperature pools. Diversity matters more than raw sample count |
| Kovacs, Deutsch, Freitag 2024, *Mitigating Metric Bias in MBR*, arXiv:2411.03524 [V] (WMT'24) | MBR with one utility over-fits that metric. An ensemble of utilities wins in human evaluation. | REFINES: our score is partly an LLM judge, so a pure-lexical utility under-serves that 26% |
| Wu et al. 2024, *Better Instruction-Following Through MBR*, arXiv:2410.02902 [V] (ICLR'25) | MBR with a reference-based LLM judge as the utility beats greedy, best-of-N with a reference-free judge, and lexical/embedding MBR. Small judges can supervise 70B generators. | REFINES: add a judge term to the utility |
| Fernandes et al. 2022, *Quality-Aware Decoding*, arXiv:2205.00978 [V] | Two-stage QE reranking followed by MBR beats MAP decoding. | Optional: QE filter, then MBR |
| Finkelstein et al. 2023, *MBR and QE Finetuning*, arXiv:2309.10966 [V] (ICLR'24) | Fine-tuning on MBR-selected outputs keeps most of the MBR gain at greedy-decoding cost. Using a stronger teacher beats human references. | ALIGNS with the winners' RAFT self-distillation. Train on MBR medoids, not raw gold |

## 4. Retrieval for African languages

- Chen et al. 2024, *BGE M3-Embedding*, arXiv:2402.03216 [V]: dense, sparse and multi-vector scoring from one model, and the hybrid is best. **ALIGNS** with the winners' BM25 expansion for Luganda. BGE-M3's own sparse and ColBERT scores are free extra GBDT features.
- Uemura, Zhang, Adelani 2025, *AfriMTEB and AfriE5*, arXiv:2510.23896 [V] (EACL'26): mE5 adapted to African languages by cross-lingual contrastive learning is the best open-weight model on AfriMTEB. **REFINES**: add AfriE5 similarity as a candidate source or feature. Check that Luganda, Twi and Amharic are covered.
- Adeyemi et al. 2024, *CIRAL* (SIGIR'24 resource, ciralproject on GitHub) [V, no arXiv id confirmed]: an African-language CLIR collection. BM25 + dense hybrids and query translation are strong baselines there.
- Olatunji et al. 2024, *AfriMed-QA*, arXiv:2411.15640 [V] (ACL'25): 15k pan-African English medical QA items. A possible source of RAFT/demo data for Eng_Gha.

## 5. Fine-tuning for RAG, copying and length

- Zhang et al. 2024, *RAFT*, arXiv:2403.10131 [V]: training with distractor documents plus verbatim citation of the oracle improves in-domain RAG. **ALIGNS** with the winners' RAFT and the 2nd/3rd "copy exact wording" instruction. Copying is directly rewarded by ROUGE.
- Yang et al. 2024, *Self-Distillation Fine-Tuning (SDFT)*, arXiv:2402.13669 [V] (ACL'24): rewriting targets in the model's own distribution reduces forgetting and matches or beats plain SFT. **ALIGNS** with RAFT self-distillation.
- Dubois et al. 2024, *Length-Controlled AlpacaEval*, arXiv:2404.04475 [V]: LLM judges prefer longer outputs. Controlling for length raises the correlation with Chatbot Arena from 0.94 to 0.98. **Tension**: ROUGE-F penalizes length mismatch while the judge rewards length. That supports per-subset length anchors set slightly above the median gold length, tuned on CV for the combined metric. See also Zheng et al. 2023, *Judging LLM-as-a-Judge*, arXiv:2306.05685 [M], on verbosity and position bias.
- Liu et al. 2023, *Lost in the Middle*, arXiv:2307.03172 [M]: position effects in long contexts. Put the retrieved demos closest to the question.

No paper found contradicts the core design. The main corrections are to how the LLM chooser is run (PSC, PriDe) and to how the MBR pool and utility are built (epsilon sampling, an ensemble utility).

## Top 8 recommendations (ranked)

| # | Action | Expected effect | Subset | Cost (1×95 GB GPU) | Key paper |
|---|---|---|---|---|---|
| 1 | **Permutation-averaged letter logprobs.** For the top-5, run all 5 cyclic shifts (or 5 random shuffles) and average each candidate's probability. Use mean, max, std and rank as **LambdaRank features**, not as a hard override. | Removes the position-prior noise that otherwise swamps a 5-way choice. PSC-style gains of +7–18% relative. This is the most likely route to closing part of the 61→76 (Luganda) and 78→87 (Eng_Uga) gaps | Luganda, Eng_Uga, Swa_Ken, Eng_Ken, Eng_Eth | Prefill-only, 5× passes. Minutes to under 1 h in vLLM with prefix caching | 2310.07712, 2406.15657 |
| 2 | **PriDe calibration.** Estimate the per-letter prior on about 5% of queries (all permutations) and divide it out before the softmax | Cheap fix for a bias of a few points on A vs E. Stacks with #1 | Closed | Negligible | 2309.03882 |
| 3 | **Cross-fitted LoRA chooser.** QLoRA on Gemma-4-31B (or Qwen3.8-27B) over train queries whose gold answer is in the top-5, with randomly shuffled order per example. Loss: CE on the gold letter, or FIRST's top-weighted LTR. Use the same 5 folds as the cross-encoder so the GBDT features are out-of-fold | Fine-tuned listwise rankers beat zero-shot by a wide margin (RankZephyr ≈ GPT-4). Teaches Luganda-specific answer style | Closed, mostly Luganda | About 2–4 h per fold at 31B QLoRA, so 10–20 h for 5 folds. Use 3 folds or a smaller model if pressed | 2312.02724, 2406.15657 |
| 4 | **Collapse near-duplicate answers** in the top-k before the chooser (normalized-text or ROUGE-L>0.9 clusters). Sum the cluster scores and expose the cluster size as a feature. Refill the freed slots from rank 6+ | Recovers gold answers that sit outside the top-5. Pools votes from duplicate questions that share one answer | Eng_Uga, Luganda | CPU only | 2203.11171 (principle), 1905.02851 |
| 5 | **Epsilon-sampling MBR pool** (ε≈0.02, T=1) mixed across Gemma and Qwen LoRAs, with more models rather than more samples per model | Better medoid quality per sample. Diversity drives MBR gains | Akan, Eng_Gha, Amharic | Same generation budget | 2305.09860, 2410.15021 |
| 6 | **Ensemble MBR utility**: 0.74·(R1+RL)/2 lexical plus a small judge term (for example a reference-based LLM judge on the top-3 lexical medoids only) | Targets the 26% judge share without losing ROUGE. Reduces reward hacking of a single utility | Open subsets | Judge on 3 candidates per question, under 1 h | 2410.02902, 2411.03524 |
| 7 | **MBR self-distillation round**: fine-tune the generator on its own MBR medoids from train-fold questions (filtered by ROUGE to gold) | Keeps most of the MBR gain in single samples and sharpens the next MBR pool | Open subsets | One extra QLoRA epoch, about 3–6 h | 2309.10966, 2402.13669 |
| 8 | **Add AfriE5 and BGE-M3 sparse/ColBERT scores** as candidate sources and features. Raise the top-k recall that the chooser sees | A small recall gain on Luganda and better demo retrieval for Akan/Amharic | Luganda, open-subset demos | Embedding pass, under 30 min | 2510.23896, 2402.03216 |

Also tune the per-subset length anchors against the combined metric, since the judge favors length and ROUGE-F penalizes mismatch (2404.04475). Use PRP in both orders only as a tie-breaker when the GBDT margin between the top two is small (2306.17563).
