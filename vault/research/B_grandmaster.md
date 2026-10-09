# B. Grandmaster strategies from related Kaggle/Zindi competitions

Verification key: **[V-code]** = read in the winner's repo code/config; **[V-doc]** = winner README / paper / host blog; **[U]** = could not reach the primary post (Kaggle discussion pages and Zindi threads do not render for fetch). Kaggle discussion links are given for reference.

## 1. What the winners actually did

### Eedi – Mining Misconceptions (Kaggle 2024, 1st, Raja Biswas) — closest match to our closed-subset problem
Source: [repo rbiswasfc/eedi-mining-misconceptions](https://github.com/rbiswasfc/eedi-mining-misconceptions), [writeup 551688](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688).
- Pipeline: LLM-embedding retrievers (e5-mistral-7b, bge-en-icl, Qwen2.5-14B) give top 32–64, then **pointwise** rerankers (Qwen2.5-14B/32B), then a **listwise** reranker (Qwen2.5-72B) over the **top 5**. [V-doc]
- Listwise format [V-code]: system "Pick the misconception that explains the incorrect answer most specifically"; candidates labelled `A.`–`E.`; final line "Which ... ? (A, B, C, D, or E)". Score = **logits of the 5 letter tokens at the last position** (no generation), softmax over 5.
- Training [V-code]: 4-bit QLoRA, **r=64, alpha=128**, all attention+MLP projections, **1 epoch**, LR 1e-6 base / 1e-5 lora_A / 5e-5 lora_B, AdamW8bit, max_len 2048. Loss = **0.5·CE(gold letter) + 0.5·soft-CE vs teacher logits** (distillation from the earlier rankers).
- **Few-shot "reference examples" per candidate** [V-code]: for each candidate label, sample another *training* question that had that label, and show 0–4 such (question → label) demos in the prompt. Demos come only from the training fold, and the query itself is excluded.
- **Rationales**: three fine-tuned "reasoners" (7B/14B/32B, trained on 6k Claude-3.5 CoTs) each write a "Thought:" line. At train time 0–3 thoughts are randomly dropped; at inference all 3 are shown. [V-code]
- **TTA**: second pass with the **candidate order reversed**; per-candidate probabilities averaged. [V-code]
- Pointwise rankers: group size 16 (1 positive + 15 negatives, softmax over the group), r=64, distillation at T=0.95. [V-code] Inference uses AWQ-quantized merged models in vLLM. [V-doc]
- 5-fold splits, ~1.8k real + ~10.6k synthetic examples. [V-doc]
- 2nd place: listwise reranking of the top 25 with Qwen2.5 + CoT. [V-doc, host blog only: [Eedi](https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions)]

### LMSYS Chatbot Arena (Kaggle 2024, 1st) and WSDM Cup Multilingual Arena
- 1st LMSYS ([repo](https://github.com/shyoulala/LMSYS_BlackPearl)): Llama3-70B and Qwen2-72B were trained **5-fold**. Their OOF probabilities were **distilled (KL + CE + cosine)** into Gemma2-9B, with LoRA averaged across folds. [V-doc] The inference code has a **TTA that swaps the A/B order**. [V-code] Per [Nebius](https://nebius.com/blog/posts/chatbot-arena-competition-review), the same distillation recipe also drove top WSDM results. [V-doc secondary]
- 3rd LMSYS ([repo](https://github.com/rbiswasfc/lmsys-arena), [post 527766](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)): details of pseudo-labels and swap-TTA are [U].

### AIMO-2 (Kaggle 2025, 1st NemoSkills) — selection among sampled generations
- [Paper](https://arxiv.org/html/2504.16891) describes **GenSelect**: the model sees **16 candidate solutions** (as summaries) and picks one; **majority@8 over GenSelect picks**. It gained about 10 pp over majority@64 internally, **but was not used in the Kaggle submission**, which used majority voting. [V-doc]

### MAP – Charting Student Math Misunderstandings (Kaggle 2025)
- From the host [case study](https://the-learning-agency.com/the-cutting-ed/article/case-study-math-misconceptions-competition) [V-doc secondary]: 1st place used shared-prefix attention, scoring all label suffixes in one pass. 2nd used **soft labels averaged over models**. Ensembles ran **72B only on low-confidence rows**, and **multi-seed validation** was called the most critical factor.

### Zindi (Kenya Clinical Reasoning, prior health QA) — no transferable winner write-ups found
The Kenya Clinical Reasoning threads I reached ([27427](https://zindi.africa/competitions/kenya-clinical-reasoning-challenge/discussions/27427), 27425, 27428, 27431) are complaints about the ROUGE metric, not solutions. I found no public top-3 repos. [U] The best Zindi-specific reference remains our own competition's winners, listed in the brief.

Not verified / not used: the LLM Science Exam 1st-place and LECR 1st-place write-ups. Their posts were not fetchable and no winner repo was found.

## 2. Ranked techniques mapped to our pipeline

Rough impact model: a closed-subset pick fixed from wrong to right is worth about +0.6–0.8 on that row (ROUGE goes from ~0.3 to 1.0, plus the judge). So +1 pp pick accuracy on Eng_Uga (28.4% of test) ≈ +0.002 LB, and on Lug_Uga (14.3%) ≈ +0.001.

**1. Fine-tune the listwise chooser (Eedi recipe) instead of running it zero-shot.** Closed subsets.
Use Gemma-4-31B QLoRA (we already have the trainer), r=64/α=128, 1 epoch, low LR as in the Eedi config. Input is our LambdaRank OOF top-5 with letters A–E; score the letter-token logits at the last position. Train only on rows where gold is in the top 5, using **random candidate permutations** as augmentation.
Impact: the Eedi listwise 72B was the last and strongest stage. Closing ~⅓ of the top-5-vs-picked gap (Eng_Uga +3 pp, Lug_Uga +5 pp) gives about **+0.010–0.012**.
Cost: ~6–10 GPU-h per training run. Cross-fitting OOF lists for 5 folds is about 5× that, so use 2–3 folds.

**2. Per-candidate "reference examples" from the answer bank.** Closed subsets. Pairs with #1; cheap to add.
In our data the same stored answer serves several train questions. Next to each candidate letter, show 1–2 *other* train questions whose gold was that answer (fold-train only, never the query). This is the direct analogue of the Eedi few-shot block, and it turns "does this answer fit?" into "is my question like these questions?", which is easier, especially for Luganda.
Cost: only longer prompts (+30–50% tokens).

**3. Order-permutation TTA, then feed the result to LambdaRank as features rather than replacing it.** All closed subsets.
Run 2–5 cyclic shifts or reversals and average per-candidate probabilities (Eedi reversal, LMSYS swap). Add `llm_p`, `llm_rank` and `llm_p − max_other` as features to the existing LambdaRank, cross-fit. Eedi stacked rankers in sequence rather than switching from one to the other.
Applies to the zero-shot test you are about to run, too: without permutation TTA the position bias of a 5-option prompt is large.
Cost: ×2–5 inference. A 31B prompt-only forward pass over ~5k test+OOF rows takes under 1 GPU-h per pass in vLLM with prefix caching.

**4. Soft, metric-aware targets (distillation).** Closed subsets.
Instead of a one-hot gold letter, train against softmax(0.37R1 + 0.37RL of each candidate vs gold, T≈0.1), mixed 50/50 with CE as in Eedi's `0.5·CE + 0.5·distill`. Near-duplicate stored answers then stop being penalised as wrong. This matches the 1st-place "graded targets" idea in our own competition and MAP's 2nd-place soft labels.
Cost: free on top of #1.

**5. Expected-metric (MBR) decision over the top 5.** Closed subsets. CPU-only, no GPU, minutes to run.
Pick argmax_i Σ_j p_j·score(c_i, c_j) using the calibrated LambdaRank (+LLM) probabilities, rather than argmax p. When p is split between two paraphrases, this picks the hedge that maximises expected ROUGE. It mirrors 1st place's MBR on the generation side.
Impact: small but positive, about +0.002–0.004. Tune a confidence gate on OOF.

**6. Confidence-gated cascade.** Closed subsets.
Following the MAP practice of running 72B only on low-confidence rows: run the costly LLM chooser (with TTA and longer prompts) only where the LambdaRank margin p1−p2 is below a threshold, and keep LambdaRank's pick elsewhere. This also limits the damage on the ~78% of rows LambdaRank already gets right.
Cost: cuts #1–#3 inference by about 50–70%. Tune the threshold on OOF per subset.

**7. Rationale or gloss "Thought:" lines.** Lug_Uga first.
Eedi fed three reasoner CoTs, randomly dropped during training. Our cheap analogue: have Gemma write a one-line English gloss of the Luganda question ("Thought: the user asks about…") and put it before the candidates. Keep dropout during training so the model does not over-trust the gloss. Also check whether Lug_Uga questions are translations of Eng_Uga ones. If they are, an English-side match is an extra feature.
Cost: one short generation pass, ~1 GPU-h.

**8. Generative selection for the open subsets (GenSelect).** Aka_Gha, Eng_Gha, Amh_Eth.
Show the LLM 8 of our 13 MBR samples, ask it to pick the best, repeat over several random subsets or orders, and vote. Use the vote share as a second utility, e.g. MBR-ROUGE + λ·GenSelect, where λ is tuned on OOF. ROUGE-MBR already optimises the 74% ROUGE part; GenSelect targets the **26% LLM-judge** part.
Evidence is only moderate: NemoSkills did not use it on Kaggle.
Cost: 2–4 GPU-h.

## 3. Validation hygiene (applies to all of the above)
- Train and evaluate the chooser on **cross-fit OOF candidate lists**. In-fold top-5 lists are easier than test lists, and that is the main leakage risk.
- Draw reference-example demos only from the training fold (as Eedi does), excluding the query.
- Report pick accuracy **per subset over ≥3 seeds or folds** before trusting gains below 0.003 (the MAP lesson).
- Optional pseudo-labelling: add high-confidence test picks (top decile of margin) as extra "questions answered by this answer" demos for #2. These affect inference only, so retraining is not needed.
