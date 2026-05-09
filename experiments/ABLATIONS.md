# Ablations

Systematic "what happens if we remove / change this one knob?" tables. Update after any experiment that changes a single variable relative to a prior run.

## Template

```
### Ablation: <what changed>
Base run: exp0XX
Variant runs: exp0YY, exp0ZZ

| variant | change | R1 | RL | AfroLM-BS | Judge | Combined | Δ combined |
|---------|--------|----|----|-----------|-------|----------|------------|
| base    | —      |    |    |           |       |          | +0.0000    |
| v1      |        |    |    |           |       |          |            |
```

## Planned ablation sweeps

1. **LoRA rank**: r ∈ {16, 32, 64}, alpha = 2·r, everything else held constant. Expect diminishing returns after 32 on small data.
2. **Decoding strategy**: greedy vs. beam-5 vs. beam-8+reranker. Reranker cost must be justified by ≥ 0.005 combined-score lift.
3. **Data mix**: comp-only vs. comp + medmcqa_translated vs. comp + synthetic_qa vs. all. Measure per-language to spot failure modes.
4. **Prompt template**: native-language labels vs. English "Question:/Answer:" vs. no labels. Expected: native labels win on AfroLM-BS but margin small.
5. **Per-language length policy**: fixed max=128 vs. per-language quantile policy. Expected: quantile policy halves truncation rate on Amharic.
6. **Judge model**: Aya-Expanse self-judge vs. Gemma-2-judge. Only matters if they disagree with final public LB — else irrelevant.
