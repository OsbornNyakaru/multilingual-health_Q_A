# Experiment Log

Append-only. Every submission gets a row. **No exceptions.** Never submit to Zindi without first appending here.

Column definitions:

- **ID** — `expNNN` (zero-padded, monotonic).
- **Date** — ISO date of the submission.
- **Hypothesis** — one sentence. Must state what you expect to happen.
- **Model** — backbone + adapter name.
- **Data mix** — `comp` + any augmentation sources used.
- **Key hyperparams** — LR, rank, epochs, decoding strategy (short form).
- **Local R1 / RL / AfroLM-BS / Judge** — local held-out scores (not val — the 5% held-out slice).
- **Local combined** — weighted mean using Zindi weights (0.25, 0.25, 0.30, 0.20).
- **Public LB** — fill in after Zindi returns the score.
- **Submission file** — path under `submissions/`.
- **Notes** — what surprised you; what to try next.

| ID | Date | Hypothesis (one sentence) | Model | Data mix | Key hyperparams | Local R1 | Local RL | Local AfroLM-BS | Local Judge | Local combined | Public LB | Submission file | Notes |
|----|------|---------------------------|-------|----------|-----------------|----------|----------|-----------------|-------------|----------------|-----------|-----------------|-------|
| exp000 | 2026-04-24 | Scaffold repo; no model change expected. | — | — | — | — | — | — | — | — | — | — | Bootstrap commit. |
