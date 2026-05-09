"""Trainer callbacks: early stopping + per-language eval logging."""

from __future__ import annotations

from transformers import EarlyStoppingCallback, TrainerCallback


def build_early_stopping(patience: int = 3, threshold: float = 0.001) -> EarlyStoppingCallback:
    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )


class PerLanguageEvalCallback(TrainerCallback):
    """Logs per-language eval metrics to W&B at every eval step.

    Assumes the eval dataset has a ``language`` column preserved through
    tokenisation (add ``language`` to ``remove_unused_columns=False``).
    """

    def __init__(self, eval_df, predict_fn, languages: tuple[str, ...]):
        self.eval_df = eval_df
        self.predict_fn = predict_fn
        self.languages = languages

    def on_evaluate(self, args, state, control, **kwargs):
        import pandas as pd

        try:
            import wandb
        except ImportError:
            return

        rows = []
        for lang in self.languages:
            subset = self.eval_df[self.eval_df["Language"] == lang]
            if subset.empty:
                continue
            preds = self.predict_fn(subset["Question"].tolist(), lang)
            # Cheap proxy: average prediction length — real eval is run by trainer.
            rows.append({"language": lang, "n": len(subset), "mean_pred_chars": float(pd.Series([len(p) for p in preds]).mean())})
        if wandb.run is not None and rows:
            wandb.log({"per_language_proxy": pd.DataFrame(rows).to_dict("records")}, step=state.global_step)
