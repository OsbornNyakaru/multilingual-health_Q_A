"""Tokeniser audit: chars-per-token per language per model.

Why: Latin-BPE tokenisers (Llama-family) often explode on Amharic (1-2 chars/tok)
vs. on English (~4 chars/tok), making max_length settings deceptive. This audit
tells us whether our ``max_length: 1024`` actually fits the longest Amharic
answers in the training set.

Run:
    python -m afro_health_qa.data.tokeniser_audit --output docs/TOKENISER_AUDIT.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = (
    "CohereForAI/aya-expanse-8b",
    "google/gemma-2-9b-it",
    "Jacaranda/UlizaLlama-7B",
)
DEFAULT_LANGS = ("swa", "lug", "aka", "amh")


@dataclass
class AuditRow:
    model: str
    language: str
    n_samples: int
    chars_per_token_p50: float
    chars_per_token_p05: float
    tokens_p50: int
    tokens_p95: int
    tokens_max: int


def audit(
    df,
    models: tuple[str, ...] = DEFAULT_MODELS,
    languages: tuple[str, ...] = DEFAULT_LANGS,
    text_col: str = "Response",
    lang_col: str = "Language",
    max_samples: int = 500,
) -> list[AuditRow]:
    """Compute chars/token and token-length distributions per (model, language)."""
    import numpy as np
    from transformers import AutoTokenizer

    rows: list[AuditRow] = []
    for model_id in models:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as exc:
            print(f"[skip] {model_id}: {exc}")
            continue
        for lang in languages:
            subset = df[df[lang_col] == lang][text_col].astype(str).tolist()
            if not subset:
                continue
            subset = subset[:max_samples]
            token_counts = []
            cpt = []
            for txt in subset:
                ids = tok(txt, add_special_tokens=False)["input_ids"]
                n = len(ids)
                if n == 0:
                    continue
                token_counts.append(n)
                cpt.append(len(txt) / n)
            if not token_counts:
                continue
            rows.append(
                AuditRow(
                    model=model_id,
                    language=lang,
                    n_samples=len(token_counts),
                    chars_per_token_p50=float(np.percentile(cpt, 50)),
                    chars_per_token_p05=float(np.percentile(cpt, 5)),
                    tokens_p50=int(np.percentile(token_counts, 50)),
                    tokens_p95=int(np.percentile(token_counts, 95)),
                    tokens_max=int(max(token_counts)),
                )
            )
    return rows


def render_markdown(rows: list[AuditRow]) -> str:
    lines = ["# Tokeniser audit", "", "Chars per token and token counts per model × language.", ""]
    lines.append(
        "| Model | Lang | N | cpt p50 | cpt p05 | tokens p50 | tokens p95 | tokens max |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.model} | {r.language} | {r.n_samples} | "
            f"{r.chars_per_token_p50:.2f} | {r.chars_per_token_p05:.2f} | "
            f"{r.tokens_p50} | {r.tokens_p95} | {r.tokens_max} |"
        )
    lines.append("")
    lines.append(
        "**Action items:** if tokens p95 > 80% of `max_length`, bump the config. "
        "If cpt p05 < 1.5 for any language, flag in KNOWN_ISSUES."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/raw/Train.csv")
    parser.add_argument("--output", default="docs/TOKENISER_AUDIT.md")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.train_csv, dtype=str).fillna("")
    rows = audit(df, max_samples=args.max_samples)
    md = render_markdown(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
