"""ROUGE-1 and ROUGE-L F1 using the ``rouge_score`` library.

We use RougeScorer directly (not ``evaluate``) because Zindi's scorer
uses the same library and our local scores need to line up.

Tokenization matters more than anything else here (see
``vault/facts/metric-replica.md``). rouge-score's DEFAULT tokenizer lowercases
and replaces every character outside ``[a-z0-9]`` with a space, so Amharic
(Ge'ez script) text tokenizes to ``[]`` and scores ~0 no matter how good the
answer is. Available tokenizers:

* ``"whitespace"`` (default) — ``str(text).strip().split()``, case-sensitive,
  punctuation stays attached, no stemming. Byte-for-byte what the official
  starter notebook uses; our best guess for the host grader (H-013).
* ``"unicode"`` — lowercase + ``\\w+`` (Unicode-aware) regex. Identical to the
  rouge-score default on ASCII text, but keeps Ge'ez / Akan letters and drops
  Ethiopic punctuation (``።`` ``፣`` ``፤`` ...). Diagnostic.
* ``"default"`` / ``"default_stem"`` — rouge-score's own tokenizer without / with
  the Porter stemmer. Kept only to reproduce legacy numbers (EXP-004 was scored
  with ``"default_stem"``); never use for decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

DEFAULT_TOKENIZER = "whitespace"
TOKENIZER_CHOICES = ("whitespace", "unicode", "default", "default_stem")

_UNICODE_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


class WhitespaceTokenizer:
    """Starter-notebook tokenizer: split on whitespace, nothing else."""

    def tokenize(self, text):
        if text is None:
            return []
        return str(text).strip().split()


class UnicodeWordTokenizer:
    """Lowercased Unicode ``\\w+`` tokens — the default tokenizer, minus the ASCII-only bug."""

    def tokenize(self, text):
        if text is None:
            return []
        return _UNICODE_WORD_RE.findall(str(text).lower())


@dataclass
class RougeResult:
    rouge1: float
    rouge_l: float
    n: int


def make_scorer(tokenizer: str = DEFAULT_TOKENIZER):
    """Build a ``RougeScorer`` for rouge1 + rougeL with the named tokenizer."""
    from rouge_score import rouge_scorer

    if tokenizer == "whitespace":
        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer())
    if tokenizer == "unicode":
        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False, tokenizer=UnicodeWordTokenizer())
    if tokenizer == "default":
        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    if tokenizer == "default_stem":
        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    raise ValueError(f"unknown tokenizer {tokenizer!r}; choose from {TOKENIZER_CHOICES}")


def score_rouge_per_example(
    predictions: list[str], references: list[str], tokenizer: str = DEFAULT_TOKENIZER
) -> list[tuple[float, float]]:
    """Return per-row (rouge1_f1, rougeL_f1). Useful for per-subset breakdowns."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must match length: {len(predictions)} vs {len(references)}"
        )
    scorer = make_scorer(tokenizer)
    out = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(str(ref), str(pred))  # rouge-score signature is (target, prediction)
        out.append((scores["rouge1"].fmeasure, scores["rougeL"].fmeasure))
    return out


def score_rouge(
    predictions: list[str], references: list[str], tokenizer: str = DEFAULT_TOKENIZER
) -> RougeResult:
    """Compute per-example ROUGE-1 F1 and ROUGE-L F1, return the averages.

    Args:
        predictions: list of generated strings.
        references: list of gold strings, same length as predictions.
        tokenizer: one of ``TOKENIZER_CHOICES``; default is the starter's whitespace split.

    Returns:
        RougeResult with averaged rouge1 F1, rougeL F1, and n.
    """
    rows = score_rouge_per_example(predictions, references, tokenizer=tokenizer)
    if not rows:
        return RougeResult(rouge1=0.0, rouge_l=0.0, n=0)
    return RougeResult(
        rouge1=float(mean(r[0] for r in rows)),
        rouge_l=float(mean(r[1] for r in rows)),
        n=len(rows),
    )
