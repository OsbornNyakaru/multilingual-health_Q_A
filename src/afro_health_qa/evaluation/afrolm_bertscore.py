"""AfroLM-based BertScore F1.

This is the single most important evaluation function in the repo. Zindi uses
``bonadossou/afrolm_active_learning`` as the embedding backbone for its BertScore
sub-metric (30% of the final score). Our local validation reliability depends
on matching Zindi's implementation closely.

Implementation:
    1. Encode the prediction and reference with AfroLM.
    2. Take last-hidden-state token embeddings; drop [CLS]/[SEP] and padding.
    3. Compute pairwise cosine similarity matrix between prediction tokens
       and reference tokens.
    4. Precision = mean of max-over-reference per prediction token.
       Recall    = mean of max-over-prediction per reference token.
       F1        = harmonic mean.

We cache reference embeddings on disk (keyed by SHA-1 of the text) because the
same val set is re-scored many times across experiments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

AFROLM_MODEL_ID = "bonadossou/afrolm_active_learning"
_CACHE_DIR = Path(".cache/afrolm_embeddings")


@dataclass
class AfroLMBertScoreResult:
    precision: float
    recall: float
    f1: float
    n: int


@lru_cache(maxsize=1)
def _load_afrolm():
    """Load AfroLM tokenizer + model once per process."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(AFROLM_MODEL_ID)
    model = AutoModel.from_pretrained(AFROLM_MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed_tokens(text: str, use_cache: bool = True) -> np.ndarray:
    """Return an (n_tokens, hidden_dim) array of last-layer hidden states,
    with special tokens and padding stripped."""
    import torch

    cache_key = _text_hash(text)
    cache_path = _CACHE_DIR / f"{cache_key}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path)

    tok, model, device = _load_afrolm()
    max_positions = int(getattr(model.config, "max_position_embeddings", 256) or 256)
    safe_max_length = max(8, min(256, max_positions - 2))
    enc = tok(text, return_tensors="pt", truncation=True, max_length=safe_max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # (seq_len, dim)
    mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
    hidden = hidden[mask]

    # Drop [CLS] (first) and [SEP] (last) if present — standard BertScore practice.
    if hidden.shape[0] >= 3:
        hidden = hidden[1:-1]

    # L2-normalise row-wise.
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    hidden = hidden / norms

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, hidden)

    return hidden


def _pair_bertscore(pred_emb: np.ndarray, ref_emb: np.ndarray) -> tuple[float, float, float]:
    """BertScore precision/recall/F1 for a single (prediction, reference) pair."""
    if pred_emb.size == 0 or ref_emb.size == 0:
        return 0.0, 0.0, 0.0
    sim = pred_emb @ ref_emb.T  # (n_pred, n_ref), already L2-normalised
    precision = float(sim.max(axis=1).mean())
    recall = float(sim.max(axis=0).mean())
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_afrolm_bertscore(
    predictions: list[str],
    references: list[str],
    use_cache: bool = True,
) -> AfroLMBertScoreResult:
    """Compute AfroLM-BertScore averaged over all rows.

    Args:
        predictions: generated strings.
        references: gold strings, same length as predictions.
        use_cache: cache embeddings on disk (keyed by SHA-1 of input text).

    Returns:
        AfroLMBertScoreResult with averaged precision, recall, f1, and n.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must match length: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return AfroLMBertScoreResult(0.0, 0.0, 0.0, 0)

    ps, rs, fs = [], [], []
    for pred, ref in zip(predictions, references):
        pred_emb = _embed_tokens(pred, use_cache=use_cache)
        ref_emb = _embed_tokens(ref, use_cache=use_cache)
        p, r, f = _pair_bertscore(pred_emb, ref_emb)
        ps.append(p)
        rs.append(r)
        fs.append(f)

    return AfroLMBertScoreResult(
        precision=float(np.mean(ps)),
        recall=float(np.mean(rs)),
        f1=float(np.mean(fs)),
        n=len(predictions),
    )


def score_afrolm_bertscore_per_example(
    predictions: list[str], references: list[str], use_cache: bool = True
) -> list[float]:
    """Per-row F1 scores — useful for the reranker and for per-language breakdowns."""
    out = []
    for pred, ref in zip(predictions, references):
        pred_emb = _embed_tokens(pred, use_cache=use_cache)
        ref_emb = _embed_tokens(ref, use_cache=use_cache)
        _, _, f = _pair_bertscore(pred_emb, ref_emb)
        out.append(f)
    return out
