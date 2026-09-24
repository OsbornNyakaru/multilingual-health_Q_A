"""Dense-retrieval baseline on the held-out slice (the report's central evidence).

For each held-out question we retrieve the nearest *training* question — within
the same language subset, never from the held-out slice itself — using a dense
multilingual embedder, and copy that neighbour's answer as the prediction. We
then score those predictions with the frozen, leaderboard-matching ROUGE.

This replaces the crude token-overlap TF-IDF lower-bound recorded in
``autoresearch_nlp/COMPETITION_INTEL.md`` with a real measurement, and validates
the central thesis of the retrospective: dense retrieval over the labelled pool
is a strong baseline on this competition.

Primary embedder: BAAI/bge-m3 (the model the 11th-place solution used).
Fallback (if BGE-M3 cannot download / is too heavy): intfloat/multilingual-e5-small,
clearly recorded in the output JSON so the report can label it as a proxy.

Usage:
    python scripts/retrieval_baseline_bge.py [--model bge-m3|e5-small] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from afro_health_qa.evaluation.rouge import score_rouge  # noqa: E402
from afro_health_qa.evaluation.combined import W_R1, W_RL  # noqa: E402

OUT_DIR = REPO / "docs" / "competition_report" / "data"
CACHE_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude"
    r"\C--Users-ADMIN-OneDrive-Desktop-OSBORN-AGENTIC-AI-DVPT-ZINDI-PROJECTS-afro-health-qa"
    r"\12b7cb83-c045-4a3f-bea0-f953b0ecea11\scratchpad\emb_cache"
)

MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-small": "intfloat/multilingual-e5-small",
}
# e5 models expect "query:"/"passage:" prefixes; bge-m3 does not.
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "


def _encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot product after this
    ).astype(np.float32)


def _cache_path(model_key: str, split: str) -> Path:
    return CACHE_DIR / f"{model_key}__{split}.npy"


def run(model_key: str, limit: int | None) -> dict:
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(REPO / "data" / "processed" / "train_core.csv", dtype=str).fillna("")
    held = pd.read_csv(REPO / "data" / "processed" / "held_out.csv", dtype=str).fillna("")
    if limit:
        # keep the per-subset mix when sampling for a quick smoke test
        held = held.groupby("subset", group_keys=False).head(max(1, limit // held["subset"].nunique()))

    hf_id = MODELS[model_key]
    is_e5 = model_key.startswith("e5")
    print(f"[load] {hf_id}")
    t0 = time.time()
    model = SentenceTransformer(hf_id)
    print(f"[load] done in {time.time() - t0:.1f}s")

    # Encode the whole pool once (cached), then slice per subset.
    pool_cache = _cache_path(model_key, "pool")
    if pool_cache.exists():
        print(f"[cache] pool embeddings <- {pool_cache.name}")
        pool_emb = np.load(pool_cache)
    else:
        print(f"[encode] pool questions: {len(pool)}")
        pool_emb = _encode(model, pool["input"].tolist(), E5_DOC_PREFIX if is_e5 else "", 64)
        np.save(pool_cache, pool_emb)

    print(f"[encode] held-out questions: {len(held)}")
    held_emb = _encode(model, held["input"].tolist(), E5_QUERY_PREFIX if is_e5 else "", 64)

    pool_subset = pool["subset"].to_numpy()
    pool_answers = pool["output"].to_numpy()
    pool_idx_all = np.arange(len(pool))

    preds: list[str] = []
    sims: list[float] = []
    for i in range(len(held)):
        s = held["subset"].iloc[i]
        mask = pool_subset == s
        cand_emb = pool_emb[mask]
        cand_ans = pool_answers[mask]
        if len(cand_emb) == 0:  # no same-subset pool (shouldn't happen) -> global
            cand_emb, cand_ans = pool_emb, pool_answers
        scores = cand_emb @ held_emb[i]
        j = int(np.argmax(scores))
        preds.append(str(cand_ans[j]))
        sims.append(float(scores[j]))

    held = held.copy()
    held["prediction"] = preds
    held["retrieval_sim"] = sims

    # --- score with the frozen, LB-matching ROUGE, per subset and overall ---
    def _score(df: pd.DataFrame) -> dict:
        r = score_rouge(df["prediction"].tolist(), df["output"].tolist())
        rouge_only = W_R1 * r.rouge1 + W_RL * r.rouge_l
        return {
            "n": int(r.n),
            "rouge1": round(r.rouge1, 4),
            "rougeL": round(r.rouge_l, 4),
            "rouge_only_combined": round(rouge_only, 4),
            "mean_sim": round(float(df["retrieval_sim"].mean()), 4),
        }

    per_subset = {s: _score(g) for s, g in held.groupby("subset")}
    overall = _score(held)

    results = {
        "model_key": model_key,
        "model_hf_id": hf_id,
        "is_fallback_proxy": is_e5,
        "pool": "data/processed/train_core.csv",
        "queries": "data/processed/held_out.csv",
        "retrieval": "nearest train question within same subset, cosine on normalized embeddings",
        "note": (
            "ROUGE is MEASURED (frozen LB-matching tokenizer). The 0.26 LLM-judge "
            "term is NOT computed here; the report labels any judge figure as estimated."
        ),
        "weights": {"rouge1": W_R1, "rougeL": W_RL},
        "overall": overall,
        "per_subset": per_subset,
    }

    preds_path = OUT_DIR / "bge_retrieval_heldout.csv"
    held[["ID", "subset", "input", "output", "prediction", "retrieval_sim"]].to_csv(
        preds_path, index=False, encoding="utf-8"
    )
    scores_path = OUT_DIR / "bge_retrieval_scores.json"
    scores_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== MEASURED RETRIEVAL RESULTS ({}) ===".format(model_key))
    print(f"{'subset':<10} {'n':>5} {'R1':>7} {'RL':>7} {'comb*':>7} {'sim':>6}")
    for s in sorted(per_subset):
        d = per_subset[s]
        print(f"{s:<10} {d['n']:>5} {d['rouge1']:>7} {d['rougeL']:>7} {d['rouge_only_combined']:>7} {d['mean_sim']:>6}")
    print(f"{'ALL':<10} {overall['n']:>5} {overall['rouge1']:>7} {overall['rougeL']:>7} "
          f"{overall['rouge_only_combined']:>7} {overall['mean_sim']:>6}")
    print(f"\n[write] {preds_path}")
    print(f"[write] {scores_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="bge-m3")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: cap held-out rows")
    ap.add_argument("--fallback", action="store_true", help="use e5-small if bge-m3 errors")
    args = ap.parse_args()

    try:
        run(args.model, args.limit)
    except Exception as exc:  # noqa: BLE001
        if args.model == "bge-m3" and (args.fallback or True):
            print(f"\n[warn] {args.model} failed ({type(exc).__name__}: {exc}).")
            print("[warn] falling back to e5-small proxy.\n")
            run("e5-small", args.limit)
        else:
            raise


if __name__ == "__main__":
    main()
