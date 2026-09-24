"""Track 1 — BGE-M3 retrieval submission on the real Zindi test set.

For each Test.csv question we retrieve the nearest *labelled* question within the
same language subset and copy its answer as the prediction. The pool is the FULL
labelled data (train_core + held_out + Val); the test set is disjoint from all of
it, so this is not leakage — it just gives the richest possible neighbour pool.

Predictions are written as a Zindi-format submission via the existing
``submission/format.py`` writer and checked with ``submission/validate.py``.

This is the autoresearch "lock the floor" step (program.md P8): a real, scorable
practice submission produced before any modelling.

Usage:
    python scripts/retrieval_test_submission.py [--model bge-m3|e5-small] [--limit N]
"""

from __future__ import annotations

import argparse
import os

# Thread/offline settings MUST be set before numpy/torch import. This box is
# 8 GB CPU-only; unbounded OpenMP threads + large batches segfault bge-m3 (exit
# 139), so we cap threads and stay offline (the model is already cached).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from afro_health_qa.submission.format import (  # noqa: E402
    SubmissionMetadata,
    build_submission_df,
    write_submission_csv,
)
from afro_health_qa.submission.validate import validate_submission_csv  # noqa: E402

OUT_DIR = REPO / "submissions"
SAMPLE_CSV = REPO / "data" / "raw" / "SampleSubmission.csv"
TEST_CSV = REPO / "data" / "raw" / "Test.csv"
POOL_FILES = [
    REPO / "data" / "processed" / "train_core.csv",  # cache-aligned first (see below)
    REPO / "data" / "processed" / "held_out.csv",
    REPO / "data" / "raw" / "Val.csv",
]

CACHE_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude"
    r"\C--Users-ADMIN-OneDrive-Desktop-OSBORN-AGENTIC-AI-DVPT-ZINDI-PROJECTS-afro-health-qa"
    r"\12b7cb83-c045-4a3f-bea0-f953b0ecea11\scratchpad\emb_cache"
)

MODELS = {"bge-m3": "BAAI/bge-m3", "e5-small": "intfloat/multilingual-e5-small"}
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "


def _encode(model, texts: list[str], prefix: str, batch_size: int = 16) -> np.ndarray:
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot product after this
    ).astype(np.float32)


def _encode_cached(model, texts: list[str], prefix: str, cache: Path, chunk: int = 256) -> np.ndarray:
    """Encode in chunks, persisting after each chunk so a segfault can resume."""
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(texts):
            print(f"[cache] embeddings <- {cache.name} ({len(emb)})")
            return emb
        if len(emb) < len(texts):
            print(f"[resume] {cache.name} has {len(emb)}/{len(texts)}; continuing")
            done = emb
        else:
            done = np.empty((0, model.get_sentence_embedding_dimension()), np.float32)
    else:
        done = np.empty((0, model.get_sentence_embedding_dimension()), np.float32)
    start = len(done)
    parts = [done] if start else []
    for i in range(start, len(texts), chunk):
        block = _encode(model, texts[i:i + chunk], prefix)
        parts.append(block)
        np.save(cache, np.vstack(parts).astype(np.float32))  # checkpoint
        print(f"[encode] {min(i + chunk, len(texts))}/{len(texts)}")
    emb = np.vstack(parts).astype(np.float32)
    np.save(cache, emb)
    return emb


def _load_pool(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str).fillna("")
        df["_source"] = f.name
        frames.append(df[["ID", "input", "output", "subset", "_source"]])
    return pd.concat(frames, ignore_index=True)


def _pool_embeddings(model, model_key: str, pool: pd.DataFrame, is_e5: bool) -> np.ndarray:
    """Encode the full pool, reusing the cached train_core embeddings when aligned.

    ``retrieval_baseline_bge.py`` cached ``<key>__pool.npy`` for the 28,319-row
    train_core (in file order). Our pool puts train_core first, so if that cache
    exists and matches, we only encode the held_out + Val tail.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_cache = CACHE_DIR / f"{model_key}__fullpool.npy"
    if full_cache.exists():
        emb = np.load(full_cache)
        if len(emb) == len(pool):
            print(f"[cache] full pool embeddings <- {full_cache.name}")
            return emb
        print(f"[cache] {full_cache.name} stale ({len(emb)} != {len(pool)}); re-encoding")

    n_train = sum(pool["_source"] == "train_core.csv")
    train_cache = CACHE_DIR / f"{model_key}__pool.npy"
    prefix = E5_DOC_PREFIX if is_e5 else ""
    if train_cache.exists() and len(np.load(train_cache, mmap_mode="r")) == n_train:
        print(f"[cache] reusing train_core embeddings <- {train_cache.name} ({n_train} rows)")
        head = np.load(train_cache)
        tail_texts = pool["input"].iloc[n_train:].tolist()
        print(f"[encode] remaining pool rows: {len(tail_texts)}")
        tail = _encode(model, tail_texts, prefix)
        emb = np.vstack([head, tail]).astype(np.float32)
    else:
        print(f"[encode] full pool rows: {len(pool)}")
        emb = _encode(model, pool["input"].tolist(), prefix)

    np.save(full_cache, emb)
    return emb


def run(model_key: str, limit: int | None, pool_mode: str) -> Path:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hf_id = MODELS[model_key]
    is_e5 = model_key.startswith("e5")

    # pool=core (default) uses train_core only — its embeddings are already cached,
    # so no heavy re-encoding is needed and the held-out R1=0.560 was measured on it.
    files = POOL_FILES if pool_mode == "full" else POOL_FILES[:1]
    pool = _load_pool(files)
    test = pd.read_csv(TEST_CSV, dtype=str).fillna("")
    if limit:
        test = test.groupby("subset", group_keys=False).head(
            max(1, limit // test["subset"].nunique())
        )
    print(f"[data] pool={len(pool)} ({pool_mode}) rows  test={len(test)} rows")

    print(f"[load] {hf_id}")
    t0 = time.time()
    model = SentenceTransformer(hf_id)
    print(f"[load] done in {time.time() - t0:.1f}s")

    if pool_mode == "full":
        pool_emb = _pool_embeddings(model, model_key, pool, is_e5)
    else:
        train_cache = CACHE_DIR / f"{model_key}__pool.npy"
        pool_emb = np.load(train_cache)
        if len(pool_emb) != len(pool):
            raise SystemExit(f"core cache {len(pool_emb)} != pool {len(pool)}; run retrieval_baseline_bge.py")
        print(f"[cache] core pool embeddings <- {train_cache.name} ({len(pool_emb)})")

    print(f"[encode] test questions: {len(test)}")
    test_cache = CACHE_DIR / f"{model_key}__test{'_lim' + str(limit) if limit else ''}.npy"
    test_emb = _encode_cached(
        model, test["input"].tolist(), E5_QUERY_PREFIX if is_e5 else "", test_cache
    )

    pool_subset = pool["subset"].to_numpy()
    pool_answers = pool["output"].to_numpy()

    preds: list[str] = []
    sims: list[float] = []
    for i in range(len(test)):
        s = test["subset"].iloc[i]
        mask = pool_subset == s
        cand_emb = pool_emb[mask]
        cand_ans = pool_answers[mask]
        if len(cand_emb) == 0:  # no same-subset pool (shouldn't happen) -> global
            cand_emb, cand_ans = pool_emb, pool_answers
        scores = cand_emb @ test_emb[i]
        j = int(np.argmax(scores))
        preds.append(str(cand_ans[j]))
        sims.append(float(scores[j]))

    # --- write Zindi submission, mirroring SampleSubmission columns exactly ---
    sample_columns = tuple(pd.read_csv(SAMPLE_CSV, dtype=str).columns.tolist())
    sub_df = build_submission_df(test["ID"].tolist(), preds, submission_columns=sample_columns)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{ts}_{model_key}_retrieval_test.csv"
    meta = SubmissionMetadata(
        run_id=f"{ts}_{model_key}_retrieval",
        hypothesis="Track1: full-pool BGE-M3 within-subset nearest-neighbour retrieval on the test set.",
        model_hf_id=hf_id,
        adapter_path=None,
        decoding_config="retrieval-top1 (no generation)",
        n_rows=len(sub_df),
        created_utc=ts,
        git_hash="",
        notes=(
            f"pool={len(pool)} (train_core+held_out+Val); mean retrieval cos={np.mean(sims):.4f}. "
            "Held-out R1=0.560/RL=0.491 measured on the same retriever."
        ),
    )
    write_submission_csv(sub_df, out_path, metadata=meta, submission_columns=sample_columns)

    stats = validate_submission_csv(out_path, test_csv=TEST_CSV, sample_csv=SAMPLE_CSV)
    print("\n=== SUBMISSION WRITTEN & VALIDATED ===")
    print(f"  path:        {out_path}")
    print(f"  rows:        {stats['n_rows']}  (expected 2618)")
    print(f"  empty preds: {stats['n_empty_answers']}")
    print(f"  mean chars:  {stats['mean_chars']:.1f}")
    print(f"  mean cos:    {np.mean(sims):.4f}")
    # per-subset mean similarity (a proxy for retrieval confidence)
    test = test.assign(_sim=sims)
    print("  per-subset mean cos:")
    for s, g in test.groupby("subset"):
        print(f"    {s:<10} n={len(g):>4}  cos={g['_sim'].mean():.4f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="bge-m3")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: cap test rows")
    ap.add_argument("--pool", choices=["core", "full"], default="core",
                    help="core=train_core only (cached, reliable); full=+held_out+Val")
    args = ap.parse_args()
    run(args.model, args.limit, args.pool)


if __name__ == "__main__":
    main()
