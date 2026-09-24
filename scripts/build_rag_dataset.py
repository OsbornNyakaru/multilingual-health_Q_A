"""Track 2 prep — build the RAG-enriched fine-tuning dataset (local, CPU, no model).

For each training row we retrieve its k=3 nearest *same-subset* neighbours (by the
cached BGE-M3 question embeddings), prepend them as in-language few-shot examples,
and set the completion to the row's own reference answer. Fine-tuning on this
teaches the model to adapt retrieved context into the target answer — the
11th-place "RAG-enriched fine-tuning" recipe.

Prompt formatting mirrors ``autoresearch_nlp/train.py`` exactly (same per-language
system instruction, question/answer markers), so a fine-tuned adapter drops into
that harness for inference unchanged.

Guarantees (validated at the end):
  - a row never retrieves itself,
  - held_out.csv is never used (pool = train_core only),
  - each example has up to k neighbours (fewer only if its subset is tiny).

Inputs:
  data/processed/train_core.csv  +  cached  <emb_cache>/bge-m3__pool.npy
  (the cache is produced by scripts/retrieval_baseline_bge.py --model bge-m3)

Outputs:
  data/processed/train_rag_enriched.jsonl   (95% stratified)
  data/processed/val_rag_enriched.jsonl     ( 5% stratified, for early stopping)

Usage:
    python scripts/build_rag_dataset.py [--k 3] [--val-frac 0.05] [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "autoresearch_nlp"))  # for the prompt conventions

# Pull the exact language conventions from the harness so prompts match 1:1.
from train import (  # type: ignore  # noqa: E402
    SYSTEM_INSTRUCTIONS,
    PROMPT_TEMPLATES,
    ANSWER_MARKERS,
    lang_of,
)

TRAIN_CORE = REPO / "data" / "processed" / "train_core.csv"
HELD_OUT = REPO / "data" / "processed" / "held_out.csv"
OUT_TRAIN = REPO / "data" / "processed" / "train_rag_enriched.jsonl"
OUT_VAL = REPO / "data" / "processed" / "val_rag_enriched.jsonl"

CACHE_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude"
    r"\C--Users-ADMIN-OneDrive-Desktop-OSBORN-AGENTIC-AI-DVPT-ZINDI-PROJECTS-afro-health-qa"
    r"\12b7cb83-c045-4a3f-bea0-f953b0ecea11\scratchpad\emb_cache"
)
POOL_CACHE = CACHE_DIR / "bge-m3__pool.npy"


def _q_label(lang: str) -> str:
    # "Question: {question}\nAnswer:" -> "Question:"
    return PROMPT_TEMPLATES[lang].split("{question}")[0].strip()


def _example_block(neighbours: list[tuple[str, str]], lang: str) -> str:
    ql, al = _q_label(lang), ANSWER_MARKERS[lang]
    return "\n\n".join(
        f"{ql} {nq.strip()}\n{al} {na.strip()}" for nq, na in neighbours
    )


def build_prompt_with_rag(question: str, subset: str, neighbours: list[tuple[str, str]]) -> str:
    """System instruction + in-language few-shot examples + the target question."""
    lang = lang_of(subset)
    sys_instr = SYSTEM_INSTRUCTIONS[lang]
    user = PROMPT_TEMPLATES[lang].format(question=question.strip())
    if neighbours:
        return f"{sys_instr}\n\n{_example_block(neighbours, lang)}\n\n{user}"
    return f"{sys_instr}\n\n{user}"


def run(k: int, val_frac: float, seed: int) -> None:
    if not POOL_CACHE.exists():
        raise SystemExit(
            f"missing embedding cache: {POOL_CACHE}\n"
            "Run:  python scripts/retrieval_baseline_bge.py --model bge-m3  first."
        )
    train = pd.read_csv(TRAIN_CORE, dtype=str).fillna("")
    emb = np.load(POOL_CACHE)
    if len(emb) != len(train):
        raise SystemExit(
            f"cache rows {len(emb)} != train_core rows {len(train)} — re-encode the pool."
        )
    held_ids = set(pd.read_csv(HELD_OUT, dtype=str)["ID"].astype(str))
    assert not (set(train["ID"].astype(str)) & held_ids), "train_core overlaps held_out!"

    inputs = train["input"].to_numpy()
    outputs = train["output"].to_numpy()
    ids = train["ID"].astype(str).to_numpy()
    subsets = train["subset"].to_numpy()

    records: list[dict] = []
    print(f"[retrieve] k={k} same-subset neighbours per row over {len(train)} rows")
    for s in sorted(set(subsets)):
        loc = np.where(subsets == s)[0]               # global indices in this subset
        sub_emb = emb[loc]                            # (n_s, d), already normalized
        sims = sub_emb @ sub_emb.T                    # (n_s, n_s) cosine
        np.fill_diagonal(sims, -1.0)                  # never retrieve self
        n_s = len(loc)
        kk = min(k, n_s - 1)
        # top-kk per row via argpartition then sort those few
        part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk] if kk > 0 else None
        for li in range(n_s):
            gi = loc[li]
            if kk > 0:
                cols = part[li]
                cols = cols[np.argsort(-sims[li, cols])]   # order by similarity desc
                nbr_global = loc[cols]
                neighbours = [(str(inputs[g]), str(outputs[g])) for g in nbr_global]
                nbr_ids = [str(ids[g]) for g in nbr_global]
            else:
                neighbours, nbr_ids = [], []
            prompt = build_prompt_with_rag(str(inputs[gi]), str(s), neighbours)
            records.append({
                "id": str(ids[gi]),
                "subset": str(s),
                "lang": lang_of(str(s)),
                "retrieval_ids": nbr_ids,
                "prompt": prompt,
                "completion": " " + str(outputs[gi]).strip(),
            })
        print(f"  {s:<10} n={n_s:>5}  k={kk}")

    # ── stratified train/val split (by subset) ──────────────────────────────
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records)
    val_mask = np.zeros(len(df), dtype=bool)
    for s, grp in df.groupby("subset"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_mask[idx[:n_val]] = True
    train_df, val_df = df[~val_mask], df[val_mask]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    for path, d in [(OUT_TRAIN, train_df), (OUT_VAL, val_df)]:
        with path.open("w", encoding="utf-8") as f:
            for rec in d.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── validation / summary ────────────────────────────────────────────────
    self_hit = sum(r["id"] in r["retrieval_ids"] for r in records)
    leak = sum(any(rid in held_ids for rid in r["retrieval_ids"]) for r in records)
    n_neighbours = [len(r["retrieval_ids"]) for r in records]
    print("\n=== RAG DATASET BUILT ===")
    print(f"  total rows:        {len(records)}")
    print(f"  train / val:       {len(train_df)} / {len(val_df)}")
    print(f"  self-retrievals:   {self_hit}   (must be 0)")
    print(f"  held_out leaks:    {leak}   (must be 0)")
    print(f"  neighbours/row:    min={min(n_neighbours)} max={max(n_neighbours)} "
          f"mean={np.mean(n_neighbours):.2f}")
    print(f"  prompt chars:      mean={int(np.mean([len(r['prompt']) for r in records]))}")
    print(f"  -> {OUT_TRAIN}")
    print(f"  -> {OUT_VAL}")
    # show one English example so the format is eyeballable
    ex = next(r for r in records if r["lang"] == "eng" and r["retrieval_ids"])
    print("\n--- sample (eng) ---\n" + ex["prompt"][:900] + "\n[COMPLETION]" + ex["completion"][:200])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    run(args.k, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
