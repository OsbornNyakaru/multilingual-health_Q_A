"""Beam-candidate reranker using AfroLM-BertScore.

Given N diverse beam candidates per question, score each against a set of
reference answers from the same language (retrieved from the training pool)
and pick the highest-averaging candidate.

Design note (see docs/DECISIONS.md ADR-004 at decision time):
We use BM25 retrieval to pick the reference pool because embedding retrieval
adds a second encoder pass per row; BM25 is ~100x faster and the reference
pool quality is dominated by lexical overlap with the question in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RerankerConfig:
    reference_pool_size: int = 32
    reference_pool_strategy: str = "bm25"  # "bm25" | "random" | "all"


class AfroLMReranker:
    """Rerank beam candidates against a per-language retrieval pool."""

    def __init__(
        self,
        training_df: pd.DataFrame,
        language_col: str = "Language",
        response_col: str = "Response",
        question_col: str = "Question",
        config: RerankerConfig | None = None,
    ):
        self.config = config or RerankerConfig()
        self._pool_by_lang: dict[str, list[str]] = {}
        self._questions_by_lang: dict[str, list[str]] = {}
        self._bm25_by_lang: dict[str, object] = {}

        for lang, group in training_df.groupby(language_col, sort=True):
            self._pool_by_lang[lang] = group[response_col].astype(str).tolist()
            self._questions_by_lang[lang] = group[question_col].astype(str).tolist()
            if self.config.reference_pool_strategy == "bm25":
                self._bm25_by_lang[lang] = self._build_bm25(self._questions_by_lang[lang])

    @staticmethod
    def _build_bm25(corpus_questions: list[str]):
        # Tiny BM25 implementation to avoid another dep.
        return _SimpleBM25([q.split() for q in corpus_questions])

    def _pool_for(self, language: str, question: str) -> list[str]:
        pool = self._pool_by_lang.get(language, [])
        if not pool:
            return []
        n = min(self.config.reference_pool_size, len(pool))
        strategy = self.config.reference_pool_strategy
        if strategy == "all" or n >= len(pool):
            return pool[:n]
        if strategy == "random":
            rng = np.random.default_rng(abs(hash(question)) % (2**32))
            idx = rng.choice(len(pool), size=n, replace=False)
            return [pool[i] for i in idx]
        # bm25
        bm25 = self._bm25_by_lang[language]
        scores = bm25.scores(question.split())
        top = np.argsort(scores)[::-1][:n]
        return [pool[i] for i in top]

    def rerank(
        self,
        question: str,
        language: str,
        candidates: list[str],
    ) -> str:
        """Pick the candidate with the highest mean AfroLM-BertScore against the pool."""
        from afro_health_qa.evaluation.afrolm_bertscore import (
            score_afrolm_bertscore_per_example,
        )

        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        pool = self._pool_for(language, question)
        if not pool:
            return candidates[0]

        # Score each candidate as mean F1 over the pool.
        best_idx = 0
        best_score = -1.0
        for i, cand in enumerate(candidates):
            preds = [cand] * len(pool)
            refs = pool
            scores = score_afrolm_bertscore_per_example(preds, refs)
            mean_f1 = float(np.mean(scores)) if scores else 0.0
            if mean_f1 > best_score:
                best_score = mean_f1
                best_idx = i
        return candidates[best_idx]


class _SimpleBM25:
    """Minimal BM25 — avoids pulling in rank-bm25.

    Good enough for our retrieval-pool step; not optimised for huge corpora.
    """

    def __init__(self, tokenised_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenised_docs
        self.n_docs = len(tokenised_docs)
        self.avgdl = (
            sum(len(d) for d in tokenised_docs) / self.n_docs if self.n_docs else 0.0
        )
        self.doc_lengths = [len(d) for d in tokenised_docs]
        self.doc_freq: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in tokenised_docs:
            tf_row: dict[str, int] = {}
            for term in doc:
                tf_row[term] = tf_row.get(term, 0) + 1
            self.tf.append(tf_row)
            for term in tf_row:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        # IDF per term.
        import math

        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def scores(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=float)
        for q in query_terms:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, tf_row in enumerate(self.tf):
                tf = tf_row.get(q, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores
