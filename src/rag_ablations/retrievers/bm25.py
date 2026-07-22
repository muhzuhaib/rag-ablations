"""Okapi BM25 — the baseline every other row in the results table is measured against.

BM25 is not a straw man. It still beats a lot of dense retrievers on
out-of-domain corpora, which is exactly why it is the baseline here: a dense
system that cannot beat it has not earned its GPU.

Written out rather than imported from `rank_bm25` because the baseline is the
load-bearing number in this project and it should be readable and tunable
(k1/b are ablation variables), not a black box.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..text import tokenize


class BM25:
    """Sparse lexical retrieval over an in-memory inverted index.

    Parameters follow the standard Okapi formulation:

    * ``k1`` controls term-frequency saturation — how quickly repeated
      occurrences of a term stop adding score. 1.2 is the Lucene default.
    * ``b`` controls length normalisation, from 0 (ignore document length) to
      1 (fully normalise). 0.75 is the Lucene default.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, *, stem: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.stem = stem
        self.doc_ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len = np.zeros(0)
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}

    def index(self, doc_ids: list[str], texts: list[str]) -> "BM25":
        self.doc_ids = list(doc_ids)
        lengths = []
        doc_freq: Counter[str] = Counter()

        for position, text in enumerate(texts):
            counts = Counter(tokenize(text, stem=self.stem))
            lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self._postings[term].append((position, freq))
            doc_freq.update(counts.keys())

        n = len(self.doc_ids)
        self._doc_len = np.array(lengths, dtype=np.float32)
        self._avg_len = float(self._doc_len.mean()) if n else 0.0

        # Robertson/Sparck-Jones idf with the +1 that keeps it non-negative for
        # terms appearing in more than half the collection.
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self._doc_len / self._avg_len)

        for term in tokenize(query, stem=self.stem):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            positions, freqs = zip(*postings)
            positions = np.asarray(positions)
            freqs = np.asarray(freqs, dtype=np.float32)
            scores[positions] += idf * (freqs * (self.k1 + 1)) / (freqs + norm[positions])

        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]
