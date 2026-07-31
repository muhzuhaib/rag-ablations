"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF combines rankings using only rank position:

    score(d) = sum over systems of  1 / (rrf_k + rank(d))

Score-based fusion is the obvious alternative and it is the wrong one here.
BM25 scores are unbounded sums of idf terms while cosine similarities sit in
[-1, 1], so blending them requires normalising two distributions whose shape
changes per query, a tuning knob that quietly becomes a per-dataset fit. RRF
ignores magnitudes entirely, has one constant, and is what the TREC literature
uses. Choosing the method with fewer degrees of freedom is what keeps the
comparison honest.
"""

from __future__ import annotations

from collections import defaultdict


class RRF:
    """Fuse the rankings of two or more retrievers.

    ``rrf_k`` damps the contribution of low ranks; 60 is the constant from the
    original Cormack et al. paper and is left at the default rather than tuned,
    since tuning it against the test set is exactly the leak this project is
    supposed to be arguing against.
    """

    def __init__(self, retrievers: list, *, rrf_k: int = 60, name: str = "hybrid") -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF needs at least two retrievers to fuse")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.name = name

    def index(self, doc_ids: list[str], texts: list[str]) -> "RRF":
        for retriever in self.retrievers:
            retriever.index(doc_ids, texts)
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        # Each system contributes its own top k. Fusing shallower lists than we
        # return would cap recall below what either system alone achieves.
        fused: dict[str, float] = defaultdict(float)
        for retriever in self.retrievers:
            for rank, (doc_id, _) in enumerate(retriever.search(query, k=k)):
                fused[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return ranked[:k]
