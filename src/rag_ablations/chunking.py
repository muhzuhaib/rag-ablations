"""Chunking strategies, and the wrapper that makes them measurable.

Chunking is the RAG design choice most often decided by copying a tutorial's
`chunk_size=512, overlap=50`. It is a benchmark variable here.

Measuring it needs care. Chunking changes the *unit of retrieval*, but the
relevance judgments are written against whole documents, so scoring chunks
directly would compare against the wrong ground truth. `Chunked` therefore
retrieves chunks and pools them back to document level with a max — a document
is as relevant as its best passage — which is both the standard approach and
the one that keeps every row in the table scored against the same qrels.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Chunker = Callable[[str], list[str]]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def whole_document(text: str) -> list[str]:
    """No chunking — the control condition."""
    return [text]


def fixed_window(size: int = 128, overlap: int = 32) -> Chunker:
    """Fixed-length word windows with a sliding overlap.

    Overlap exists to stop a passage that straddles a boundary from being split
    across two chunks and diluted in both. It costs index size proportionally.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never advances")

    def chunk(text: str) -> list[str]:
        words = text.split()
        if len(words) <= size:
            return [text]
        step = size - overlap
        return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i:i + size]]

    return chunk


def sentences(per_chunk: int = 3, stride: int = 2) -> Chunker:
    """Sentence-aligned windows.

    Respecting sentence boundaries avoids cutting mid-clause, which matters
    more for a bi-encoder than for BM25: a truncated clause changes the
    embedding, while bag-of-words scoring barely notices.
    """

    def chunk(text: str) -> list[str]:
        parts = [s for s in _SENTENCE.split(text) if s.strip()]
        if len(parts) <= per_chunk:
            return [text]
        return [
            " ".join(parts[i : i + per_chunk])
            for i in range(0, len(parts), stride)
            if parts[i:i + per_chunk]
        ]

    return chunk


class Chunked:
    """Retrieve over chunks, report results at document level."""

    def __init__(self, retriever, chunker: Chunker, *, name: str | None = None) -> None:
        self.retriever = retriever
        self.chunker = chunker
        self.name = name or f"{getattr(retriever, 'name', 'retriever')} + chunking"
        self._parent: list[str] = []
        self.chunks_per_doc = 0.0

    def index(self, doc_ids: list[str], texts: list[str]) -> "Chunked":
        chunk_ids, chunk_texts, parents = [], [], []
        for doc_id, text in zip(doc_ids, texts):
            for i, piece in enumerate(self.chunker(text)):
                chunk_ids.append(f"{doc_id}#{i}")
                chunk_texts.append(piece)
                parents.append(doc_id)

        self._parent = parents
        self.chunks_per_doc = len(chunk_ids) / max(len(doc_ids), 1)
        self._index = {cid: parent for cid, parent in zip(chunk_ids, parents)}
        self.retriever.index(chunk_ids, chunk_texts)
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        # Over-retrieve: several chunks of the same document collapse into one
        # result, so asking for exactly k chunks would return fewer than k
        # documents and silently depress recall.
        best: dict[str, float] = {}
        for chunk_id, score in self.retriever.search(query, k=k * 4):
            parent = self._index.get(chunk_id, chunk_id)
            if score > best.get(parent, float("-inf")):
                best[parent] = score

        return sorted(best.items(), key=lambda kv: -kv[1])[:k]
