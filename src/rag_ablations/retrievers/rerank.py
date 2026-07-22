"""Cross-encoder reranking over a first-stage candidate list.

A bi-encoder embeds the query and the document separately, so it never sees
them together and cannot model term interaction. A cross-encoder scores the
pair jointly, which is far more accurate and far too slow to run over a whole
corpus — hence the two-stage shape: retrieve cheaply, rerank the top of the
list expensively.

The consequence is stated in the results table rather than left implicit: a
reranker only reorders what the first stage returned, so **Recall@depth of the
first stage is a hard ceiling on the reranked nDCG**. A reranker cannot recover
a document that was never retrieved.
"""

from __future__ import annotations

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranked:
    """Wrap a first-stage retriever with a cross-encoder second stage."""

    def __init__(
        self,
        first_stage,
        model_name: str = DEFAULT_MODEL,
        *,
        depth: int = 100,
        batch_size: int = 64,
        name: str | None = None,
    ) -> None:
        self.first_stage = first_stage
        self.model_name = model_name
        self.depth = depth
        self.batch_size = batch_size
        self.name = name or f"{getattr(first_stage, 'name', 'first')} + rerank"
        self._texts: dict[str, str] = {}
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def index(self, doc_ids: list[str], texts: list[str]) -> "Reranked":
        self.first_stage.index(doc_ids, texts)
        # The cross-encoder needs the document text at query time, so keep it.
        self._texts = dict(zip(doc_ids, texts))
        return self

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        # Always pull `depth` candidates even when k is smaller — reranking a
        # deeper list is the entire benefit, and truncating to k first would
        # measure nothing but the first stage.
        candidates = self.first_stage.search(query, k=max(self.depth, k))[: self.depth]
        if not candidates:
            return []

        doc_ids = [doc_id for doc_id, _ in candidates]
        scores = self._load_model().predict(
            [(query, self._texts.get(doc_id, "")) for doc_id in doc_ids],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = sorted(zip(doc_ids, (float(s) for s in scores)), key=lambda kv: -kv[1])

        # Anything below the rerank depth keeps its first-stage order and sits
        # underneath. Dropping it would damage recall@k for k > depth and make
        # the recall column incomparable across rows.
        tail = [(doc_id, 0.0) for doc_id, _ in candidates[self.depth :]]
        return (reranked + tail)[:k]
