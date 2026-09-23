"""Fusion and reranking tests.

Both stages are tested against stub retrievers rather than real models. The
fusion arithmetic and the two-stage plumbing are what can silently be wrong;
downloading a 90 MB transformer to assert that MiniLM likes the right document
would test HuggingFace, not this code, and would make CI depend on the network.
"""

import pytest

from rag_ablations.retrievers import RRF, Reranked


class StubRetriever:
    """Returns a fixed ranking, ignoring the query."""

    def __init__(self, ranking, name="stub"):
        self.ranking = ranking
        self.name = name
        self.indexed = None

    def index(self, doc_ids, texts):
        self.indexed = list(doc_ids)
        return self

    def search(self, query, k=100):
        return [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(self.ranking)][:k]


def test_rrf_requires_two_systems():
    with pytest.raises(ValueError):
        RRF([StubRetriever(["a"])])


def test_rrf_indexes_every_child():
    a, b = StubRetriever(["x"]), StubRetriever(["y"])
    RRF([a, b]).index(["d1"], ["text"])
    assert a.indexed == ["d1"] and b.indexed == ["d1"]


def test_rrf_promotes_the_document_both_systems_agree_on():
    # "shared" is 2nd for one system and 3rd for the other, never 1st, but it
    # is the only document both rank, which is exactly what RRF rewards.
    a = StubRetriever(["a1", "shared", "a2"])
    b = StubRetriever(["b1", "b2", "shared"])
    assert RRF([a, b]).search("q")[0][0] == "shared"


def test_rrf_scores_match_the_formula():
    a = StubRetriever(["d1", "d2"])
    b = StubRetriever(["d2", "d1"])
    scores = dict(RRF([a, b], rrf_k=60).search("q"))
    # Each document is 1st for one system and 2nd for the other.
    expected = 1 / 61 + 1 / 62
    assert scores["d1"] == pytest.approx(expected)
    assert scores["d2"] == pytest.approx(expected)


def test_rrf_returns_the_union_of_both_result_lists():
    a = StubRetriever(["a1", "a2"])
    b = StubRetriever(["b1"])
    assert {doc for doc, _ in RRF([a, b]).search("q")} == {"a1", "a2", "b1"}


def test_rrf_respects_k():
    a = StubRetriever(["a1", "a2", "a3"])
    b = StubRetriever(["b1", "b2", "b3"])
    assert len(RRF([a, b]).search("q", k=2)) == 2


def test_rrf_ranks_by_descending_score():
    a = StubRetriever(["x", "y", "z"])
    b = StubRetriever(["x", "z", "y"])
    scores = [s for _, s in RRF([a, b]).search("q")]
    assert scores == sorted(scores, reverse=True)


class StubCrossEncoder:
    """Scores a pair by how early a marker appears in the document text."""

    def predict(self, pairs, **kwargs):
        return [1.0 if "GOOD" in text else 0.0 for _, text in pairs]


@pytest.fixture
def reranked(monkeypatch):
    first = StubRetriever(["bad1", "bad2", "good"])
    stage = Reranked(first, depth=3)
    monkeypatch.setattr(stage, "_load_model", lambda: StubCrossEncoder())
    return stage.index(["bad1", "bad2", "good"], ["no", "no", "GOOD"])


def test_reranking_lifts_the_relevant_document_to_the_top(reranked):
    assert reranked.search("q")[0][0] == "good"


def test_reranker_cannot_recover_documents_the_first_stage_missed(monkeypatch):
    # The whole point of reporting first-stage recall: "good" is not in the
    # candidate list, so no amount of reranking can surface it.
    first = StubRetriever(["bad1", "bad2"])
    stage = Reranked(first, depth=2)
    monkeypatch.setattr(stage, "_load_model", lambda: StubCrossEncoder())
    stage.index(["bad1", "bad2", "good"], ["no", "no", "GOOD"])
    assert "good" not in {doc for doc, _ in stage.search("q")}


def test_reranking_keeps_the_candidate_set_intact(reranked):
    assert {doc for doc, _ in reranked.search("q")} == {"bad1", "bad2", "good"}


def test_reranked_search_respects_k(reranked):
    assert len(reranked.search("q", k=1)) == 1


def test_k_beyond_depth_still_returns_the_first_stage_tail(monkeypatch):
    # The rerank depth is a compute budget, not a result cap: documents below
    # it keep their first-stage order, so recall@k stays comparable for k > depth.
    first = StubRetriever(["a", "b", "c", "d"])
    stage = Reranked(first, depth=2)
    monkeypatch.setattr(stage, "_load_model", lambda: StubCrossEncoder())
    stage.index(["a", "b", "c", "d"], ["no", "no", "no", "no"])
    assert [doc for doc, _ in stage.search("q", k=4)] == ["a", "b", "c", "d"]


def test_empty_first_stage_returns_nothing(monkeypatch):
    stage = Reranked(StubRetriever([]), depth=10)
    monkeypatch.setattr(stage, "_load_model", lambda: StubCrossEncoder())
    stage.index([], [])
    assert stage.search("q") == []
