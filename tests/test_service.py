"""Service tests against a stubbed corpus.

The real corpus is a network download, so `corpus.load` is patched. What is
being tested is the API contract — status codes, response shape, validation —
not BM25, which has its own tests.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag_ablations import corpus, service  # noqa: E402

DOCS = [
    corpus.Document("d1", "Helium balloons", "Helium is lighter than air."),
    corpus.Document("d2", "Gardening", "Tomatoes need sunlight and water."),
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        corpus,
        "load",
        lambda name, split="test", data_dir=None: corpus.Dataset(
            name=name, documents=DOCS, queries={}, qrels={}
        ),
    )
    monkeypatch.setattr(service.corpus, "load", corpus.load)
    with TestClient(service.app) as client:
        yield client


def test_health_reports_the_indexed_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 2
    assert body["system"] == "bm25"


def test_search_returns_the_matching_document(client):
    hits = client.get("/search", params={"q": "helium"}).json()["hits"]
    assert hits[0]["doc_id"] == "d1"
    assert hits[0]["title"] == "Helium balloons"


def test_search_response_carries_the_system_that_produced_it(client):
    # A result is not interpretable without knowing which pipeline ran.
    assert client.get("/search", params={"q": "helium"}).json()["system"] == "bm25"


def test_search_respects_k(client):
    hits = client.get("/search", params={"q": "helium tomatoes", "k": 1}).json()["hits"]
    assert len(hits) == 1


def test_empty_query_is_rejected(client):
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_k_above_the_maximum_is_rejected(client):
    assert client.get("/search", params={"q": "x", "k": 500}).status_code == 422


def test_query_with_no_matches_returns_an_empty_list_not_an_error(client):
    response = client.get("/search", params={"q": "chromodynamics"})
    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_unknown_system_is_rejected_at_build_time():
    with pytest.raises(ValueError):
        service.build_retriever("magic")
