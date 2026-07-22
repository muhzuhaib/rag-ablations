"""BM25 behaviour tests.

These pin the properties BM25 is supposed to have — not the exact float it
returns, which would just freeze in whatever the implementation does today.
"""

import pytest

from rag_ablations.retrievers import BM25

CORPUS = {
    "d1": "the cat sat on the mat",
    "d2": "dogs are loyal companions and dogs are friendly",
    "d3": "a cat and a dog living together",
    "d4": "quantum chromodynamics describes the strong interaction",
}


@pytest.fixture
def index():
    return BM25().index(list(CORPUS), list(CORPUS.values()))


def test_finds_the_obviously_matching_document(index):
    assert index.search("quantum chromodynamics")[0][0] == "d4"


def test_returns_nothing_for_out_of_vocabulary_queries(index):
    assert index.search("helicopter") == []


def test_documents_without_query_terms_are_excluded(index):
    returned = {doc_id for doc_id, _ in index.search("cat")}
    assert returned == {"d1", "d3"}


def test_respects_k(index):
    assert len(index.search("cat and dog", k=2)) == 2


def test_results_are_sorted_by_descending_score(index):
    scores = [score for _, score in index.search("cat and dog")]
    assert scores == sorted(scores, reverse=True)


def test_rare_terms_outweigh_common_ones(index):
    # "cat" appears in two documents, "chromodynamics" in one, so the rare term
    # should decide the ranking when a query contains both.
    assert index.search("cat chromodynamics")[0][0] == "d4"


def test_term_frequency_saturates():
    # d2 mentions "dogs" twice, d3 once. Two occurrences must score higher, but
    # not twice as high — that saturation is the whole point of k1.
    index = BM25().index(["d2", "d3"], [CORPUS["d2"], CORPUS["d3"]])
    scores = dict(index.search("dogs"))
    assert scores["d2"] > scores["d3"]
    assert scores["d2"] < 2 * scores["d3"]


def test_stemming_matches_morphological_variants():
    stemmed = BM25(stem=True).index(["d2"], [CORPUS["d2"]])
    raw = BM25(stem=False).index(["d2"], [CORPUS["d2"]])
    assert stemmed.search("dog")  # "dogs" stems to "dog"
    assert not raw.search("dog")


def test_length_normalisation_prefers_the_shorter_document():
    short = "machine learning"
    padded = "machine learning " + " ".join(f"filler{i}" for i in range(200))
    index = BM25().index(["short", "padded"], [short, padded])
    assert index.search("machine learning")[0][0] == "short"


def test_empty_query_returns_nothing(index):
    assert index.search("") == []
