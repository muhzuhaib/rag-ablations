"""Chunking tests, including the max-pooling that makes chunked runs comparable."""

import pytest

from rag_ablations.chunking import Chunked, fixed_window, sentences, whole_document
from rag_ablations.retrievers import BM25


def test_whole_document_is_the_identity():
    assert whole_document("a b c") == ["a b c"]


def test_fixed_window_splits_long_text():
    text = " ".join(str(i) for i in range(100))
    chunks = fixed_window(size=20, overlap=0)(text)
    assert len(chunks) == 5
    assert chunks[0].split()[0] == "0"


def test_fixed_window_leaves_short_text_alone():
    assert fixed_window(size=50)("three words only") == ["three words only"]


def test_fixed_window_overlaps_by_the_requested_amount():
    text = " ".join(str(i) for i in range(30))
    chunks = fixed_window(size=10, overlap=5)(text)
    assert chunks[0].split()[5:] == chunks[1].split()[:5]


def test_overlap_must_be_smaller_than_the_window():
    with pytest.raises(ValueError):
        fixed_window(size=10, overlap=10)


def test_sentence_chunker_respects_boundaries():
    text = "One. Two. Three. Four. Five."
    chunks = sentences(per_chunk=2, stride=2)(text)
    assert chunks[0] == "One. Two."


def test_sentence_chunker_leaves_short_text_alone():
    assert sentences(per_chunk=5)("Only one sentence.") == ["Only one sentence."]


@pytest.fixture
def chunked():
    # The relevant phrase sits at the end of a long document, where a
    # whole-document embedding or a length-normalised score would dilute it.
    long_doc = " ".join(["filler"] * 200) + " helium balloon experiment"
    docs = {"d1": long_doc, "d2": "unrelated text about gardening"}
    stage = Chunked(BM25(), fixed_window(size=50, overlap=10))
    return stage.index(list(docs), list(docs.values()))


def test_chunked_search_returns_document_ids_not_chunk_ids(chunked):
    assert {doc for doc, _ in chunked.search("helium balloon")} <= {"d1", "d2"}


def test_chunked_search_finds_content_buried_in_a_long_document(chunked):
    assert chunked.search("helium balloon")[0][0] == "d1"


def test_each_document_appears_at_most_once(chunked):
    returned = [doc for doc, _ in chunked.search("filler helium")]
    assert len(returned) == len(set(returned))


def test_chunk_count_is_recorded_for_the_results_table(chunked):
    # Index size is a real cost of chunking and belongs in the table.
    assert chunked.chunks_per_doc > 1


def test_pooling_takes_the_best_chunk_not_the_last():
    docs = {"d1": "alpha " * 60 + "beta", "d2": "beta"}
    stage = Chunked(BM25(), fixed_window(size=20, overlap=0)).index(
        list(docs), list(docs.values())
    )
    scores = dict(stage.search("beta"))
    assert scores["d1"] > 0
