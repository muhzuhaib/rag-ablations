"""Metric tests with hand-computed expectations.

Every number in the results table flows through these two functions, so they
are checked against values worked out by hand rather than against whatever the
implementation happened to return the first time.
"""

import math

import pytest

from rag_ablations.metrics import evaluate, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": ["d1", "d2", "d3"]}
    assert ndcg_at_k(qrels, run, k=10) == pytest.approx(1.0)


def test_no_relevant_documents_retrieved_scores_zero():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": ["d2", "d3"]}
    assert ndcg_at_k(qrels, run, k=10) == 0.0


def test_ndcg_matches_hand_computation():
    # One relevant document sitting at rank 2 (index 1).
    # DCG  = 1 / log2(3) ; IDCG = 1 / log2(2) = 1
    qrels = {"q1": {"d9": 1}}
    run = {"q1": ["d1", "d9", "d3"]}
    assert ndcg_at_k(qrels, run, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_rewards_higher_ranks():
    qrels = {"q1": {"d1": 1}}
    better = ndcg_at_k(qrels, {"q1": ["d1", "x", "y"]}, k=10)
    worse = ndcg_at_k(qrels, {"q1": ["x", "y", "d1"]}, k=10)
    assert better > worse


def test_graded_relevance_prefers_the_higher_gain_first():
    qrels = {"q1": {"d1": 2, "d2": 1}}
    correct = ndcg_at_k(qrels, {"q1": ["d1", "d2"]}, k=10)
    swapped = ndcg_at_k(qrels, {"q1": ["d2", "d1"]}, k=10)
    assert correct == pytest.approx(1.0)
    assert swapped < correct


def test_cutoff_excludes_documents_beyond_k():
    qrels = {"q1": {"d5": 1}}
    run = {"q1": ["a", "b", "c", "d", "d5"]}
    assert ndcg_at_k(qrels, run, k=3) == 0.0
    assert ndcg_at_k(qrels, run, k=5) > 0.0


def test_queries_without_relevant_documents_are_skipped():
    # q2 is unjudged; it must not drag the average toward zero.
    qrels = {"q1": {"d1": 1}, "q2": {}}
    run = {"q1": ["d1"], "q2": ["anything"]}
    assert ndcg_at_k(qrels, run, k=10) == pytest.approx(1.0)


def test_missing_query_in_run_scores_zero_not_crash():
    qrels = {"q1": {"d1": 1}}
    assert ndcg_at_k(qrels, {}, k=10) == 0.0


def test_recall_counts_relevant_documents_found():
    qrels = {"q1": {"d1": 1, "d2": 1, "d3": 1}}
    run = {"q1": ["d1", "x", "d3"]}
    assert recall_at_k(qrels, run, k=100) == pytest.approx(2 / 3)


def test_recall_respects_the_cutoff():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": ["d1", "junk", "d2"]}
    assert recall_at_k(qrels, run, k=2) == pytest.approx(0.5)


def test_metrics_average_across_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {"q1": ["d1"], "q2": ["wrong"]}
    assert ndcg_at_k(qrels, run, k=10) == pytest.approx(0.5)


def test_evaluate_reports_the_table_columns():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": ["d1"]}
    assert evaluate(qrels, run, ks=(10,)) == {
        "ndcg@10": pytest.approx(1.0),
        "recall@100": pytest.approx(1.0),
    }
