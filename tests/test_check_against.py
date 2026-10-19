"""The reproducibility check covers both stored retrieval metrics."""

import json

import pytest

from rag_ablations.benchmark import Result, check_against


@pytest.fixture
def stored_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {"results": [{"system": "bm25", "ndcg@10": 0.5, "recall@100": 0.9}]}
        ),
        encoding="utf-8",
    )
    return path


def _result(ndcg, recall):
    return Result(
        system="bm25",
        notes="",
        metrics={"ndcg@10": ndcg, "recall@100": recall},
        index_seconds=0.0,
        query_seconds=0.0,
    )


def test_check_fails_on_recall_drift(stored_results, capsys):
    result = _result(0.5, 0.1)
    assert check_against(stored_results, [result], tolerance=0.001) == 1
    assert (
        "bm25: Recall@100 0.1000 vs stored 0.9000 (drift 0.8000)"
        in capsys.readouterr().out
    )


def test_check_passes_when_both_metrics_match(stored_results):
    result = _result(0.5, 0.9)
    assert check_against(stored_results, [result], tolerance=0.001) == 0


def test_check_still_fails_on_ndcg_drift(stored_results, capsys):
    result = _result(0.2, 0.9)
    assert check_against(stored_results, [result], tolerance=0.001) == 1
    assert (
        "bm25: nDCG@10 0.2000 vs stored 0.5000 (drift 0.3000)"
        in capsys.readouterr().out
    )


def test_check_skips_recall_missing_from_old_results(stored_results):
    stored_results.write_text(
        json.dumps({"results": [{"system": "bm25", "ndcg@10": 0.5}]}),
        encoding="utf-8",
    )
    result = _result(0.5, 0.1)
    assert check_against(stored_results, [result], tolerance=0.001) == 0
