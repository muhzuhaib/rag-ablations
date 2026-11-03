"""The reproducibility check covers both stored retrieval metrics."""

import json

import pytest

from rag_ablations import benchmark
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


def _stored_with_split(tmp_path, split):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "split": split,
                "results": [{"system": "bm25", "ndcg@10": 0.5, "recall@100": 0.9}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_check_fails_when_the_stored_split_differs(tmp_path, capsys):
    stored = _stored_with_split(tmp_path, "test")
    result = _result(0.5, 0.9)
    assert check_against(stored, [result], tolerance=0.001, split="train") == 1
    assert "requested split 'train' vs stored 'test'" in capsys.readouterr().out


def test_check_passes_when_the_stored_split_matches(tmp_path):
    stored = _stored_with_split(tmp_path, "test")
    result = _result(0.5, 0.9)
    assert check_against(stored, [result], tolerance=0.001, split="test") == 0


def test_check_skips_the_split_missing_from_old_results(stored_results):
    result = _result(0.5, 0.9)
    assert check_against(stored_results, [result], tolerance=0.001, split="train") == 0


def test_default_split_keeps_the_existing_results_filename():
    # The committed results files are named without a split.
    assert benchmark.results_filename("scifact", "sparse", "test") == "scifact-sparse.json"


def test_other_splits_get_their_own_results_filename():
    assert benchmark.results_filename("scifact", "sparse", "train") == "scifact-sparse-train.json"
