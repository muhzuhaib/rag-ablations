"""Retrieval metrics.

Implemented here rather than pulled from `pytrec_eval` for two reasons: the C
extension needs a build toolchain (which would break "reproducible by anyone on
a laptop"), and the definitions below are the ones every number in the results
table depends on, so they are worth being able to read.

Conventions follow TREC / BEIR:

* Relevance judgments (`qrels`) are `{query_id: {doc_id: gain}}`. A document
  absent from the judgments is treated as non-relevant (gain 0), which is the
  standard pooled-judgment assumption.
* Runs are `{query_id: [doc_id, ...]}`, ranked best-first.
* Scores are averaged over queries, unweighted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the log2(rank + 1) discount."""
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """Normalised DCG at k, averaged over queries that have judgments.

    Queries with no relevant documents are skipped rather than scored 0 — they
    carry no information and including them just scales every system down by
    the same constant.
    """
    scores = []
    for query_id, judged in qrels.items():
        ideal_gains = sorted((g for g in judged.values() if g > 0), reverse=True)
        if not ideal_gains:
            continue
        retrieved = run.get(query_id, [])[:k]
        actual = dcg([judged.get(doc_id, 0) for doc_id in retrieved])
        ideal = dcg(ideal_gains[:k])
        scores.append(actual / ideal if ideal else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(qrels: Qrels, run: Run, k: int = 100) -> float:
    """Fraction of relevant documents that appear in the top k, averaged over queries.

    This is the ceiling on anything a reranker can achieve: a reranker only
    reorders what the first stage handed it, so recall@k of the first stage
    bounds the whole pipeline.
    """
    scores = []
    for query_id, judged in qrels.items():
        relevant = {doc_id for doc_id, gain in judged.items() if gain > 0}
        if not relevant:
            continue
        retrieved = set(run.get(query_id, [])[:k])
        scores.append(len(relevant & retrieved) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(qrels: Qrels, run: Run, ks: Sequence[int] = (10,)) -> dict[str, float]:
    """Score a run on the metrics reported in the results table."""
    results = {f"ndcg@{k}": ndcg_at_k(qrels, run, k) for k in ks}
    results["recall@100"] = recall_at_k(qrels, run, 100)
    return results
