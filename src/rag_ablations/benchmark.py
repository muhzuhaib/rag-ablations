"""Benchmark runner.

One command regenerates every number in the README:

    python -m rag_ablations.benchmark --dataset scifact

Results are written to `results/<dataset>.json` alongside the machine and
library versions they were produced on, so a table in the README can always be
traced back to a run rather than to a claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import corpus
from .metrics import evaluate
from .retrievers import BM25

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TOP_K = 100


@dataclass(frozen=True)
class System:
    """One row of the results table."""

    name: str
    build: Callable[[], object]
    notes: str = ""


def baseline_systems() -> list[System]:
    """The sparse baselines.

    Both analyzers are reported because the published BEIR BM25 numbers come
    from a stemming, stopword-removing Lucene analyzer. Showing the unstemmed
    variant makes the size of that effect explicit instead of leaving a
    discrepancy for a reader to trip over.
    """
    return [
        System(
            name="BM25 (stemmed)",
            build=lambda: BM25(k1=1.2, b=0.75, stem=True),
            notes="Lucene defaults k1=1.2 b=0.75, Porter stemming, stopwords removed",
        ),
        System(
            name="BM25 (no stemming)",
            build=lambda: BM25(k1=1.2, b=0.75, stem=False),
            notes="Same parameters, raw tokens",
        ),
    ]


@dataclass
class Result:
    system: str
    notes: str
    metrics: dict[str, float]
    index_seconds: float
    query_seconds: float
    extra: dict = field(default_factory=dict)


def run_system(system: System, dataset: corpus.Dataset) -> Result:
    doc_ids = [d.doc_id for d in dataset.documents]
    contents = [d.content for d in dataset.documents]

    retriever = system.build()
    started = time.perf_counter()
    retriever.index(doc_ids, contents)
    index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    run = {
        query_id: [doc_id for doc_id, _ in retriever.search(text, k=TOP_K)]
        for query_id, text in dataset.queries.items()
    }
    query_seconds = time.perf_counter() - started

    return Result(
        system=system.name,
        notes=system.notes,
        metrics=evaluate(dataset.qrels, run, ks=(10,)),
        index_seconds=index_seconds,
        query_seconds=query_seconds,
        extra={"ms_per_query": 1000 * query_seconds / max(len(run), 1)},
    )


def to_markdown(results: list[Result]) -> str:
    lines = [
        "| System | nDCG@10 | Recall@100 | Index (s) | ms/query |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.system} | {r.metrics['ndcg@10']:.4f} | {r.metrics['recall@100']:.4f} "
            f"| {r.index_seconds:.1f} | {r.extra['ms_per_query']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=sorted(corpus.DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"Loading {args.dataset} ...", flush=True)
    dataset = corpus.load(args.dataset, split=args.split)
    print(
        f"  {len(dataset.documents)} documents, {len(dataset.queries)} queries, "
        f"{sum(len(v) for v in dataset.qrels.values())} judgments",
        flush=True,
    )

    results = []
    for system in baseline_systems():
        print(f"Running {system.name} ...", flush=True)
        result = run_system(system, dataset)
        print(
            f"  nDCG@10 {result.metrics['ndcg@10']:.4f}  "
            f"Recall@100 {result.metrics['recall@100']:.4f}",
            flush=True,
        )
        results.append(result)

    print()
    print(to_markdown(results))

    out = args.out or RESULTS_DIR / f"{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "documents": len(dataset.documents),
                "queries": len(dataset.queries),
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
                "results": [
                    {
                        "system": r.system,
                        "notes": r.notes,
                        **r.metrics,
                        "index_seconds": round(r.index_seconds, 2),
                        "ms_per_query": round(r.extra["ms_per_query"], 2),
                    }
                    for r in results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
