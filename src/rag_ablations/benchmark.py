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

from . import chunking, corpus, paths
from .metrics import evaluate
from .chunking import Chunked
from .retrievers import BM25, RRF, Dense, Reranked

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


def dense_systems() -> list[System]:
    """Dense, hybrid and reranked systems. Requires the `dense` extra."""
    return [
        System(
            name="Dense (MiniLM-L6)",
            build=lambda: Dense(),
            notes="all-MiniLM-L6-v2, 384-dim, cosine over the full matrix",
        ),
        System(
            name="Hybrid (BM25 + dense, RRF)",
            build=lambda: RRF([BM25(), Dense()]),
            notes="Reciprocal rank fusion, rrf_k=60, untuned",
        ),
        System(
            name="BM25 + cross-encoder rerank",
            build=lambda: Reranked(BM25(), depth=100),
            notes="ms-marco-MiniLM-L-6-v2 over the top 100",
        ),
        System(
            name="Hybrid + cross-encoder rerank",
            build=lambda: Reranked(RRF([BM25(), Dense()]), depth=100),
            notes="Same reranker over the fused top 100",
        ),
    ]


def chunking_systems() -> list[System]:
    """Chunking held against a fixed retriever.

    Only the chunker varies, so any difference in the table is attributable to
    chunking rather than to chunking plus a different scoring function. BM25 is
    the fixed retriever because it needs no model download, which keeps this
    ablation runnable by anyone.
    """
    return [
        System(
            name="BM25 + whole document",
            build=lambda: Chunked(BM25(), chunking.whole_document, name="whole"),
            notes="Control condition, no chunking",
        ),
        System(
            name="BM25 + 128-word windows (32 overlap)",
            build=lambda: Chunked(BM25(), chunking.fixed_window(128, 32), name="w128"),
            notes="The size most tutorials default to",
        ),
        System(
            name="BM25 + 64-word windows (16 overlap)",
            build=lambda: Chunked(BM25(), chunking.fixed_window(64, 16), name="w64"),
            notes="Smaller windows, more precise but more diluted idf",
        ),
        System(
            name="BM25 + 3-sentence windows (stride 2)",
            build=lambda: Chunked(BM25(), chunking.sentences(3, 2), name="sent3"),
            notes="Sentence-aligned rather than word-count aligned",
        ),
    ]


def select_systems(group: str) -> list[System]:
    if group == "sparse":
        return baseline_systems()
    if group == "chunking":
        return chunking_systems()
    return baseline_systems() + dense_systems()


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


def check_against(path: Path, results: list[Result], tolerance: float) -> int:
    """Fail if a fresh run no longer reproduces the committed results.

    This is what stops the README from drifting away from the code. A refactor
    that quietly changes the analyzer or the idf formula still passes the unit
    tests, which assert properties rather than corpus-level scores, but it will move
    nDCG on 300 real queries, and this catches that.

    Timings are deliberately not checked: they vary with the runner.
    """
    if not path.exists():
        print(f"No stored results at {path} to check against.")
        return 1

    stored = {row["system"]: row for row in json.loads(path.read_text())["results"]}
    failures = []

    for result in results:
        expected = stored.get(result.system)
        if expected is None:
            failures.append(f"{result.system}: not present in {path.name}")
            continue
        drift = abs(expected["ndcg@10"] - result.metrics["ndcg@10"])
        if drift > tolerance:
            failures.append(
                f"{result.system}: nDCG@10 {result.metrics['ndcg@10']:.4f} "
                f"vs stored {expected['ndcg@10']:.4f} (drift {drift:.4f})"
            )

    if failures:
        print("\nResults no longer reproduce:")
        for failure in failures:
            print(f"  {failure}")
        print("\nIf the change was intended, rerun without --check to update them.")
        return 1

    print(f"\nAll {len(results)} systems reproduce {path.name} within {tolerance}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", choices=sorted(corpus.DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--systems",
        default="all",
        choices=["all", "sparse", "chunking"],
        help="'sparse' runs only the BM25 baselines and needs no model downloads",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the stored results and exit non-zero on drift, "
        "instead of overwriting them",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="allowed absolute nDCG@10 difference under --check",
    )
    args = parser.parse_args(argv)

    print(f"Loading {args.dataset} ...", flush=True)
    dataset = corpus.load(args.dataset, split=args.split)
    print(
        f"  {len(dataset.documents)} documents, {len(dataset.queries)} queries, "
        f"{sum(len(v) for v in dataset.qrels.values())} judgments",
        flush=True,
    )

    results = []
    for system in select_systems(args.systems):
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

    out = args.out or paths.results_dir() / f"{args.dataset}-{args.systems}.json"

    if args.check:
        return check_against(out, results, args.tolerance)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "systems": args.systems,
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
