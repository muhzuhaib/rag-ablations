"""A retrieval API over the benchmarked pipeline.

Deliberately thin. The interesting claims in this project are in the results
table, and a service is only worth including if it runs *the same code* that
was measured. Otherwise the benchmark describes something the API does not do.
So the retriever here is built by the same factory the benchmark uses, and the
configuration it ran under is exposed at `/health`.

    uvicorn rag_ablations.service:app

Environment:
    RAG_DATASET   corpus to index at startup (default: scifact)
    RAG_SYSTEM    "bm25" (default) or "hybrid"; hybrid needs the dense extra
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import corpus
from .retrievers import BM25, RRF, Dense

STATE: dict = {}


def build_retriever(system: str):
    if system == "bm25":
        return BM25()
    if system == "hybrid":
        return RRF([BM25(), Dense()])
    raise ValueError(f"Unknown system {system!r}; expected 'bm25' or 'hybrid'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Index at startup, not on the first request.

    Indexing takes seconds, and doing it lazily would hand the first user of a
    fresh container a request that appears to hang. Failing at startup also
    means a bad configuration is caught by the health check rather than by a
    caller.
    """
    dataset_name = os.environ.get("RAG_DATASET", "scifact")
    system = os.environ.get("RAG_SYSTEM", "bm25")

    started = time.perf_counter()
    dataset = corpus.load(dataset_name)
    retriever = build_retriever(system)
    retriever.index(
        [d.doc_id for d in dataset.documents], [d.content for d in dataset.documents]
    )

    STATE.update(
        retriever=retriever,
        documents={d.doc_id: d for d in dataset.documents},
        dataset=dataset_name,
        system=system,
        index_seconds=round(time.perf_counter() - started, 2),
    )
    yield
    STATE.clear()


app = FastAPI(
    title="rag-ablations",
    description="Retrieval over a benchmarked pipeline",
    lifespan=lifespan,
)


class Hit(BaseModel):
    doc_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    system: str
    took_ms: float
    hits: list[Hit]


@app.get("/health")
def health() -> dict:
    """What is indexed and under which configuration."""
    if not STATE:
        raise HTTPException(status_code=503, detail="index not ready")
    return {
        "status": "ok",
        "dataset": STATE["dataset"],
        "system": STATE["system"],
        "documents": len(STATE["documents"]),
        "index_seconds": STATE["index_seconds"],
    }


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="query text"),
    k: int = Query(10, ge=1, le=100),
) -> SearchResponse:
    if not STATE:
        raise HTTPException(status_code=503, detail="index not ready")

    started = time.perf_counter()
    results = STATE["retriever"].search(q, k=k)
    took_ms = (time.perf_counter() - started) * 1000

    hits = []
    for doc_id, score in results:
        document = STATE["documents"].get(doc_id)
        hits.append(
            Hit(
                doc_id=doc_id,
                score=round(score, 5),
                title=document.title if document else "",
                snippet=(document.text[:280] if document else ""),
            )
        )

    return SearchResponse(
        query=q, system=STATE["system"], took_ms=round(took_ms, 2), hits=hits
    )
