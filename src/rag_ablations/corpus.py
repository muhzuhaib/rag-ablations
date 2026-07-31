"""BEIR corpus loading.

Only the stdlib is used to fetch and parse datasets: no `datasets`, no
`beir`. Those pull a large dependency tree for what is, in the end, three
files of JSONL and TSV, and every extra dependency is another way for a
reviewer's `pip install` to fail.

Datasets are cached under `data/` and never re-downloaded.
"""

from __future__ import annotations

import json
import shutil
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi

from . import paths

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# Deliberately small corpora. Both fit in memory and embed on a CPU in minutes,
# which is what keeps the benchmark reproducible without a GPU.
DATASETS = {
    "scifact": "Scientific claim verification. 5k abstracts, 300 test claims.",
    "nfcorpus": "Medical information retrieval. 3.6k docs, 323 test queries.",
}

def _default_data_dir() -> Path:
    """Resolved per call, not at import, so a test or CI can move it."""
    return paths.data_dir()


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """Title and body joined, because BEIR's own baselines index both."""
        return f"{self.title}\n\n{self.text}".strip()


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Dataset({self.name!r}, {len(self.documents)} docs, "
            f"{len(self.queries)} queries)"
        )


def download(name: str, data_dir: Path | None = None) -> Path:
    """Fetch and unzip a BEIR dataset, returning its directory. Idempotent."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}")

    root = data_dir or _default_data_dir()
    target = root / name
    if target.exists():
        return target

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}.zip.part"

    # Python does not use the operating system trust store on Windows and does
    # not chase AIA links to fetch missing intermediate certificates, so the
    # default context fails on hosts that browsers and curl accept. Pinning
    # certifi's bundle makes the download behave the same on every platform.
    context = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(
        BEIR_URL.format(name=name), context=context, timeout=60
    ) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)

    # Extract to a temporary directory first so an interrupted run cannot leave
    # a half-populated dataset that the `target.exists()` check would then
    # treat as a valid cache.
    staging = root / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    (staging / name).rename(target)

    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink()
    return target


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load(name: str, split: str = "test", data_dir: Path | None = None) -> Dataset:
    """Load a BEIR dataset, downloading it on first use.

    Queries are filtered to those that actually appear in the qrels for the
    split. BEIR ships the full query file regardless of split, and scoring
    queries with no judgments would silently dilute every metric.
    """
    directory = download(name, data_dir)

    documents = [
        Document(
            doc_id=row["_id"],
            title=row.get("title", ""),
            text=row.get("text", ""),
        )
        for row in _read_jsonl(directory / "corpus.jsonl")
    ]

    qrels: dict[str, dict[str, int]] = {}
    with (directory / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        header = next(fh)
        if not header.lower().startswith("query-id"):
            fh.seek(0)
        for line in fh:
            if not line.strip():
                continue
            query_id, doc_id, score = line.split("\t")[:3]
            gain = int(score)
            if gain > 0:
                qrels.setdefault(query_id, {})[doc_id] = gain

    queries = {
        row["_id"]: row["text"]
        for row in _read_jsonl(directory / "queries.jsonl")
        if row["_id"] in qrels
    }

    return Dataset(name=name, documents=documents, queries=queries, qrels=qrels)
