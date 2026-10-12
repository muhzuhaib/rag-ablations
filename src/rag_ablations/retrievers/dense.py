"""Dense retrieval with local sentence-transformer embeddings.

No embedding API is used. That is partly the zero-cost constraint, but mainly
reproducibility: a reviewer with this repo and a CPU gets the same vectors and
therefore the same numbers, which is not true of a hosted embedding endpoint
that can change under a stable model name.

Exhaustive cosine search over the full matrix, not an ANN index. These corpora
are a few thousand documents; a HNSW index would add a dependency and an
approximation error to a search that already takes milliseconds. ANN belongs in
the service layer at a corpus size that justifies it, not in a benchmark whose
job is to isolate the effect of the *model*.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from .. import paths

# Small, fast, and the community default for CPU retrieval: 384 dimensions and
# a 256-token window. That window is short enough that it forces the chunking
# question rather than hiding it, which is a benchmark variable here.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Dense:
    """Bi-encoder retrieval: embed once, then cosine similarity per query."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
        cache: bool = True,
        name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = cache
        self.name = name or model_name.split("/")[-1]
        self.doc_ids: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode to unit-length vectors so cosine similarity is a dot product."""
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _cache_path(self, texts: list[str]) -> Path:
        """Key the cache on the model and the exact corpus contents.

        Hashing the text rather than the dataset name means a change to
        chunking invalidates the cache automatically. A stale embedding matrix
        silently scoring a different corpus would corrupt every result in the
        table.
        """
        digest = hashlib.sha256(self.model_name.encode())
        for text in texts:
            digest.update(text.encode("utf-8", "replace"))
            digest.update(b"\0")
        return paths.embeddings_dir() / f"{digest.hexdigest()[:16]}.npy"

    def index(self, doc_ids: list[str], texts: list[str]) -> "Dense":
        self.doc_ids = list(doc_ids)

        path = self._cache_path(texts)
        if self.cache:
            cached = self._load_cached(path, len(self.doc_ids))
            if cached is not None:
                self._matrix = cached
                return self

        self._matrix = self._encode(list(texts))

        if self.cache:
            self._save_cached(path, self._matrix)
        return self

    @staticmethod
    def _load_cached(path: Path, rows: int) -> np.ndarray | None:
        """The cached matrix, or None if it is missing or unusable.

        A file that cannot be read, or whose row count does not match the
        corpus, is treated as a miss and re-encoded rather than trusted: a
        truncated file would otherwise fail every later run, and a matrix of
        the wrong height would pair vectors with the wrong documents.
        """
        if not path.exists():
            return None
        try:
            matrix = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        if matrix.ndim != 2 or matrix.shape[0] != rows:
            return None
        return matrix

    @staticmethod
    def _save_cached(path: Path, matrix: np.ndarray) -> None:
        """Write to a temporary file and rename, so a reader never sees half a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, matrix)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        scores = self._matrix @ self._encode([query])[0]
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]
