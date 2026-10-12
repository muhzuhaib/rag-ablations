"""Embedding cache tests.

The model is stubbed: what can go wrong here is the cache, not the encoder.
"""

import numpy as np
import pytest

from rag_ablations import paths
from rag_ablations.retrievers import Dense


@pytest.fixture
def dense(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    retriever = Dense()
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.eye(len(texts), 4, dtype=np.float32)

    monkeypatch.setattr(retriever, "_encode", encode)
    retriever.calls = calls
    return retriever


def test_second_index_is_served_from_the_cache(dense):
    dense.index(["a", "b"], ["x", "y"])
    dense.index(["a", "b"], ["x", "y"])
    assert len(dense.calls) == 1
    assert dense._matrix.shape == (2, 4)


def test_cache_write_leaves_no_temporary_files(dense):
    dense.index(["a", "b"], ["x", "y"])
    files = list(paths.embeddings_dir().iterdir())
    assert [f.suffix for f in files] == [".npy"]


def test_a_truncated_cache_file_is_re_encoded(dense):
    path = dense._cache_path(["x", "y"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x93NUMPY")  # an interrupted write
    dense.index(["a", "b"], ["x", "y"])
    assert len(dense.calls) == 1
    assert np.load(path).shape == (2, 4)


def test_a_cache_of_the_wrong_height_is_not_trusted(dense):
    path = dense._cache_path(["x", "y"])
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((3, 4), dtype=np.float32))
    dense.index(["a", "b"], ["x", "y"])
    assert len(dense.calls) == 1
    assert dense._matrix.shape == (2, 4)
