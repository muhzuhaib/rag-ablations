"""Download verification tests.

The network is stubbed: what is tested is that an archive is only unpacked
when it matches its published checksum.
"""

import hashlib
import io
import zipfile

import pytest

from rag_ablations import corpus


def _archive(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(f"{name}/corpus.jsonl", "")
    return buffer.getvalue()


@pytest.fixture
def serve(monkeypatch):
    def install(payload):
        monkeypatch.setattr(
            corpus.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
        )

    return install


def test_a_matching_archive_is_unpacked(serve, monkeypatch, tmp_path):
    payload = _archive("scifact")
    monkeypatch.setitem(corpus.SHA256, "scifact", hashlib.sha256(payload).hexdigest())
    serve(payload)
    target = corpus.download("scifact", tmp_path)
    assert (target / "corpus.jsonl").is_file()
    assert not list(tmp_path.glob("*.part"))


def test_a_mismatched_archive_is_rejected_and_discarded(serve, tmp_path):
    serve(_archive("scifact"))
    with pytest.raises(corpus.ChecksumError):
        corpus.download("scifact", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_every_dataset_has_a_checksum():
    assert set(corpus.SHA256) == set(corpus.DATASETS)
