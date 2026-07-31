"""Where the corpus cache and the results files live.

The package was written inside a checkout, where `data/` and `results/` sit
next to `src/` and the committed results are what `--check` compares against.
Installed from a wheel there is no checkout: the same relative walk lands in
the interpreter's `site-packages` parent, so a plain `pip install` followed by
a benchmark run would download a corpus into the library directory, or fail
outright where that directory is not writable.

So the location is resolved rather than assumed. A checkout is identified by
its `pyproject.toml`; anywhere else the working directory is used, which is
what a command-line tool is expected to do and keeps the run inspectable.
`RAG_ABLATIONS_HOME` overrides both, for CI or a read-only working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "RAG_ABLATIONS_HOME"


def project_root() -> Path:
    """The directory `data/` and `results/` hang off."""
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()

    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout

    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def results_dir() -> Path:
    return project_root() / "results"


def embeddings_dir() -> Path:
    return data_dir() / "embeddings"
