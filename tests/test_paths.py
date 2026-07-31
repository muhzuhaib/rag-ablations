"""Where the corpus cache and the results land, in a checkout and out of one.

These matter because the failure they guard against is invisible in
development: inside the repo every path resolves correctly, and only a user who
installed the wheel would find the benchmark writing a corpus into the
interpreter's library directory.
"""

import sys
from pathlib import Path

from rag_ablations import paths


def test_a_checkout_is_found_by_its_pyproject():
    # The repo the tests are running from is itself the case being asserted.
    root = paths.project_root()
    assert (root / "pyproject.toml").is_file()
    assert paths.data_dir() == root / "data"
    assert paths.results_dir() == root / "results"


def test_embeddings_live_under_the_data_directory():
    assert paths.embeddings_dir().parent == paths.data_dir()


def test_an_installed_package_uses_the_working_directory(tmp_path, monkeypatch):
    """Installed from a wheel there is no pyproject.toml above the package.

    Simulated by pointing the module's own file at a directory with no
    pyproject above it, which is exactly the shape of `site-packages`.
    """
    fake_site_packages = tmp_path / "lib" / "site-packages" / "rag_ablations"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(paths, "__file__", str(fake_site_packages / "paths.py"))

    working = tmp_path / "somewhere-else"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.delenv(paths.HOME_ENV, raising=False)

    assert paths.project_root() == Path.cwd()
    assert paths.data_dir() == working / "data"
    # The point of the whole module: nothing is written next to the install.
    assert Path(sys.prefix) not in paths.data_dir().parents


def test_the_environment_variable_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    assert paths.project_root() == tmp_path.resolve()
    assert paths.results_dir() == tmp_path.resolve() / "results"
