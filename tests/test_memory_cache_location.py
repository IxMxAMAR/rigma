"""Where the embedding model lives (owner report 2026-10-23).

The fastembed cache was `%TEMP%/fastembed_cache`, and %TEMP% is the directory
Windows empties. The owner's nomic model was gone, retrieval had silently
dropped to lexical-only, and the only trace was four ERROR lines from fastembed
about a tar.gz it could not find. A model is not a temporary file.

The resolution order matters in both directions: a fresh install must put the
model somewhere durable, and an install that already HAS the model under %TEMP%
must keep using it rather than re-downloading 500 MB.
"""
import os

from rigma import memory


def _make_cache(root, name="models--nomic-ai--nomic-embed-text-v1.5") -> str:
    path = os.path.join(str(root), "fastembed_cache")
    os.makedirs(os.path.join(path, name), exist_ok=True)
    return path


def test_fresh_install_uses_rigma_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    got = memory._embed_cache_dir()
    assert got == str(home / "fastembed_cache")
    assert not got.startswith(str(tmp_path / "temp")), \
        "the model must not live in the directory Windows empties"


def test_an_existing_temp_cache_keeps_working(monkeypatch, tmp_path):
    legacy = _make_cache(tmp_path / "temp")
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    assert memory._embed_cache_dir() == legacy


def test_rigma_home_wins_once_it_has_a_model(monkeypatch, tmp_path):
    _make_cache(tmp_path / "temp")
    here = _make_cache(tmp_path / "home")
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    assert memory._embed_cache_dir() == here


def test_an_empty_cache_dir_does_not_count_as_a_model(monkeypatch, tmp_path):
    # This is the exact state the owner's machine was in: the folder exists,
    # holds nothing, and used to read as "cached" to anything that only checked
    # the path.
    empty = tmp_path / "temp" / "fastembed_cache"
    empty.mkdir(parents=True)
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    assert memory._cache_has_model(str(empty)) is False
    assert memory._embed_cache_dir() == str(tmp_path / "home" / "fastembed_cache")
