"""Engine bootstrap: pin upgrades, manifest provenance, archive extraction.

Covers the three findings that make `ensure_engine` unsafe or unusable:
F25 (a pin bump could never install), F26 (a fetched manifest could point the
downloader anywhere) and the runtime half of F29 (the tarfile `data` filter
does not exist before 3.11.4).
"""
import io
import json
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from rigma import runtime


def _manifest(version: str) -> dict:
    """A pin bump exactly as the registry publishes one.

    The build id appears in `url_base` *and* in every asset filename, so
    bumping it always changes the bytes stored under an unchanged os/backend
    key — which is the whole mechanism behind F25. Derived from the packaged
    manifest rather than hand-written so the shape can never drift from it.
    """
    packaged = runtime._engines_manifest()
    return json.loads(json.dumps(packaged).replace(packaged["version"], version))


def _zip_bytes(marker: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("llama-server.exe", marker.encode())
    return buf.getvalue()


def _fetch_marked(fetched: list):
    """A _fetch double whose payload differs per URL, like the real releases."""
    def _f(url, dest):
        fetched.append(url)
        dest.write_bytes(_zip_bytes(url))
    return _f


def _tar_bytes(members: "dict[str, bytes]") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            t.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def _response(payload) -> httpx.Response:
    # a real Response: raise_for_status() needs the request bound to it
    return httpx.Response(
        200, json=payload,
        request=httpx.Request("GET", runtime.ENGINES_MANIFEST_URL))


# --- F25 -------------------------------------------------------------------

def test_a_pin_bump_installs_the_new_engine(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    fetched: list = []
    monkeypatch.setattr(runtime, "_fetch", _fetch_marked(fetched))

    old, new = _manifest("b9867"), _manifest("b9900")
    assert old["assets"]["windows/vulkan"] != new["assets"]["windows/vulkan"]

    monkeypatch.setattr(runtime, "_engines_manifest", lambda: old)
    first = runtime.ensure_engine("vulkan", "windows")
    monkeypatch.setattr(runtime, "_engines_manifest", lambda: new)
    second = runtime.ensure_engine("vulkan", "windows")

    assert second.exists() and second != first
    assert "b9900" in str(second)
    lock = json.loads((home / "engines" / "lock.json").read_text())
    assert len(lock) == 2, f"the two pins must not share a lock key: {lock}"
    assert len(fetched) == 2


def test_the_same_pin_serving_different_bytes_is_still_refused(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    man = _manifest("b9867")
    monkeypatch.setattr(runtime, "_engines_manifest", lambda: man)
    monkeypatch.setattr(runtime, "_fetch", _fetch_marked([]))
    exe = runtime.ensure_engine("vulkan", "windows")

    # force a re-download of the *same* pin, and serve different bytes for it
    root = exe.parent
    for p in sorted(root.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    monkeypatch.setattr(
        runtime, "_fetch",
        lambda url, dest: dest.write_bytes(_zip_bytes("tampered")))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runtime.ensure_engine("vulkan", "windows")
    assert not list(root.glob("*.zip")), "the rejected archive must be removed"


# --- F26 -------------------------------------------------------------------

def test_update_refuses_a_manifest_that_downloads_from_an_unlisted_host(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("RIGMA_HOME", str(home))
    good = _manifest("b9900")
    evil = dict(good, url_base="https://evil.example/releases/download/b9900/")

    monkeypatch.setattr(runtime.httpx, "get", lambda *a, **k: _response(evil))
    assert runtime.update_engines_manifest() is False
    assert not (home / "engines.json").exists()

    monkeypatch.setattr(runtime.httpx, "get", lambda *a, **k: _response(good))
    assert runtime.update_engines_manifest() is True
    assert json.loads((home / "engines.json").read_text())["version"] == "b9900"


def test_update_refuses_an_asset_name_that_escapes_the_url_base(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("RIGMA_HOME", str(home))
    man = _manifest("b9900")
    man["assets"]["windows/vulkan"] = "../../../evil/payload.zip"
    monkeypatch.setattr(runtime.httpx, "get", lambda *a, **k: _response(man))
    assert runtime.update_engines_manifest() is False
    assert not (home / "engines.json").exists()


def test_a_stored_manifest_from_an_unlisted_host_does_not_outrank_the_wheel(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("RIGMA_HOME", str(home))
    evil = dict(_manifest("b9999"), url_base="https://evil.example/dl/")
    (home / "engines.json").write_text(json.dumps(evil), encoding="utf-8")

    man = runtime._engines_manifest()
    assert man["version"] != "b9999"
    assert man["url_base"].startswith(runtime.ENGINE_URL_ALLOWLIST)


def test_ensure_engine_never_fetches_outside_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    evil = dict(_manifest("b9900"), url_base="https://evil.example/dl/")
    monkeypatch.setattr(runtime, "_engines_manifest", lambda: evil)
    monkeypatch.setattr(runtime, "_fetch", lambda url, dest: pytest.fail(
        f"downloaded from a manifest that is not on the allowlist: {url}"))
    with pytest.raises(RuntimeError, match="url_base|allowlist"):
        runtime.ensure_engine("vulkan", "windows")


# --- F29 (runtime.py half) -------------------------------------------------

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_tarball_extracts_without_the_data_filter(tmp_path, monkeypatch):
    # 3.11.0-3.11.3 have neither tarfile.data_filter nor extractall(filter=)
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "engine.tar.gz"
    archive.write_bytes(_tar_bytes({"build/bin/llama-server": b"fake-binary"}))
    dest = tmp_path / "out"
    dest.mkdir()
    runtime._extract(archive, dest)
    assert (dest / "build" / "bin" / "llama-server").read_bytes() == b"fake-binary"


def test_a_traversing_tar_member_is_refused_without_the_data_filter(
        tmp_path, monkeypatch):
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "engine.tar.gz"
    archive.write_bytes(_tar_bytes({"../escaped.txt": b"pwn"}))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError):
        runtime._extract(archive, dest)
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.skipif(not hasattr(tarfile, "data_filter"),
                    reason="interpreter predates the data filter")
def test_a_traversing_tar_member_is_a_runtime_error_with_the_data_filter(tmp_path):
    # the CLI fallback ladder catches RuntimeError only; tarfile.FilterError
    # would sail straight past it
    archive = tmp_path / "engine.tar.gz"
    archive.write_bytes(_tar_bytes({"../escaped.txt": b"pwn"}))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError):
        runtime._extract(archive, dest)
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_both_extract_paths_agree_on_a_clean_archive(tmp_path, monkeypatch):
    """The fallback must not be a downgrade in behaviour, only in the checks
    it can lean on."""
    archive = tmp_path / "engine.tar.gz"
    archive.write_bytes(_tar_bytes({"bin/llama-server": b"x",
                                    "bin/libggml.so": b"y"}))
    with_filter, without = tmp_path / "a", tmp_path / "b"
    with_filter.mkdir()
    without.mkdir()
    runtime._extract(archive, with_filter)
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    runtime._extract(archive, without)

    def tree(d: Path) -> list:
        return sorted(str(p.relative_to(d)) for p in d.rglob("*"))

    assert tree(with_filter) == tree(without)
