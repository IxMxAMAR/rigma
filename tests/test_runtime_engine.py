import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from rigma import runtime


@pytest.fixture
def fake_engine_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("llama-server.exe", b"fake-binary")
    p = tmp_path / "llama-fake-bin-windows-vulkan-x64.zip"
    p.write_bytes(buf.getvalue())
    return p


def test_ensure_engine_downloads_extracts_and_locks(tmp_path, monkeypatch, fake_engine_zip):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime, "_fetch", lambda url, dest: dest.write_bytes(
        fake_engine_zip.read_bytes()))
    exe = runtime.ensure_engine("vulkan", "windows")
    assert exe.name == "llama-server.exe" and exe.exists()
    lock = json.loads((tmp_path / "home" / "engines" / "lock.json").read_text())
    assert any(v.get("sha256") for v in lock.values())
    # second call: no re-download (fetch would blow up), cached path returned
    monkeypatch.setattr(runtime, "_fetch",
                        lambda *a: (_ for _ in ()).throw(AssertionError))
    assert runtime.ensure_engine("vulkan", "windows") == exe


def test_a_torn_lock_json_does_not_kill_ensure_engine(tmp_path, monkeypatch,
                                                      fake_engine_zip):
    """AUDIT F08-4: a crash mid-write left lock.json unparseable and every later
    `rigma up` died on it with a raw JSONDecodeError."""
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    (home / "engines").mkdir(parents=True)
    (home / "engines" / "lock.json").write_text('{"b9867:windows/cpu:llama-b9')
    monkeypatch.setattr(runtime, "_fetch", lambda url, dest: dest.write_bytes(
        fake_engine_zip.read_bytes()))

    exe = runtime.ensure_engine("vulkan", "windows")

    assert exe.exists()
    lock = json.loads((home / "engines" / "lock.json").read_text())
    assert any(v.get("sha256") for v in lock.values())


def test_a_non_dict_lock_json_is_rebuilt_too(tmp_path, monkeypatch,
                                             fake_engine_zip):
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    (home / "engines").mkdir(parents=True)
    (home / "engines" / "lock.json").write_text("[1, 2, 3]")
    monkeypatch.setattr(runtime, "_fetch", lambda url, dest: dest.write_bytes(
        fake_engine_zip.read_bytes()))

    assert runtime.ensure_engine("vulkan", "windows").exists()
    assert isinstance(json.loads(
        (home / "engines" / "lock.json").read_text()), dict)


def test_the_lock_is_written_atomically(tmp_path, monkeypatch, fake_engine_zip):
    """AUDIT F08-4: the lock must land via a temp + replace, so a crash cannot
    leave a half-written lock.json behind.

    R3-STORE-10: the assertion used to name the temp `lock.tmp`, which pinned the
    FIXED name the store was moved off — the temp is now unique per write, so the
    test checks the SHAPE (a temp beside the target, renamed onto it) rather than
    one spelling of it. `atomicio` replaces via `os.replace`, which `Path.replace`
    does not intercept, so the spy is on `os.replace`.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    monkeypatch.setattr(runtime, "_fetch", lambda url, dest: dest.write_bytes(
        fake_engine_zip.read_bytes()))
    replaced = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    runtime.ensure_engine("vulkan", "windows")

    lock_replaces = [t for t in replaced if t[1] == "lock.json"]
    assert lock_replaces, replaced
    tmp_name = lock_replaces[0][0]
    assert tmp_name != "lock.json" and tmp_name.endswith(".tmp"), tmp_name
    # the temp name must be unique per write, not a shared fixed one
    assert any(ch.isdigit() for ch in tmp_name) or "." in tmp_name.strip(".tmp")
    assert not (home / "engines" / tmp_name).exists()
