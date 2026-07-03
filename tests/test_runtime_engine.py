import io
import json
import zipfile

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
