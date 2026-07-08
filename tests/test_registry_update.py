import io
import json
import zipfile

import pytest

from rigma import registry as reg_mod
from rigma.registry import Registry, update_registry


def _fake_repo_zip() -> bytes:
    buf = io.BytesIO()
    gpus = [{"match": "RTX 5090", "vendor": "nvidia", "arch": "blackwell"}]
    model = {
        "slug": "test-model", "family": "t", "kind": "dense", "n_layers": 2,
        "full_attn_layers": 2, "kv_heads": 1, "head_dim": 64, "native_ctx": 8192,
        "ggufs": [{"repo": "r/x", "file": "f.gguf", "bytes": 2000000, "quant": "Q8_0"}],
    }
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rigma-registry-master/gpus.json", json.dumps(gpus))
        z.writestr("rigma-registry-master/models/test-model.json", json.dumps(model))
        z.writestr("rigma-registry-master/combos/_class/vram-32/ram-64/general.json",
                   json.dumps({"model": "test-model", "quant": "Q8_0",
                               "backend": "cuda", "flags": {"ctx": 8192}}))
    return buf.getvalue()


def test_update_registry_installs_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_REGISTRY_DIR", raising=False)
    blob = _fake_repo_zip()
    monkeypatch.setattr(reg_mod, "_fetch_bytes", lambda url: blob)
    dest = update_registry()
    assert (dest / "gpus.json").exists()
    r = Registry.load()  # cache tier now wins over bundled
    assert "test-model" in r.models
    assert any(row["match"] == "RTX 5090" for row in r.gpus)


def test_load_falls_back_to_bundled_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_REGISTRY_DIR", raising=False)
    r = Registry.load()
    assert "qwen3.6-35b-a3b" in r.models  # bundled snapshot


def test_update_is_atomic_on_bad_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(reg_mod, "_fetch_bytes", lambda url: b"not a zip")
    with pytest.raises(Exception):
        update_registry()
    assert not (tmp_path / "registry" / "gpus.json").exists()
