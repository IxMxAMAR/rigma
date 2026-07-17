"""Hangar: custom-model install, model listing, deletion, capability edits."""
import struct

import pytest

from rigma import hangar
from rigma.hangar import HangarError
from rigma.registry import Registry

T_U32, T_STR, T_ARR = 4, 8, 9


def _s(b):
    return struct.pack("<Q", len(b)) + b


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", T_U32) + struct.pack("<I", v)


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", T_STR) + _s(v)


def _gguf_bytes(kvs):
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _dense_gguf(tmp_path, name=b"Spicy Tune 8B", fname="SpicyTune-Q4_K_M.gguf"):
    p = tmp_path / fname
    p.write_bytes(_gguf_bytes([
        _kv_str(b"general.architecture", b"qwen3"),
        _kv_str(b"general.name", name),
        _kv_u32(b"qwen3.block_count", 8),
        _kv_u32(b"qwen3.context_length", 32768),
        _kv_u32(b"qwen3.embedding_length", 1024),
        _kv_u32(b"qwen3.attention.head_count", 16),
        _kv_u32(b"qwen3.attention.head_count_kv", 2),
        _kv_str(b"tokenizer.chat_template", b"{% if tools %}x{% endif %}"),
    ]) + b"\x00" * 512)
    return p


def _mmproj_gguf(tmp_path, fname="mmproj-spicy.gguf"):
    p = tmp_path / fname
    p.write_bytes(_gguf_bytes([
        _kv_str(b"general.architecture", b"clip"),
        _kv_u32(b"clip.vision.image_size", 768),
    ]))
    return p


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_install_moves_file_writes_spec_and_registry_sees_it(home, tmp_path):
    src = _dense_gguf(tmp_path)
    spec = hangar.install_model(src)
    assert spec.slug == "spicy-tune-8b"
    assert not src.exists()                                  # moved, not copied
    assert (home / "models" / "SpicyTune-Q4_K_M.gguf").exists()
    assert spec.ggufs[0].quant == "Q4_K_M"                  # from filename
    assert spec.capabilities == ["tools"]
    reg = Registry.load()
    assert reg.models["spicy-tune-8b"].custom is True
    assert reg.models["spicy-tune-8b"].native_ctx == 32768


def test_install_rejects_non_gguf_and_missing(home, tmp_path):
    with pytest.raises(HangarError, match="no such file"):
        hangar.install_model(tmp_path / "ghost.gguf")
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with pytest.raises(HangarError, match=".gguf"):
        hangar.install_model(bad)
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(HangarError, match="not a GGUF"):
        hangar.install_model(junk)
    assert junk.exists()                     # failed install never eats a file


def test_install_rejects_slug_collision_with_registry(home, tmp_path):
    src = _dense_gguf(tmp_path, name=b"qwen3.6 35b a3b")
    with pytest.raises(HangarError, match="already exists"):
        hangar.install_model(src)
    assert src.exists()


def test_mmproj_requires_attach_and_grants_vision(home, tmp_path):
    mm = _mmproj_gguf(tmp_path)
    with pytest.raises(HangarError, match="mmproj"):
        hangar.install_model(mm)
    hangar.install_model(_dense_gguf(tmp_path))
    spec = hangar.install_model(mm, attach_to="spicy-tune-8b")
    assert spec.mmproj is not None and "vision" in spec.capabilities
    assert (home / "models" / "mmproj-spicy.gguf").exists()
    with pytest.raises(HangarError, match="not a custom model"):
        hangar.install_model(_mmproj_gguf(tmp_path, "mm2.gguf"),
                             attach_to="qwen3.6-35b-a3b")


def test_list_models_reports_disk_and_running(home, tmp_path):
    hangar.install_model(_dense_gguf(tmp_path))
    from rigma import state as st
    import os
    st.write_state("spicy-tune-8b", "Q4_K_M", 11500,
                   engine_pid=os.getpid(), ui_pid=os.getpid())
    out = hangar.list_models()
    by_slug = {m["slug"]: m for m in out["models"]}
    me = by_slug["spicy-tune-8b"]
    assert me["custom"] and me["running"]
    assert me["quants"][0]["on_disk"] is True
    assert by_slug["qwen3.6-35b-a3b"]["running"] is False
    assert out["disk"]["free_gb"] > 0


def test_delete_refuses_running_file_then_deletes_when_stopped(home, tmp_path):
    hangar.install_model(_dense_gguf(tmp_path))
    from rigma import state as st
    import os
    st.write_state("spicy-tune-8b", "Q4_K_M", 11500,
                   engine_pid=os.getpid(), ui_pid=os.getpid())
    with pytest.raises(HangarError, match="running"):
        hangar.delete_file("spicy-tune-8b", "SpicyTune-Q4_K_M.gguf")
    st.clear_state()
    hangar.delete_file("spicy-tune-8b", "SpicyTune-Q4_K_M.gguf")
    assert not (home / "models" / "SpicyTune-Q4_K_M.gguf").exists()


def test_delete_model_removes_spec_and_files_custom_only(home, tmp_path):
    hangar.install_model(_dense_gguf(tmp_path))
    hangar.install_model(_mmproj_gguf(tmp_path), attach_to="spicy-tune-8b")
    hangar.delete_model("spicy-tune-8b")
    assert "spicy-tune-8b" not in Registry.load().models
    assert not (home / "models" / "SpicyTune-Q4_K_M.gguf").exists()
    assert not (home / "models" / "mmproj-spicy.gguf").exists()
    with pytest.raises(HangarError, match="custom"):
        hangar.delete_model("qwen3.6-35b-a3b")


def test_patch_capabilities_custom_only_vision_needs_mmproj(home, tmp_path):
    hangar.install_model(_dense_gguf(tmp_path))
    spec = hangar.patch_capabilities("spicy-tune-8b", ["tools", "thinking"])
    assert spec.capabilities == ["thinking", "tools"]
    with pytest.raises(HangarError, match="mmproj"):
        hangar.patch_capabilities("spicy-tune-8b", ["vision"])
    with pytest.raises(HangarError, match="custom"):
        hangar.patch_capabilities("qwen3.6-35b-a3b", ["tools"])
    with pytest.raises(HangarError, match="unknown"):
        hangar.patch_capabilities("spicy-tune-8b", ["telepathy"])


def test_registry_wins_slug_collision_on_load(home, tmp_path):
    hangar.custom_dir().mkdir(parents=True, exist_ok=True)
    (hangar.custom_dir() / "evil.json").write_text(
        '{"slug": "qwen3.6-35b-a3b", "family": "x", "kind": "dense",'
        '"n_layers": 1, "full_attn_layers": 1, "kv_heads": 1, "head_dim": 1,'
        '"native_ctx": 2048, "ggufs": [], "custom": true}', encoding="utf-8")
    reg = Registry.load()
    assert reg.models["qwen3.6-35b-a3b"].custom is False    # bundled spec wins


def test_install_refuses_filename_collision_on_disk(home, tmp_path):
    """Review 2026-07-17 CRITICAL: same filename, different general.name —
    the move must never clobber a file another model already owns."""
    hangar.install_model(_dense_gguf(tmp_path))
    src2 = _dense_gguf(tmp_path, name=b"Totally Different Model")
    with pytest.raises(HangarError, match="already exists in Rigma"):
        hangar.install_model(src2)
    assert src2.exists()                        # source untouched
    spec = hangar._load_custom("spicy-tune-8b")
    assert spec is not None                     # original spec intact


def test_ensure_model_short_circuits_disk_and_never_fetches_local(home,
                                                                  tmp_path):
    """Review 2026-07-17 CRITICAL: repo='local' must never reach HF, and any
    on-disk file returns without a network call."""
    import rigma.runtime as runtime
    from rigma.models import GgufFile

    def _boom(**kw):
        raise AssertionError("hf_hub_download must not be called")
    orig = runtime.hf_hub_download
    runtime.hf_hub_download = _boom
    try:
        f = hangar.models_dir() / "onDisk.gguf"
        f.write_bytes(b"x")
        got = runtime.ensure_model(GgufFile(repo="unsloth/whatever",
                                            file="onDisk.gguf", bytes=1,
                                            quant="Q4"))
        assert got == f
        got = runtime.ensure_model(GgufFile(repo="local", file="onDisk.gguf",
                                            bytes=1, quant="LOCAL"))
        assert got == f
        with pytest.raises(RuntimeError, match="local-only"):
            runtime.ensure_model(GgufFile(repo="local", file="gone.gguf",
                                          bytes=1, quant="LOCAL"))
    finally:
        runtime.hf_hub_download = orig


def test_list_models_marks_pullable(home, tmp_path):
    """User-reported 2026-07-18: no download option for added/registry models.
    Registry + HF-added models have a real repo (pullable); drag-dropped ones
    are repo='local' (not)."""
    hangar.install_model(_dense_gguf(tmp_path))     # drag-drop -> repo 'local'
    out = hangar.list_models()
    by = {m["slug"]: m for m in out["models"]}
    # a bundled registry model's quants are downloadable
    reg_q = by["qwen3.6-35b-a3b"]["quants"][0]
    assert reg_q["pullable"] is True and reg_q["on_disk"] is False
    # the drag-dropped custom's quant is local-only (not re-downloadable)
    assert by["spicy-tune-8b"]["quants"][0]["pullable"] is False


def test_list_models_mmproj_pullable(home, tmp_path):
    """User-reported 2026-07-18: downloaded the model but no way to get its
    vision projector. mmproj of an HF-added model must be downloadable."""
    from rigma.models import GgufFile
    hangar.install_model(_dense_gguf(tmp_path))
    # simulate an HF-added model with a real-repo mmproj (not local)
    spec = hangar._load_custom("spicy-tune-8b")
    spec = spec.model_copy(update={
        "mmproj": GgufFile(repo="some/hf-repo", file="mmproj-x.gguf",
                           bytes=900 * 2**20, quant="F16"),
        "capabilities": sorted(set(spec.capabilities) | {"vision"})})
    hangar._write_spec(spec)
    out = hangar.list_models()
    mm = {m["slug"]: m for m in out["models"]}["spicy-tune-8b"]["mmproj"]
    assert mm is not None and mm["on_disk"] is False
    assert mm["pullable"] is True     # real repo -> a Download button appears


def test_download_file_streams_with_progress_and_resume(home, tmp_path,
                                                        monkeypatch):
    """User-reported 2026-07-18: download progress stuck at 'connecting…'
    because hf's partials are hashed. The direct downloader reports exact
    bytes and resumes from a .part file."""
    import httpx
    from rigma import hangar

    class _Resp:
        def __init__(self, code, chunks, hdrs=None):
            self.status_code = code
            self._chunks = chunks
            self.headers = hdrs or {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_bytes(self, n):
            yield from self._chunks

    # fresh download: 200 with 3 x 1MB chunks
    seen = []
    monkeypatch.setattr(hangar.httpx if hasattr(hangar, "httpx") else httpx,
                        "stream", lambda *a, **k: _Resp(200, [b"x" * 2**20] * 3))
    import rigma.hangar as H
    monkeypatch.setattr("httpx.stream",
                        lambda *a, **k: _Resp(200, [b"x" * 2**20] * 3))
    dest = tmp_path / "m.gguf"
    n = H._download_file("r/x", "m.gguf", dest, lambda b: seen.append(b))
    assert n == 3 * 2**20 and dest.exists() and dest.stat().st_size == n
    assert seen and seen[-1] == n and seen == sorted(seen)   # monot, exact
    assert not (tmp_path / "m.gguf.part").exists()           # renamed cleanly

    # resume: a .part already has 1MB, server returns 206 with the remaining 2MB
    dest2 = tmp_path / "r.gguf"
    (tmp_path / "r.gguf.part").write_bytes(b"y" * 2**20)
    ranged = {}
    def _stream(method, url, headers=None, **k):
        ranged["range"] = (headers or {}).get("range")
        return _Resp(206, [b"z" * 2**20] * 2)
    monkeypatch.setattr("httpx.stream", _stream)
    n2 = H._download_file("r/x", "r.gguf", dest2, lambda b: None)
    assert ranged["range"] == "bytes=1048576-"        # asked to resume
    assert n2 == 3 * 2**20 and dest2.stat().st_size == n2   # 1 + 2 MB


def test_pull_progress_reads_live_bytes(home):
    from rigma import hangar
    hangar._PULLS["slugX::big.gguf"] = {"status": "downloading",
                                        "total": 100, "done": 42}
    try:
        assert hangar.pull_progress("big.gguf", 100) == 42
        assert hangar.pull_progress("other.gguf", 100) == 0
    finally:
        hangar._PULLS.pop("slugX::big.gguf", None)
