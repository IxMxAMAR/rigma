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


# --- healing a spec written by an older probe ---------------------------------
def _hybrid_gguf(path, *, nextn: bool = True, template: bool = True):
    """Qwen3.5/3.8 shape: scalar kv head count + full_attention_interval, and a
    tensor table that may or may not carry the MTP projections."""
    def tensor(name, dims):
        return (_s(name) + struct.pack("<I", len(dims))
                + b"".join(struct.pack("<Q", d) for d in dims)
                + struct.pack("<I", 0) + struct.pack("<Q", 0))
    kvs = [
        _kv_str(b"general.architecture", b"qwen35"),
        _kv_str(b"general.name", b"Hybrid Tune"),
        _kv_u32(b"qwen35.block_count", 9),          # 8 real + 1 MTP block
        _kv_u32(b"qwen35.context_length", 262144),
        _kv_u32(b"qwen35.embedding_length", 512),
        _kv_u32(b"qwen35.attention.head_count", 8),
        _kv_u32(b"qwen35.attention.head_count_kv", 2),
        _kv_u32(b"qwen35.attention.key_length", 64),
        _kv_u32(b"qwen35.full_attention_interval", 4),
        _kv_u32(b"qwen35.nextn_predict_layers", 1),
    ]
    if template:
        kvs.append(_kv_str(b"tokenizer.chat_template", b"{% if tools %}x{% endif %}"))
    tensors = [tensor(b"blk.0.attn_q.weight", [8, 8])]
    if nextn:
        tensors.append(tensor(b"blk.8.nextn.eh_proj.weight", [8, 8]))
    path.write_bytes(b"GGUF" + struct.pack("<I", 3)
                     + struct.pack("<Q", len(tensors))
                     + struct.pack("<Q", len(kvs))
                     + b"".join(kvs) + b"".join(tensors))
    return path


def _stale_spec(home, fname, **over):
    """A spec as an older probe would have written it: every layer counted as
    full attention, the MTP block counted as a decoder layer, no probe_version."""
    import json
    d = home / "custom" / "models"
    d.mkdir(parents=True, exist_ok=True)
    spec = {"slug": "hybrid-tune", "family": "qwen35", "kind": "dense",
            "n_layers": 9, "full_attn_layers": 9, "kv_heads": 2,
            "head_dim": 64, "native_ctx": 262144,
            "ggufs": [{"repo": "local", "file": fname, "bytes": 4096,
                       "quant": "Q4_K_M"}],
            "capabilities": ["vision"], "custom": True}
    spec.update(over)
    (d / "hybrid-tune.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_heal_reprobes_a_spec_written_before_the_probe_improved(home, tmp_path):
    """Quant labels already healed on read; geometry did not, so a model added
    before a fix kept the wrong numbers forever and the only cure was to delete
    and re-add it. The APEX 35B on the owner's machine was stored as 41-of-41
    full-attention layers when the file says 10-of-40 — a 4x overstatement of
    its KV cache that had been capping its context since the day it was added."""
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    spec = Registry.load().models["hybrid-tune"]
    assert spec.full_attn_layers == 2      # 8 real layers // interval 4
    assert spec.n_layers == 8              # the MTP block is not a decoder layer
    assert spec.mtp_layers == 1
    assert spec.probe_version == hangar.PROBE_VERSION
    assert "tools" in spec.capabilities    # newly detected...
    assert "vision" in spec.capabilities   # ...without dropping a hand-set one


def test_heal_persists_so_the_next_load_is_free(home, tmp_path):
    import json
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    Registry.load()
    on_disk = json.loads(
        (home / "custom" / "models" / "hybrid-tune.json").read_text())
    assert on_disk["full_attn_layers"] == 2
    assert on_disk["probe_version"] == hangar.PROBE_VERSION


def test_heal_cannot_run_without_the_file_and_says_nothing_false(home):
    """Nothing downloaded yet: leave the spec alone rather than invent a
    correction. It heals on the next load after a pull."""
    _stale_spec(home, "not-downloaded.gguf")
    spec = Registry.load().models["hybrid-tune"]
    assert spec.full_attn_layers == 9       # untouched
    assert spec.probe_version == 0          # still owed a re-probe


def test_heal_drops_an_mtp_capability_the_file_disproves(home):
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf", nextn=False)
    _stale_spec(home, "h.gguf", capabilities=["mtp", "vision"])
    spec = Registry.load().models["hybrid-tune"]
    assert "mtp" not in spec.capabilities   # header claimed it, tensors deny it
    assert "vision" in spec.capabilities
    assert spec.ggufs[0].mtp is False


def test_heal_records_mtp_per_file(home):
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf", nextn=True)
    _stale_spec(home, "h.gguf")
    spec = Registry.load().models["hybrid-tune"]
    assert spec.ggufs[0].mtp is True
    assert "mtp" in spec.capabilities


def test_registry_models_are_never_rewritten_by_healing(home, tmp_path):
    """Registry entries are hand-authored and researched. Healing only ever
    touches custom imports."""
    from rigma.models import GgufFile, ModelSpec
    spec = ModelSpec(slug="curated", family="qwen3.6", kind="moe", n_layers=40,
                     full_attn_layers=10, kv_heads=2, head_dim=256,
                     native_ctx=262144, custom=False,
                     ggufs=[GgufFile(repo="r", file="x.gguf", bytes=1,
                                     quant="Q4_K_M")])
    assert hangar.heal_spec(spec) is spec


def test_missing_chat_template_is_surfaced_not_shown_as_no_capabilities(home,
                                                                       tmp_path):
    """jaromer's Qwen3.8-27B ships no tokenizer.chat_template, so it listed no
    capabilities at all next to unsloth's build of the same base model listing
    four. An empty list was the absence of evidence, presented as a finding."""
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf", template=False)
    _stale_spec(home, "h.gguf", capabilities=[])
    listed = hangar.list_models(Registry.load())["models"]
    row = next(m for m in listed if m["slug"] == "hybrid-tune")
    assert row["has_template"] is False
    assert row["capabilities"] == ["mtp"]   # from tensors, not from a template


# --- a gguf that does not say what it is --------------------------------------
def test_generic_gguf_names_fall_back_to_what_the_user_pointed_at():
    """A model is keyed by the `general.name` inside its gguf, which is what
    makes two mirrors of one model dedupe. Some quantisers never set it: the
    Apriel 1.6 decensored build calls itself "base-model", so it sat in the
    library under that name with nothing tying it to the repo. Worse, the name
    is not unique — the next model that self-describes the same way is refused
    as a duplicate of something unrelated."""
    assert hangar.model_slug("base-model", "Apriel-1.6-15b-Thinker-GGUF") == \
        "apriel-1.6-15b-thinker-gguf"
    assert hangar.model_slug("model", "Spicy-Tune-8B") == "spicy-tune-8b"
    assert hangar.model_slug("", "Fallback-Repo") == "fallback-repo"


def test_a_real_gguf_name_is_still_preferred_over_the_repo():
    # the dedupe property must survive: two mirrors of one model share its name
    assert hangar.model_slug("Qwen3.8 27B", "some-mirror-repo") == "qwen3.8-27b"


# --- renaming carries everything else keyed by the slug -----------------------
def test_rename_carries_the_repaired_template(home, tmp_path):
    """A slug is not just a label: the repaired chat template lives at
    templates/<slug>.jinja. Renaming by hand orphaned it, so a model that had
    been given a working template came back on its broken embedded one."""
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    (home / "templates").mkdir(parents=True, exist_ok=True)
    (home / "templates" / "hybrid-tune.jinja").write_text("TEMPLATE", encoding="utf-8")
    hangar.rename_model("hybrid-tune", "something-meaningful")
    assert not (home / "templates" / "hybrid-tune.jinja").exists()
    assert (home / "templates" / "something-meaningful.jinja").read_text(
        encoding="utf-8") == "TEMPLATE"
    assert "something-meaningful" in Registry.load().models
    assert "hybrid-tune" not in Registry.load().models


def test_rename_carries_the_calibration_rows(home):
    """Calibration is keyed "<slug>:<quant>:<backend>". A rename that dropped
    them silently un-tuned a model that had been measured."""
    import json
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    (home / "calibration.json").write_text(json.dumps({
        "hybrid-tune:Q4_K_M:vulkan": {"flags": {"flash_attn": "off"}},
        "other-model:Q4_K_M:vulkan": {"flags": {}},
    }), encoding="utf-8")
    hangar.rename_model("hybrid-tune", "renamed")
    rows = json.loads((home / "calibration.json").read_text(encoding="utf-8"))
    assert "renamed:Q4_K_M:vulkan" in rows
    assert "hybrid-tune:Q4_K_M:vulkan" not in rows
    assert "other-model:Q4_K_M:vulkan" in rows      # untouched


def test_rename_refuses_to_collide(home):
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    existing = next(iter(Registry.load().models))
    with pytest.raises(HangarError, match="already exists"):
        hangar.rename_model("hybrid-tune", existing)


def test_rename_refuses_while_the_model_is_running(home, monkeypatch):
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")
    from rigma import state as st
    monkeypatch.setattr(st, "read_state", lambda: {"model": "hybrid-tune"})
    with pytest.raises(HangarError, match="running"):
        hangar.rename_model("hybrid-tune", "renamed")


# --- explicit re-probe reaches what the automatic heal cannot -----------------
def test_reprobe_offline_says_so_when_nothing_is_downloaded(home):
    """heal_spec is local-file-only on purpose (it runs on every registry
    load). --offline keeps that contract and reports honestly instead of
    silently doing nothing."""
    _stale_spec(home, "never-downloaded.gguf")
    with pytest.raises(HangarError, match="downloaded"):
        hangar.reprobe("hybrid-tune", allow_remote=False)


def test_reprobe_prefers_the_local_file_over_the_network(home, monkeypatch):
    (home / "models").mkdir(parents=True, exist_ok=True)
    _hybrid_gguf(home / "models" / "h.gguf")
    _stale_spec(home, "h.gguf")

    def _boom(*a, **k):
        raise AssertionError("reprobe hit the network with the file on disk")
    monkeypatch.setattr("rigma.hf_browse.remote_inspect", _boom)
    spec = hangar.reprobe("hybrid-tune")
    assert spec.full_attn_layers == 2 and spec.n_layers == 8


def test_reprobe_refuses_registry_models(home):
    with pytest.raises(HangarError, match="hand-authored"):
        hangar.reprobe("definitely-not-a-custom-model")
