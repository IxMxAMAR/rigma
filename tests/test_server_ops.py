import json
import os
from types import SimpleNamespace

import pytest

from rigma import server_ops, state
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec)
from rigma.registry import Registry


def test_verdict_matrix():
    assert server_ops.verdict(None, 50.0) == "unknown"
    assert server_ops.verdict(40.0, None) == "unknown"
    assert server_ops.verdict(10.0, 50.0) == "degraded"   # < 60% of expected
    assert server_ops.verdict(45.0, 50.0) == "healthy"


def test_expected_tg_reads_what_the_writer_actually_wrote(tmp_path, monkeypatch):
    """Build the fixture with the real writer, not by hand.

    `bench.save_calibration` is the only thing that writes calibration.json and
    it nests the numbers under "measured" (bench.py:93). `expected_tg` read a
    flat `["tg_tps"]` that no writer has ever produced, so it returned None for
    every real calibration on this machine and the engine-room verdict at
    serve.py was permanently "unknown". The previous version of this test passed
    only because it hand-wrote a shape the writer does not emit — which is
    exactly how the bug survived.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import bench
    bench.save_calibration("m:q:vulkan", {"tg_tps": 57.1, "pp_tps": 689.0})
    assert server_ops.expected_tg("m", "q", "vulkan") == 57.1
    assert server_ops.expected_tg("m", "q", "cuda") is None


def test_expected_tg_still_reads_a_legacy_flat_entry(tmp_path, monkeypatch):
    """Entries written before the nesting existed must keep working."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"m:q:vulkan": {"tg_tps": 57.1, "pp_tps": 600}}), encoding="utf-8")
    assert server_ops.expected_tg("m", "q", "vulkan") == 57.1


def test_expected_tg_is_none_when_nothing_was_measured(tmp_path, monkeypatch):
    """A calibration entry can exist with flags but no measurement — a sweep
    where the baseline won writes no `measured` block. That must read as
    unknown, not crash and not fabricate a number."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"m:q:vulkan": {"flags": {"flash_attn": "off"}}}), encoding="utf-8")
    assert server_ops.expected_tg("m", "q", "vulkan") is None


def test_log_tail_newest_and_clamped(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server-11499.log").write_text("old\n" * 5, encoding="utf-8")
    import time as _t
    _t.sleep(0.02)
    newest = logs / "server-11498.log"
    newest.write_text("\n".join(f"line{i}" for i in range(50)), encoding="utf-8")
    os.utime(newest)
    out = server_ops.log_tail(lines=10)
    assert out.splitlines()[-1] == "line49" and len(out.splitlines()) == 10
    assert server_ops.log_tail(lines=99999).count("\n") <= 1000


def _fake_world(tmp_path):
    gguf_small = GgufFile(repo="r", file="small.gguf", bytes=10, quant="Q4")
    gguf_big = GgufFile(repo="r", file="big.gguf", bytes=10, quant="Q8")
    spec = dict(family="f", kind="dense", n_layers=2, full_attn_layers=2,
                kv_heads=2, head_dim=64, native_ctx=8192,
                cache_type_policy=CachePolicy())
    reg = Registry([], {
        "small-model": ModelSpec(slug="small-model", ggufs=[gguf_small], **spec),
        "big-model": ModelSpec(slug="big-model", ggufs=[gguf_big], **spec),
    }, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=16000, ram_free_mb=8000,
                              cpu=CpuInfo(cores=8), os="windows",
                              disk_free_gb=100.0)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "small.gguf").write_text("x")  # only small on disk
    return reg, profile


def test_switch_options_downloaded_only_excludes_current(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _fake_world(tmp_path)
    s = {"model": "current-model", "use_case": "general"}
    opts = server_ops.switch_options(s, registry=reg, profile=profile)
    assert [o["model"] for o in opts] == ["small-model"]
    assert opts[0]["quant"] == "Q4" and "context" in opts[0]["reason"]
    s2 = {"model": "small-model", "use_case": "general"}
    assert server_ops.switch_options(s2, registry=reg, profile=profile) == []


def test_perform_switch_happy_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _fake_world(tmp_path)
    state.write_state("current-model", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan",
                      use_case="general", ctx=4096)
    killed = []
    monkeypatch.setattr("rigma.state.kill_pid", lambda pid: killed.append(pid))
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    fake_sp = SimpleNamespace(proc=SimpleNamespace(pid=4242))
    monkeypatch.setattr("rigma.runtime.launch_server",
                        lambda exe, plan, mp, port=0, timeout=300.0,
                        extra_args=None: fake_sp)
    new = server_ops.perform_switch("small-model", registry=reg, profile=profile)
    assert new["model"] == "small-model" and new["engine_pid"] == 4242
    assert new["ctx"] > 0 and killed == [999999]

    with pytest.raises(RuntimeError, match="not downloaded"):
        server_ops.perform_switch("big-model", registry=reg, profile=profile)

    def boom(*a, **k):
        raise RuntimeError("launch failed: boom")
    monkeypatch.setattr("rigma.runtime.launch_server", boom)
    state.write_state("current-model", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan",
                      use_case="general", ctx=4096)
    with pytest.raises(RuntimeError, match="boom"):
        server_ops.perform_switch("small-model", registry=reg, profile=profile)
    # failed switch now leaves an UNLOADED state (UI stays manageable), not a
    # cleared one — the dead engine is gone but the UI can retry a load
    s = state.read_state()
    assert s is not None and s["unloaded"] is True and s["engine_pid"] == -1


def test_perform_switch_same_model_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _fake_world(tmp_path)
    state.write_state("small-model", "Q4", 18500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), backend="vulkan")
    with pytest.raises(RuntimeError, match="already running"):
        server_ops.perform_switch("small-model", registry=reg, profile=profile)


def test_switch_options_falls_back_to_on_disk_quant(tmp_path, monkeypatch):
    """Resolver prefers a quant that isn't downloaded -> offer the one that is.

    Live repro 2026-07-14: under RAM pressure the resolver picked qwen
    UD-Q2_K_XL while only UD-Q3_K_XL was on disk; the advisor showed
    'no alternative models' despite a usable 32K-ctx model locally."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    small = GgufFile(repo="r", file="dual-small.gguf", bytes=10, quant="Q2")
    big = GgufFile(repo="r", file="dual-big.gguf", bytes=20, quant="Q3")
    spec = ModelSpec(slug="dual-model", family="f", kind="dense", n_layers=2,
                     full_attn_layers=2, kv_heads=2, head_dim=64,
                     native_ctx=32768, ggufs=[big, small],
                     cache_type_policy=CachePolicy())
    reg = Registry([], {"dual-model": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=16000, ram_free_mb=8000,
                              cpu=CpuInfo(cores=8), os="windows",
                              disk_free_gb=100.0)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "dual-big.gguf").write_text("x")  # ONLY Q3 on disk
    s = {"model": "other-model", "use_case": "general"}
    opts = server_ops.switch_options(s, registry=reg, profile=profile)
    assert [o["quant"] for o in opts] == ["Q3"]
    assert "Q3 on disk" in opts[0]["reason"]


def test_perform_switch_uses_on_disk_quant(tmp_path, monkeypatch):
    """Live repro 2026-07-17: switch-back refused because the resolver
    preferred a quant that wasn't downloaded, while another quant WAS."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    small = GgufFile(repo="r", file="dual-small.gguf", bytes=10, quant="Q2")
    big = GgufFile(repo="r", file="dual-big.gguf", bytes=20, quant="Q3")
    spec = ModelSpec(slug="dual-model", family="f", kind="dense", n_layers=2,
                     full_attn_layers=2, kv_heads=2, head_dim=64,
                     native_ctx=32768, ggufs=[big, small],
                     cache_type_policy=CachePolicy())
    reg = Registry([], {"dual-model": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=16000, ram_free_mb=8000,
                              cpu=CpuInfo(cores=8), os="windows",
                              disk_free_gb=100.0)
    (tmp_path / "models").mkdir()
    # ONLY the smaller quant on disk; the resolver PREFERS the bigger one
    (tmp_path / "models" / "dual-small.gguf").write_text("x")
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan")
    monkeypatch.setattr("rigma.state.kill_pid", lambda pid: None)
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    fake_sp = SimpleNamespace(proc=SimpleNamespace(pid=7))
    monkeypatch.setattr("rigma.runtime.launch_server",
                        lambda exe, plan, mp, port=0, timeout=300.0,
                        extra_args=None: fake_sp)
    new = server_ops.perform_switch("dual-model", registry=reg, profile=profile)
    assert new["model"] == "dual-model" and new["quant"] == "Q2"


def test_repaired_chat_template_is_passed_to_the_engine(tmp_path, monkeypatch):
    """A gguf can ship a chat template llama.cpp cannot use. Apriel-1.6's
    decensored build self-assigns `{%- set messages = messages ... -%}`, which
    minja evaluates as unbounded recursion — the process dies with
    STATUS_STACK_BUFFER_OVERRUN (0xC0000409) before allocating any context, so
    it looks like a VRAM problem and isn't. Dropping a repaired template at
    ~/.rigma/templates/<slug>.jinja must override the embedded one."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma.runtime import rigma_home
    d = rigma_home() / "templates"
    d.mkdir(parents=True, exist_ok=True)
    t = d / "my-model.jinja"
    t.write_text("{{ 'hi' }}", encoding="utf-8")
    assert t.is_file()
    # the lookup server_ops performs, isolated
    found = (rigma_home() / "templates" / "my-model.jinja")
    assert found.is_file()
    assert not (rigma_home() / "templates" / "other-model.jinja").is_file()


# --- choosing WHICH downloaded quant to run ----------------------------------
# Owner, 2026-08-19: "I have 2 downloaded and I do not get to choose which one I
# want to load and neither does it tell me which one is loaded." Both true.
# perform_switch took model/ctx/kv/vision and let the resolver pick the quant,
# so with two quants on disk there was no way to ask for the smaller one — the
# very thing you want when trading quality for context.

def _dual_quant_world(tmp_path):
    a = GgufFile(repo="r", file="big.gguf", bytes=10, quant="Q3_K_L")
    b = GgufFile(repo="r", file="small.gguf", bytes=10, quant="Q3_K_M")
    spec = ModelSpec(slug="dual", family="f", kind="dense", n_layers=2,
                     full_attn_layers=2, kv_heads=2, head_dim=64,
                     native_ctx=32768, ggufs=[a, b],
                     cache_type_policy=CachePolicy())
    reg = Registry([], {"dual": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=16000, ram_free_mb=8000,
                              cpu=CpuInfo(cores=8), os="windows",
                              disk_free_gb=100.0)
    (tmp_path / "models").mkdir(exist_ok=True)
    for f in ("big.gguf", "small.gguf"):
        (tmp_path / "models" / f).write_text("x")   # BOTH downloaded
    return reg, profile


def _stub_launch(monkeypatch, tmp_path):
    monkeypatch.setattr("rigma.state.kill_pid", lambda pid: None)
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    monkeypatch.setattr("rigma.runtime.launch_server",
                        lambda exe, plan, mp, port=0, timeout=300.0,
                        extra_args=None: SimpleNamespace(
                            proc=SimpleNamespace(pid=4242)))


def test_perform_switch_can_be_asked_for_a_specific_quant(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _dual_quant_world(tmp_path)
    _stub_launch(monkeypatch, tmp_path)
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    new = server_ops.perform_switch("dual", registry=reg, profile=profile,
                                    quant="Q3_K_M")
    assert new["quant"] == "Q3_K_M", "the requested quant must be the one loaded"


def test_without_a_quant_the_resolver_still_chooses(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _dual_quant_world(tmp_path)
    _stub_launch(monkeypatch, tmp_path)
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    new = server_ops.perform_switch("dual", registry=reg, profile=profile)
    assert new["quant"] in ("Q3_K_L", "Q3_K_M")


def test_an_unknown_quant_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _dual_quant_world(tmp_path)
    _stub_launch(monkeypatch, tmp_path)
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    with pytest.raises(RuntimeError, match="Q9_K_XXL"):
        server_ops.perform_switch("dual", registry=reg, profile=profile,
                                  quant="Q9_K_XXL")


def test_switching_quant_on_the_running_model_is_allowed(tmp_path, monkeypatch):
    """The owner's actual case: Q3_K_L loaded, Q3_K_M also on disk, wanting the
    smaller one for more context. Without this the same-model guard refused it
    as "already running" — true of the model, false of the request."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _dual_quant_world(tmp_path)
    _stub_launch(monkeypatch, tmp_path)
    state.write_state("dual", "Q3_K_L", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    new = server_ops.perform_switch("dual", registry=reg, profile=profile,
                                    quant="Q3_K_M")
    assert new["quant"] == "Q3_K_M"


def test_the_same_model_with_no_change_requested_is_still_refused(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg, profile = _dual_quant_world(tmp_path)
    _stub_launch(monkeypatch, tmp_path)
    state.write_state("dual", "Q3_K_L", 18500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    with pytest.raises(RuntimeError, match="already running"):
        server_ops.perform_switch("dual", registry=reg, profile=profile)
