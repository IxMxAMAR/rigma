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


def test_expected_tg_reads_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"m:q:vulkan": {"tg_tps": 57.1, "pp_tps": 600}}), encoding="utf-8")
    assert server_ops.expected_tg("m", "q", "vulkan") == 57.1
    assert server_ops.expected_tg("m", "q", "cuda") is None


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
    assert state.read_state() is None  # failed switch clears stale state


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
