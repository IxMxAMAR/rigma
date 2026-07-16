"""perform_switch ctx override: real fit math, same-model relaunch allowed."""
import os

import pytest

from rigma import server_ops
from rigma import state as st
from rigma.models import CachePolicy, CpuInfo, GgufFile, GpuInfo, \
    HardwareProfile, ModelSpec
from rigma.registry import Registry


def _profile():
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = ModelSpec(slug="m", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=131072,
                     ggufs=[GgufFile(repo="r", file="m.gguf", bytes=2**30,
                                     quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy())
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "m.gguf").write_bytes(b"x")
    reg = Registry([], {"m": spec}, {})
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=32768)
    launched = {}

    class _SP:
        def __init__(self):
            self.proc = type("P", (), {"pid": 4242})()
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, osn: "llama-server.exe")
    def fake_launch(exe, rp, path, port, extra_args=None):
        launched.update(ctx=rp.flags.ctx, n_cpu_moe=rp.flags.n_cpu_moe)
        return _SP()
    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    return reg, launched


def test_same_model_ctx_relaunch(env):
    reg, launched = env
    out = server_ops.perform_switch("m", reg, _profile(), ctx=65536)
    assert launched["ctx"] == 65536 and out["ctx"] == 65536


def test_ctx_clamped_to_native(env):
    reg, launched = env
    server_ops.perform_switch("m", reg, _profile(), ctx=99999999)
    assert launched["ctx"] == 131072            # native cap


def test_impossible_ctx_is_clean_error_and_engine_survives(env, monkeypatch):
    reg, launched = env
    # shrink the card so the request genuinely can't fit
    gpu = GpuInfo(vendor="amd", name="tiny", vram_mb=1400, arch="rdna4",
                  slug="tiny-1g", backends=["vulkan"])
    prof = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=2000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)
    with pytest.raises(RuntimeError, match="doesn't fit"):
        server_ops.perform_switch("m", reg, prof, ctx=131072)
    assert not launched                        # never killed + relaunched
    assert st.read_state() is not None         # state untouched


def test_same_model_without_ctx_still_refused(env):
    reg, _ = env
    with pytest.raises(RuntimeError, match="already running"):
        server_ops.perform_switch("m", reg, _profile())


def test_ctx_relaunch_carries_dense_ngl(monkeypatch, tmp_path):
    """Review CRITICAL: the ctx path must copy the recomputed ngl, or a dense
    model relaunched at a bigger ctx keeps a stale ngl=99 and OOMs."""
    import os

    from rigma import server_ops
    from rigma import state as st
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    from rigma.registry import Registry
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    # a big dense model that only partial-offloads at high ctx
    spec = ModelSpec(slug="big", family="f", kind="dense", n_layers=40,
                     full_attn_layers=40, kv_heads=8, head_dim=128,
                     native_ctx=131072,
                     ggufs=[GgufFile(repo="r", file="big.gguf",
                                     bytes=int(22 * 2**30), quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy())
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "big.gguf").write_bytes(b"x")
    reg = Registry([], {"big": spec}, {})
    st.write_state("big", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=8192)
    launched = {}

    class _SP:
        proc = type("P", (), {"pid": 4242})()
    monkeypatch.setattr("rigma.runtime.ensure_engine", lambda b, o: "x.exe")
    monkeypatch.setattr("rigma.runtime.launch_server",
                        lambda e, rp, p, port, extra_args=None:
                        (launched.update(ngl=rp.flags.ngl, ctx=rp.flags.ctx)
                         or _SP()))
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    monkeypatch.setattr(server_ops, "_await_port_free", lambda *a, **k: None)
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16368, arch="rdna4",
                  slug="amd-radeon-rx-9070-xt-16g", backends=["vulkan"])
    prof = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)
    server_ops.perform_switch("big", reg, prof, ctx=16384)
    assert 0 < launched["ngl"] < 40      # partial offload carried through
