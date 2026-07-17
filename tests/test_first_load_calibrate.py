"""First load of a new model auto-tunes once, then launches the winner."""
import os

import pytest

from rigma import bench, server_ops
from rigma import state as st
from rigma.models import CachePolicy, CpuInfo, GgufFile, GpuInfo, \
    HardwareProfile, ModelSpec
from rigma.registry import Registry


def _profile():
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16368, arch="rdna4",
                  slug="amd-radeon-rx-9070-xt-16g", backends=["vulkan"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                           cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)


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
    # a DIFFERENT model is currently running, so switching to "m" is a fresh load
    st.write_state("cur", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=32768)
    launches = []

    class _SP:
        proc = type("P", (), {"pid": 4242})()

        def stop(self):
            pass

    monkeypatch.setattr("rigma.runtime.ensure_engine", lambda b, o: "srv.exe")

    def fake_launch(exe, rp, path, port, timeout=None, extra_args=None):
        launches.append({"env": dict(rp.flags.env), "fa": rp.flags.flash_attn})
        return _SP()

    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    monkeypatch.setattr("rigma.bench.launch_server", fake_launch)
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    monkeypatch.setattr(server_ops, "_await_port_free", lambda *a, **k: None)
    return reg, launches


def test_first_load_calibrates_then_launches_winner(env, monkeypatch):
    reg, launches = env
    # quick set = baseline, fa-off, coopmat-off; make coopmat-off fastest
    tgs = iter([40.0, 40.0, 90.0])
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=next(tgs), prompt_tokens=8, gen_tokens=8))

    assert bench.is_calibrated("m", "Q4", "vulkan") is False
    server_ops.perform_switch("m", reg, _profile())

    # 3 trial launches + 1 final launch of the winner
    assert len(launches) == 4
    assert launches[-1]["env"].get("GGML_VK_DISABLE_COOPMAT") == "1"
    # marker cleared, calibration recorded so it never re-runs
    assert server_ops.read_calib_marker() is None
    assert bench.is_calibrated("m", "Q4", "vulkan") is True


def test_second_load_skips_calibration(env, monkeypatch):
    reg, launches = env
    bench.save_calibration("m:Q4:vulkan", {"tg_tps": 90}, flags={}, calibrated=True)
    monkeypatch.setattr(bench, "run_bench", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("must not bench")))
    server_ops.perform_switch("m", reg, _profile())
    assert len(launches) == 1  # straight to launch, no trials


def test_disabled_via_env_skips_calibration(env, monkeypatch):
    reg, launches = env
    monkeypatch.setenv("RIGMA_AUTO_CALIBRATE", "0")
    monkeypatch.setattr(bench, "run_bench", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("must not bench")))
    server_ops.perform_switch("m", reg, _profile())
    assert len(launches) == 1
    assert bench.is_calibrated("m", "Q4", "vulkan") is False


def test_status_surfaces_calibrating_marker(tmp_path, monkeypatch):
    """Mid-switch (engine dead, tuning) both status endpoints report progress."""
    from fastapi.testclient import TestClient

    from rigma.serve import build_app
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    server_ops._write_calib_marker("newmodel", "coopmat-off")
    # no live engine recorded -> endpoints fall back to the calibrating marker
    client = TestClient(build_app(upstream_port=1))
    for path in ("/api/server", "/api/status"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.json()["calibrating"]["model"] == "newmodel"
        assert r.json()["calibrating"]["step"] == "coopmat-off"
    server_ops._clear_calib_marker()
    # marker gone -> normal not-running 404
    assert client.get("/api/server").status_code == 404
