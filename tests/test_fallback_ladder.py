from pathlib import Path

from rigma.models import CpuInfo, GpuInfo, HardwareProfile
from rigma.registry import Registry
from rigma.resolve import fallback_plans, resolve


def _profile():
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)


def test_ladder_descends_quants_then_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = Registry.load()
    p = _profile()
    plan = resolve(p, reg, use_case="coding")  # UD-Q3_K_XL combo
    ladder = fallback_plans(plan, reg, p)
    quants = [(s.model_slug, s.gguf.quant) for s in ladder]
    assert quants[0] == ("qwen3.6-35b-a3b", "UD-Q2_K_XL")  # next smaller that fits
    assert quants[-1] == ("qwen3-0.6b", "Q8_0")            # absolute floor
    assert ladder[-1].backend == "cpu" and ladder[-1].flags.ngl == 0
    assert (plan.model_slug, plan.gguf.quant) not in quants


def test_up_walks_ladder_on_launch_failure(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import rigma.cli as cli

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    def _fake_probe(gpu_table, raw_gpus=None):
        return _profile()   # canonical 16GB box - real RAM must never leak in

    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    # hermetic against a live rigma on this machine: skip the port preflight
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    from rigma import runtime
    monkeypatch.setattr(runtime, "ensure_engine",
                        lambda backend, os_name: Path("llama-server.exe"))
    monkeypatch.setattr(runtime, "ensure_model",
                        lambda gguf, **kw: Path(gguf.file))
    attempts = []

    def flaky_launch(exe, plan, model_path, port=11500, timeout=300.0,
                     extra_args=None):
        attempts.append(plan.gguf.quant)
        raise RuntimeError("boom: failed to become healthy")

    monkeypatch.setattr(runtime, "launch_server", flaky_launch)
    res = CliRunner().invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                       "--use-case", "coding", "--yes"])
    assert res.exit_code == 1
    assert attempts[0] == "UD-Q3_K_XL" and attempts[-1] == "Q8_0"
    assert len(attempts) >= 3
    assert "falling back" in res.output
