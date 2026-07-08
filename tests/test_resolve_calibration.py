from rigma.bench import save_calibration
from rigma.models import CpuInfo, GpuInfo, HardwareProfile
from rigma.registry import Registry
from rigma.resolve import resolve


def _profile():
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)


def test_calibration_flags_override_combo(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan",
                     {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.flags.n_cpu_moe == 8  # calibrated, not the combo's 10
    assert plan.origin.endswith("+calibrated")


def test_no_calibration_no_change(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.flags.n_cpu_moe == 10 and "+calibrated" not in plan.origin
