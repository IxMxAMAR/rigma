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


# --- C11-read: the merge refuses what the sweep refuses to crown -------------

def test_a_calibration_row_cannot_apply_a_quality_degrading_env_lever(
        tmp_path, monkeypatch):
    """C11 closed the WRITE path — a sweep no longer crowns
    `LLAMA_ATTN_ROT_DISABLE` — but a `calibration.json` written before that fix
    (or hand-edited) was still merged here and reached the child's environment
    on EVERY later launch. The read path must refuse the same lever, say so,
    and still apply the rest of the row."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan", {"tg_tps": 57.1},
                     flags={"n_cpu_moe": 8,
                            "env": {"LLAMA_ATTN_ROT_DISABLE": "1"}})
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert "LLAMA_ATTN_ROT_DISABLE" not in plan.flags.env
    assert plan.flags.env == {}               # the combo's own env, untouched
    assert plan.flags.n_cpu_moe == 8          # the rest of the row still applies
    assert any("dropped quality-degrading env lever" in e
               for e in plan.explain), plan.explain
    assert any("LLAMA_ATTN_ROT_DISABLE" in e for e in plan.explain)


def test_a_normal_calibration_env_key_still_applies(tmp_path, monkeypatch):
    """The gate must drop only the quality lever, not every env key."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan", {"tg_tps": 57.1},
                     flags={"env": {"GGML_VK_DISABLE_COOPMAT": "1"}})
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.flags.env == {"GGML_VK_DISABLE_COOPMAT": "1"}


def test_the_merge_gate_reuses_the_sweeps_one_list(monkeypatch):
    """No second list: whatever `bench._QUALITY_ENV_LEVERS` calls
    quality-degrading is exactly what the merge drops."""
    from rigma import bench
    from rigma.resolve import _without_quality_env_levers
    monkeypatch.setattr(bench, "_QUALITY_ENV_LEVERS",
                        ("LLAMA_ATTN_ROT_DISABLE", "RIGMA_FAKE_QUALITY_LEVER"))
    cleaned, dropped = _without_quality_env_levers(
        {"env": {"RIGMA_FAKE_QUALITY_LEVER": "1", "KEEP_ME": "1"}})
    assert dropped == ["RIGMA_FAKE_QUALITY_LEVER"]
    assert cleaned == {"env": {"KEEP_ME": "1"}}


def test_a_lowercase_calibration_lever_is_dropped_at_merge(tmp_path,
                                                           monkeypatch):
    """The case-insensitive match must hold on the READ path too.

    llama.cpp's `getenv` ignores case on Windows, so a hand-edited lowercase
    key in `calibration.json` would reach the child's environment on every
    later launch. The merge gate must drop it and still apply the rest of the
    row."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan", {"tg_tps": 57.1},
                     flags={"n_cpu_moe": 8,
                            "env": {"llama_attn_rot_disable": "1"}})
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.flags.env == {}
    assert plan.flags.n_cpu_moe == 8
    assert any("dropped quality-degrading env lever" in e
               for e in plan.explain), plan.explain


def test_a_non_object_calibration_flags_row_is_ignored_not_fatal(
        tmp_path, monkeypatch):
    """GUIDANCE 7: the read path treats calibration.json as untrusted, and a
    hand-edited row whose `flags` is a list must keep the existing "flags is not
    an object" behaviour. The C11-read gate must not raise AttributeError before
    the merge's TypeError handler can report it."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan", {"tg_tps": 57.1},
                     flags=["not", "an", "object"])
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.flags.n_cpu_moe == 10          # the combo's own placement
    assert any("flags is not an object" in e for e in plan.explain), plan.explain
