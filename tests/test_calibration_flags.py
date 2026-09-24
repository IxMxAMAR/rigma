"""06-4: a stored calibration is merged through validation, not model_copy.

`model_copy(update=...)` skips every ComboFlags validator, so a hand-edited
calibration.json replayed `-fa sideways` and `--cache-type-k q3_k_bogus` onto a
real launch line and bypassed `_symmetric_kv`. The merge now goes through
`ComboFlags.model_validate`; an entry that fails is ignored whole, with a note
in `explain`, and the fresh plan is kept.
"""
from rigma.bench import save_calibration
from rigma.models import CpuInfo, GpuInfo, HardwareProfile
from rigma.registry import Registry
from rigma.resolve import resolve

# The exact combo resolve() picks for this profile + use_case.
KEY = "qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan"


def _profile():
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _resolve_with(tmp_path, monkeypatch, flags):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration(KEY, {"tg_tps": 57.1}, flags=flags)
    return resolve(_profile(), Registry.load(), use_case="coding")


def test_a_valid_calibration_is_still_applied(tmp_path, monkeypatch):
    plan = _resolve_with(tmp_path, monkeypatch, {"n_cpu_moe": 8})
    assert plan.flags.n_cpu_moe == 8          # calibrated, not the combo's 10
    assert plan.origin.endswith("+calibrated")


def test_an_out_of_range_flag_never_reaches_the_launch_line(tmp_path,
                                                           monkeypatch):
    plan = _resolve_with(tmp_path, monkeypatch,
                         {"flash_attn": "sideways",
                          "cache_type_k": "q3_k_bogus",
                          "cache_type_v": "q4_0"})
    assert plan.flags.flash_attn == "on"
    assert plan.flags.cache_type_k == "q8_0"  # the combo's own, untouched
    assert "+calibrated" not in plan.origin
    assert any("calibration override ignored" in ln for ln in plan.explain)
    args = plan.server_args("m.gguf", 11500)
    assert "sideways" not in args and "q3_k_bogus" not in args


def test_one_bad_field_does_not_half_apply_the_rest(tmp_path, monkeypatch):
    """All-or-nothing: n_cpu_moe is a real placement, and applying it from an
    entry that is otherwise nonsense would leave a config nobody measured."""
    plan = _resolve_with(tmp_path, monkeypatch,
                         {"n_cpu_moe": 8, "flash_attn": "sideways"})
    assert plan.flags.n_cpu_moe == 10          # the combo's own, not the entry's
    assert "+calibrated" not in plan.origin


def test_a_non_object_flags_value_is_ignored_not_fatal(tmp_path, monkeypatch):
    plan = _resolve_with(tmp_path, monkeypatch, ["nope"])
    assert "+calibrated" not in plan.origin
    assert any("calibration override ignored" in ln for ln in plan.explain)
