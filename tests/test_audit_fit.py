"""VRAM arithmetic that was wrong in the unsafe direction.

Four separate ways the planner handed itself a budget the card does not have,
or refused one it does:

  * a stored calibration replaced a fresh plan's placement without checking
    whether the machine still looks like the machine it was measured on;
  * a sliding-window model's second, window-sized KV cache was budgeted as
    zero — ~400 MiB on a 27B-class SWA model;
  * an integrated GPU's share of system RAM was summed as video RAM;
  * a model whose native window is under CTX_FLOOR could never enter the fit
    loop at all, so it was planned onto the CPU at a context larger than it
    was trained on.
"""
import pytest

from rigma import probe
from rigma.bench import VRAM_DRIFT_TOLERANCE_MB, save_calibration
from rigma.models import (CachePolicy, ComboFlags, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec, RunPlan)
from rigma.registry import Registry
from rigma.resolve import (CACHE_BYTES, CTX_FLOOR, _budgets, fallback_plans,
                           fit_gguf, kv_bytes_per_token, resolve)

MIB = 2 ** 20


def _profile(vram=16368, ram_free=9100):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


# --- F17: a calibration is evidence about one machine state ------------------

_COMBO_KEY = "qwen3.6-35b-a3b:UD-Q3_K_XL:vulkan"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_a_calibration_measured_on_an_idle_desktop_is_not_applied_to_a_busy_one(
        home, monkeypatch):
    """Measured 2026-08-21: 2,828 MiB of desktop drift took the same combo from
    37.59 to 9.95 tok/s, because Windows silently paged a third of the weights.
    n_cpu_moe is exactly the flag that decides whether the weights fit, so
    replaying one measured on an idle card overcommits a busy one."""
    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 1100.0)
    save_calibration(_COMBO_KEY, {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})
    monkeypatch.setattr(probe, "gpu_used_mb",
                        lambda: 1100.0 + VRAM_DRIFT_TOLERANCE_MB + 1)

    plan = resolve(_profile(), Registry.load(), use_case="coding")

    assert plan.flags.n_cpu_moe == 10, "replayed a placement measured elsewhere"
    assert "+calibrated" not in plan.origin
    assert any("calibration override skipped" in ln for ln in plan.explain), \
        plan.explain


def test_a_calibration_from_the_same_machine_state_still_wins(home, monkeypatch):
    """The stopwatch beats the arithmetic when it is a stopwatch reading of the
    machine you are actually on. Nothing about the drift check may weaken that."""
    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 1100.0)
    save_calibration(_COMBO_KEY, {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})

    plan = resolve(_profile(), Registry.load(), use_case="coding")

    assert plan.flags.n_cpu_moe == 8
    assert plan.origin.endswith("+calibrated")


def test_a_placement_measured_at_another_context_is_not_replayed(home, monkeypatch):
    """server_ops._measured_placement already refuses this: "placement measured
    at 32K says nothing about 256K" — the KV cache is most of what moved. The
    resolver applied the same flags with no context check at all."""
    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 1100.0)
    base = resolve(_profile(), Registry.load(), use_case="coding")
    save_calibration(_COMBO_KEY, {"tg_tps": 57.1}, flags={"n_cpu_moe": 8},
                     ctx=base.flags.ctx // 2)

    plan = resolve(_profile(), Registry.load(), use_case="coding")

    assert plan.flags.n_cpu_moe == 10
    assert any("calibration override skipped" in ln for ln in plan.explain), \
        plan.explain


def test_a_placement_measured_on_another_engine_is_not_replayed(home, monkeypatch):
    from rigma import server_ops
    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 1100.0)
    monkeypatch.setattr(server_ops, "engine_version", lambda: "b9867")
    save_calibration(_COMBO_KEY, {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})
    monkeypatch.setattr(server_ops, "engine_version", lambda: "b9999")

    plan = resolve(_profile(), Registry.load(), use_case="coding")

    assert plan.flags.n_cpu_moe == 10
    assert any("engine" in ln for ln in plan.explain), plan.explain


def test_an_unreadable_vram_counter_does_not_retire_the_calibration(
        home, monkeypatch):
    """gpu_used_mb returns None off Windows and on a machine whose counter
    cannot be read. No reading is not a reading of zero."""
    monkeypatch.setattr(probe, "gpu_used_mb", lambda: None)
    save_calibration(_COMBO_KEY, {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})

    plan = resolve(_profile(), Registry.load(), use_case="coding")

    assert plan.flags.n_cpu_moe == 8
    assert plan.origin.endswith("+calibrated")


# --- F19: the second, window-sized KV cache ----------------------------------

class _SwaSpec(ModelSpec):
    """A ModelSpec carrying the windowed geometry `gguf_meta` now reads.

    The three fields belong on ModelSpec itself (and in
    hangar.spec_fields_from_probe, so an import stores them); until they are
    declared there this subclass is what a Gemma-style spec would look like,
    and `swa_kv_bytes` reads them off any spec that has them.
    """
    swa_layers: int = 0
    swa_kv_heads: int = 0
    swa_window: int = 0


def _swa_spec(**over) -> _SwaSpec:
    # 27B-class SWA geometry: 5 of every 6 layers windowed over 1K tokens.
    fields = dict(slug="swa", family="gemma4", kind="dense", n_layers=62,
                  full_attn_layers=10, kv_heads=16, head_dim=128,
                  native_ctx=131072, custom=True,
                  cache_type_policy=CachePolicy(k="f16", v="f16"),
                  ggufs=[GgufFile(repo="a/b", file="swa.gguf",
                                  bytes=8 * 2**30, quant="Q4_K_M")],
                  swa_layers=52, swa_kv_heads=16, swa_window=1024)
    fields.update(over)
    return _SwaSpec(**fields)


def test_the_windowed_layers_cost_what_they_allocate():
    """llama.cpp allocates a second KV cache for the windowed layers, bounded by
    the window rather than by ctx. 52 layers x 16 kv heads x 128 wide x 4 bytes
    x 1024 tokens = 416 MiB that was budgeted as exactly zero."""
    from rigma.resolve import swa_kv_bytes
    spec = _swa_spec()
    want = 52 * 16 * 128 * (CACHE_BYTES["f16"] * 2) * 1024
    assert swa_kv_bytes(spec, "f16", "f16", 32768) == want
    assert round(want / MIB) == 416
    # bounded by the WINDOW, not by ctx: it is a fixed term, proportionally
    # worst at small contexts
    assert swa_kv_bytes(spec, "f16", "f16", 512) == want / 2
    # and it quantises with the cache like any other KV
    assert swa_kv_bytes(spec, "q4_0", "q4_0", 32768) == want * (
        CACHE_BYTES["q4_0"] / CACHE_BYTES["f16"])


def test_a_model_without_sliding_window_attention_pays_nothing():
    """Every other hybrid — Qwen's SSM layers, DeltaNet's linear-attention
    layers — holds a fixed state, not a windowed KV cache. Only a spec that
    carries a window size may be charged for one."""
    from rigma.resolve import swa_kv_bytes
    plain = _swa_spec(swa_layers=0, swa_kv_heads=0, swa_window=0)
    assert swa_kv_bytes(plain, "f16", "f16", 32768) == 0.0
    assert kv_bytes_per_token(plain, "f16", "f16") > 0


def test_the_fit_reserves_the_windowed_cache():
    """The fit is the only thing that decides whether a plan overcommits, so
    the term has to land there and not only in the explainer."""
    spec = _swa_spec()
    blind = spec.model_copy(update={"swa_layers": 0, "swa_window": 0})
    # a card sized so the 416 MiB decides residency: room for the file, the
    # global-layer KV and ~200 MiB, and nothing more
    ctx = 32768
    kv_mb = (ctx * kv_bytes_per_token(spec, "f16", "f16")) / MIB
    prof = _profile(vram=1)
    room = spec.ggufs[0].bytes / MIB + kv_mb + 200
    prof = prof.model_copy(update={
        "gpus": [prof.gpus[0].model_copy(update={"vram_mb": int(
            room + _reserve_mb(prof))})]})
    fits_blind = fit_gguf(blind, blind.ggufs[0], prof, ctx, [])
    honest = fit_gguf(spec, spec.ggufs[0], prof, ctx, [])
    assert fits_blind is not None and fits_blind.ngl == 99
    assert honest is None or honest.ngl < 99 or honest.cache_type_k != "f16", \
        "promised full residency for 416 MiB the driver will page"


def _reserve_mb(profile) -> float:
    """What _budgets holds back, so a test can size a card by what is LEFT."""
    from rigma.resolve import COMPUTE_BUFFER_MB, VRAM_RESERVE_MB
    return VRAM_RESERVE_MB[profile.os] + COMPUTE_BUFFER_MB


def test_the_windowed_geometry_is_read_off_the_file(tmp_path):
    """gguf_meta already identifies the windowed layers to EXCLUDE them from
    the growing cache; it threw the geometry away instead of reporting it."""
    from tests.test_gguf_meta import _kv_arr_bool, _kv_arr_u32, _kv_str, _kv_u32, _write
    kvs = [
        _kv_str(b"general.architecture", b"gemma4"),
        _kv_u32(b"gemma4.block_count", 6),
        _kv_u32(b"gemma4.context_length", 262144),
        _kv_u32(b"gemma4.embedding_length", 2816),
        _kv_u32(b"gemma4.attention.head_count", 16),
        _kv_u32(b"gemma4.attention.key_length", 512),
        _kv_u32(b"gemma4.attention.sliding_window", 1024),
        _kv_arr_u32(b"gemma4.attention.head_count_kv", [8, 8, 8, 8, 8, 2]),
        _kv_arr_bool(b"gemma4.attention.sliding_window_pattern",
                     [True, True, True, True, True, False]),
    ]
    from rigma.gguf_meta import inspect_gguf
    f = inspect_gguf(_write(tmp_path, kvs, name="swa.gguf")).spec_fields
    assert f["full_attn_layers"] == 1          # unchanged
    assert f["swa_layers"] == 5
    assert f["swa_kv_heads"] == 8
    assert f["swa_window"] == 1024


def test_a_file_with_no_sliding_window_reports_none(tmp_path):
    from tests.test_gguf_meta import DENSE, _write
    from rigma.gguf_meta import inspect_gguf
    f = inspect_gguf(_write(tmp_path, DENSE)).spec_fields
    assert f["swa_layers"] == 0 and f["swa_window"] == 0


# --- F20: an iGPU's system RAM is not video RAM ------------------------------

def _raw(name, vram_mb, vendor_id=0x1002, device_type=None):
    d = {"vendor_id": vendor_id, "name": name, "vram_mb": vram_mb}
    if device_type is not None:
        d["device_type"] = device_type
    return d


def test_an_integrated_gpu_is_not_plannable_video_ram():
    """A UMA device reports its largest DEVICE_LOCAL heap, which IS system RAM.
    Summed into the budget it hands the planner 24 GB that does not exist on an
    8 GB card, and every quant verdict, ngl and n_cpu_moe is computed on it."""
    from rigma.probe import VK_DEVICE_TYPE_DISCRETE, VK_DEVICE_TYPE_INTEGRATED, probe_hardware
    p = probe_hardware([], raw_gpus=[
        _raw("Intel(R) UHD Graphics 770", 24576, 0x8086,
             VK_DEVICE_TYPE_INTEGRATED),
        _raw("AMD Radeon RX 9070 XT", 16304, 0x1002, VK_DEVICE_TYPE_DISCRETE)])
    assert [g.name for g in p.gpus] == ["AMD Radeon RX 9070 XT"]
    # and the iGPU cannot take over backend selection either
    assert p.primary_gpu.vram_mb == 16304
    usable, _ = _budgets(p)
    assert usable < 16304, f"budgeted {usable:.0f}MB of a 16GB card"


def test_a_device_that_does_not_say_what_it_is_stays():
    """NVML and every raw dict written before deviceType was recorded carry no
    type. Unknown is not integrated — dropping those would blind the planner to
    real cards."""
    from rigma.probe import probe_hardware
    p = probe_hardware([], raw_gpus=[_raw("AMD Radeon RX 9070 XT", 16304)])
    assert len(p.gpus) == 1


def test_an_igpu_only_machine_keeps_its_igpu():
    """Dropping the only device would take a laptop from slow to nothing."""
    from rigma.probe import VK_DEVICE_TYPE_INTEGRATED, probe_hardware
    p = probe_hardware([], raw_gpus=[
        _raw("Intel(R) Iris Xe", 16384, 0x8086, VK_DEVICE_TYPE_INTEGRATED)])
    assert len(p.gpus) == 1 and p.primary_gpu.vendor == "intel"


# --- F21: a model smaller than the floor ------------------------------------

def _small_ctx_registry(native_ctx=2048, size_gb=1.0):
    spec = ModelSpec(slug="tiny", family="llama", kind="dense", n_layers=22,
                     full_attn_layers=22, kv_heads=4, head_dim=128,
                     native_ctx=native_ctx, use_cases=["general"],
                     cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="r", file="tiny.gguf",
                                     bytes=int(size_gb * 2**30), quant="Q4")])
    return Registry([], {"tiny": spec}, {})


def test_a_2k_model_is_not_exiled_to_the_cpu(home):
    """A Phi-2/TinyLlama-class model, or any gguf whose header omits
    context_length: `while ctx >= CTX_FLOOR` never runs its body, so fit_gguf
    is never called and the plan falls through to the absolute floor — CPU,
    ngl=0 — on a card it fits in four times over."""
    reg = _small_ctx_registry()
    plan = resolve(_profile(), reg, use_case="general")
    assert plan.backend == "vulkan", f"put a 1GB model on the {plan.backend}"
    assert plan.flags.ngl == 99
    assert plan.flags.ctx == 2048


def test_the_floor_plan_never_asks_for_more_context_than_the_model_has(home):
    """When nothing fits, the floor plan hardcoded ctx=CTX_FLOOR — launching a
    2048-token model with -c 8192, a window it was never trained on."""
    reg = _small_ctx_registry(native_ctx=2048, size_gb=64.0)
    tiny_card = _profile(vram=2048, ram_free=512)
    plan = resolve(tiny_card, reg, use_case="general")
    assert plan.flags.ctx == 2048, "asked for more context than the model has"


def test_a_normal_model_still_bottoms_out_at_the_floor(home):
    """The floor is per-model now; for anything trained past CTX_FLOOR it must
    still be CTX_FLOOR."""
    reg = _small_ctx_registry(native_ctx=131072, size_gb=64.0)
    plan = resolve(_profile(vram=2048, ram_free=512), reg, use_case="general")
    assert plan.flags.ctx == CTX_FLOOR


def test_fallback_plans_clamp_the_floor_too(home):
    """The launch-failed path builds its own floor plan, with the same
    hardcoded CTX_FLOOR — a fallback that cannot start is not a fallback."""
    tiny = _small_ctx_registry(native_ctx=2048, size_gb=1.0).models["tiny"]
    big = ModelSpec(slug="big", family="llama", kind="dense", n_layers=48,
                    full_attn_layers=48, kv_heads=8, head_dim=128,
                    native_ctx=131072, use_cases=["general"],
                    cache_type_policy=CachePolicy(),
                    ggufs=[GgufFile(repo="r", file="big-q6.gguf",
                                    bytes=8 * 2**30, quant="Q6_K"),
                           GgufFile(repo="r", file="big-q4.gguf",
                                    bytes=4 * 2**30, quant="Q4_K_M")])
    reg = Registry([], {"big": big, "tiny": tiny}, {})
    plan = RunPlan(model_slug="big", gguf=big.ggufs[0], backend="vulkan",
                   flags=ComboFlags(ctx=32768), origin="calculator")

    outs = fallback_plans(plan, reg, _profile())

    floor = [p for p in outs if p.origin == "fallback:floor"]
    assert floor and floor[0].model_slug == "tiny"
    assert floor[0].flags.ctx == 2048, "floor plan asked for 8192 on a 2K model"


def test_the_models_page_never_offers_more_context_than_the_model_has(home):
    """The two screens disagreed about one model: the resolver refused a 2048
    model outright while quant_verdicts probed 8192 first and reported it as
    fitting there — a window it was never trained on."""
    from rigma.resolve import quant_verdicts
    spec = _small_ctx_registry(native_ctx=2048).models["tiny"]
    got = quant_verdicts(spec, _profile())
    assert got and got[0]["ok"]
    assert got[0]["ctx"] == 2048, got[0]


def test_a_normal_model_keeps_the_whole_ladder(home):
    """Every shipped model is trained past CTX_FLOOR; nothing about the cap may
    change what they report."""
    from rigma.resolve import quant_verdicts
    spec = _small_ctx_registry(native_ctx=131072).models["tiny"]
    got = quant_verdicts(spec, _profile())
    assert got and got[0]["ok"] and got[0]["ctx"] >= 8192
