"""06-5: the context-reduction loop must try the floor rung.

`ctx //= 2` halved PAST `_ctx_floor` whenever native_ctx was not a power-of-two
multiple of two above it (12288, 10000, 24576, ...). The floor is the one rung
`_ctx_floor` exists to guarantee, and the Models page probes exactly 8192/4096/
2048 — so the page said "fits at 8192" while `rigma up` never asked.
"""
from rigma.models import (CachePolicy, ComboFlags, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec, RunPlan)
from rigma.registry import Registry
from rigma import resolve as resolve_mod
from rigma.resolve import fallback_plans, resolve

MIB = 2 ** 20


def _profile(vram=16368, ram_free=9100):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _spec(native_ctx=12288):
    return ModelSpec(
        slug="ctxfloor", family="f", kind="dense", n_layers=8,
        full_attn_layers=8, kv_heads=2, head_dim=64, native_ctx=native_ctx,
        ggufs=[GgufFile(repo="r", file="big.gguf", bytes=2 * 2 ** 30,
                        quant="Q4_K_M"),
               GgufFile(repo="r", file="small.gguf", bytes=2 ** 30,
                        quant="Q3_K_M")],
        use_cases=["general"], cache_type_policy=CachePolicy())


def _spy(seen):
    """fit_gguf stand-in: places the model at 8192 and nowhere above it."""
    def fit(spec, gguf, profile, ctx, explain):
        seen.append(ctx)
        return ComboFlags(ctx=ctx) if ctx <= 8192 else None
    return fit


def test_the_calculator_tries_the_floor_rung(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(resolve_mod, "fit_gguf", _spy(seen))
    plan = resolve(_profile(), Registry([], {"ctxfloor": _spec()}, {}),
                   use_case="general")
    assert seen == [12288, 8192], f"rungs tried: {seen}"
    assert plan is not None and plan.flags.ctx == 8192
    assert plan.backend == "vulkan"      # the GPU plan, not the CPU floor


def test_fallback_plans_tries_the_floor_rung(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(resolve_mod, "fit_gguf", _spy(seen))
    spec = _spec()
    plan = RunPlan(model_slug="ctxfloor", gguf=spec.ggufs[0],
                   backend="vulkan", flags=ComboFlags(ctx=12288),
                   origin="calculator")
    out = fallback_plans(plan, Registry([], {"ctxfloor": spec}, {}), _profile())
    assert seen == [12288, 8192], f"rungs tried: {seen}"
    fallbacks = [p for p in out if p.origin == "fallback"]
    assert [p.flags.ctx for p in fallbacks] == [8192]


def test_the_step_never_goes_past_the_floor():
    from rigma.resolve import _next_ctx_rung
    assert _next_ctx_rung(8192, 8192) == 4096      # below floor -> loop ends
    assert _next_ctx_rung(12288, 8192) == 8192     # steps TO the floor
    assert _next_ctx_rung(32768, 8192) == 16384    # ordinary halving untouched
    assert _next_ctx_rung(10000, 8192) == 8192
    assert _next_ctx_rung(2048, 8192) == 1024      # already under the floor


def test_a_model_whose_window_is_the_floor_is_tried_once(monkeypatch):
    """Regression guard: the floor rung must not be tried twice."""
    seen: list[int] = []
    monkeypatch.setattr(resolve_mod, "fit_gguf", _spy(seen))
    plan = resolve(_profile(),
                   Registry([], {"ctxfloor": _spec(native_ctx=8192)}, {}),
                   use_case="general")
    assert seen == [8192]
    assert plan is not None and plan.flags.ctx == 8192
