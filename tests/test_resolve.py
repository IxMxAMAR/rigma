import math

from rigma.models import CpuInfo, GpuInfo, HardwareProfile
from rigma.registry import Registry
from rigma.resolve import CACHE_BYTES, kv_bytes_per_token, resolve


def _profile(vram=16368, ram_free=9100, slug="amd-radeon-rx-9070-xt-16g"):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug=slug, backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)


def test_kv_math_qwen36():
    r = Registry.load()
    spec = r.models["qwen3.6-35b-a3b"]
    # 2 * 10 full-attn layers * 2 kv heads * 256 dim = 10240 elements/token
    assert kv_bytes_per_token(spec, "f16", "f16") == 10240 * CACHE_BYTES["f16"]
    q8 = kv_bytes_per_token(spec, "q8_0", "q8_0")
    assert math.isclose(q8, 10240 * 1.0625)


def test_exact_combo_wins():
    plan = resolve(_profile(), Registry.load(), use_case="coding")
    assert plan.origin.startswith("combo:")
    assert plan.flags.n_cpu_moe == 10 and plan.flags.cache_type_k == "q8_0"
    assert plan.gguf.quant == "UD-Q3_K_XL" and plan.backend == "vulkan"


def test_calculator_kicks_in_for_unknown_gpu():
    p = _profile(vram=20480, slug="future-card-20g")  # no combo, no class file
    plan = resolve(p, Registry.load(), use_case="coding")
    assert plan.origin == "calculator"
    assert plan.flags.n_cpu_moe >= 0 and plan.explain  # math shown
    # 20GB card: bigger quant should fit than on 16GB
    assert plan.gguf.quant in ("UD-Q4_K_XL", "UD-Q3_K_XL")


def test_floor_never_fails():
    p = _profile(vram=2048, ram_free=2500, slug="tiny-2g")
    plan = resolve(p, Registry.load(), use_case="coding")
    assert plan.model_slug == "qwen3-0.6b"


def test_calculator_grows_ctx_to_fit(monkeypatch):
    """Owner finding 2026-07-16: ctx was capped at CTX_DEFAULT even when far
    more KV fit in VRAM. Calculator plans must grow toward native_ctx."""
    from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                              HardwareProfile, ModelSpec)
    from rigma.registry import Registry
    from rigma.resolve import resolve
    # tiny model: 1GB file, tiny kv/token -> plenty of room to grow
    spec = ModelSpec(slug="roomy", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=131072,
                     ggufs=[GgufFile(repo="r", file="roomy.gguf",
                                     bytes=2**30, quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy())
    reg = Registry([], {"roomy": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32000, ram_free_mb=16000,
                        cpu=CpuInfo(cores=8), os="windows", disk_free_gb=100.0)
    plan = resolve(p, reg, use_case="general")
    assert plan.flags.ctx > 16384          # grew past the old default cap
    assert plan.flags.ctx <= 131072        # never past native
    assert any("grow" in ln for ln in plan.explain)


def test_calculator_growth_respects_native_ctx(monkeypatch):
    from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                              HardwareProfile, ModelSpec)
    from rigma.registry import Registry
    from rigma.resolve import resolve
    spec = ModelSpec(slug="short", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=8192,
                     ggufs=[GgufFile(repo="r", file="s.gguf",
                                     bytes=2**30, quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy())
    reg = Registry([], {"short": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32000, ram_free_mb=16000,
                        cpu=CpuInfo(cores=8), os="windows", disk_free_gb=100.0)
    assert resolve(p, reg, use_case="general").flags.ctx == 8192


def test_coding_prefers_tools_capable_models():
    from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                              HardwareProfile, ModelSpec)
    from rigma.registry import Registry
    from rigma.resolve import resolve
    mk = lambda slug, caps, size: ModelSpec(  # noqa: E731
        slug=slug, family="f", kind="dense", n_layers=8, full_attn_layers=8,
        kv_heads=2, head_dim=64, native_ctx=32768, capabilities=caps,
        ggufs=[GgufFile(repo="r", file=f"{slug}.gguf", bytes=size, quant="Q4")],
        use_cases=["coding"], cache_type_policy=CachePolicy())
    reg = Registry([], {"big-plain": mk("big-plain", [], 8 * 2**30),
                        "small-tools": mk("small-tools", ["tools"], 2 * 2**30)},
                   {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32000, ram_free_mb=16000,
                        cpu=CpuInfo(cores=8), os="windows", disk_free_gb=100.0)
    plan = resolve(p, reg, use_case="coding")
    assert plan.model_slug == "small-tools"      # tools capability outranks size
    assert any("tools" in ln for ln in plan.explain)
    # non-coding use case: size still wins
    reg2 = Registry([], {k: v.model_copy(update={"use_cases": ["general"]})
                         for k, v in reg.models.items()}, {})
    assert resolve(p, reg2, use_case="general").model_slug == "big-plain"


def test_size_ranking_uses_gguf_bytes_not_layers():
    """Critical regression 2026-07-16: 8B vision model (36 layers) outranked
    the 35B MoE (total_b 35.0) because dense size proxy was n_layers."""
    from importlib import resources
    from rigma.probe import probe_hardware
    from rigma.registry import Registry
    from rigma.resolve import resolve
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    from pathlib import Path
    bundled = Path(str(resources.files("rigma").joinpath("data/registry")))
    reg = Registry.load(bundled)  # hermetic: user cache may be stale
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16368, arch="rdna4",
                  slug="none", backends=["vulkan"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                        cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)
    assert probe_hardware  # imported for parity; profile built manually
    plan = resolve(p, reg, use_case="general")
    assert plan.model_slug == "qwen3.6-35b-a3b"   # flagship outranks 8B VL


def test_unknown_model_override_clean_error():
    from importlib import resources
    from pathlib import Path
    from rigma.registry import Registry
    from rigma.resolve import ResolveError, resolve
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    bundled = Path(str(resources.files("rigma").joinpath("data/registry")))
    reg = Registry.load(bundled)
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16368, backends=["vulkan"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                        cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)
    import pytest as _p
    with _p.raises(ResolveError, match="unknown model"):
        resolve(p, reg, model_override="not-a-model")


def test_fit_budgets_mmproj_vram():
    """Native-vision finding 2026-07-17: Qwen3.6-35B ships a ~900MB mmproj in
    the same repo. launch passes --mmproj but fit_gguf never budgeted it —
    at grow-to-fit-maxed ctx that unbudgeted projector is an OOM."""
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    from rigma.resolve import _budgets, fit_gguf, kv_bytes_per_token
    p = _profile()
    usable_vram, _ = _budgets(p)
    file_bytes = 0  # placeholder until sized below (lambda reads it lazily)
    mk = lambda mm: ModelSpec(  # noqa: E731
        slug="v", family="f", kind="dense", n_layers=8, full_attn_layers=8,
        kv_heads=2, head_dim=64, native_ctx=32768,
        ggufs=[GgufFile(repo="r", file="v.gguf", bytes=file_bytes, quant="Q4")],
        mmproj=mm, use_cases=["general"], cache_type_policy=CachePolicy())
    kv_mb = 4096 * kv_bytes_per_token(mk(None), "f16", "f16") / 2**20
    # file sized to leave only ~200MB headroom at ctx 4096
    file_bytes = int((usable_vram - kv_mb - 200) * 2**20)
    assert fit_gguf(mk(None), mk(None).ggufs[0], p, 4096, []) is not None
    mm = GgufFile(repo="r", file="mm.gguf", bytes=900 * 2**20, quant="F16")
    assert fit_gguf(mk(mm), mk(mm).ggufs[0], p, 4096, []) is None
