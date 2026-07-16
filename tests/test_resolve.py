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
