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
    without = fit_gguf(mk(None), mk(None).ggufs[0], p, 4096, [])
    assert without is not None and without.ngl == 99      # fully fits
    # the 900MB projector must still cost GPU room: it no longer OOMs (dense
    # partial offload runs it) but it forces layers off the GPU
    mm = GgufFile(repo="r", file="mm.gguf", bytes=900 * 2**20, quant="F16")
    withmm = fit_gguf(mk(mm), mk(mm).ggufs[0], p, 4096, [])
    assert withmm is not None and withmm.ngl < 8          # projector budgeted


def _box(vram_mb=16384, ram_free=28000):
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    return HardwareProfile(
        os="windows", ram_mb=32768, ram_free_mb=ram_free,
        cpu=CpuInfo(cores=8, name="amd"), disk_free_gb=500,
        gpus=[GpuInfo(name="RX 9070 XT", vendor="amd", vram_mb=vram_mb,
                      backends=["vulkan"])])


def _dense(size_gb, layers=48, kv_heads=8, head_dim=128, ctx=262400):
    from rigma.models import GgufFile, ModelSpec
    return ModelSpec(
        slug="t", family="llama", kind="dense", n_layers=layers,
        full_attn_layers=layers, kv_heads=kv_heads, head_dim=head_dim,
        native_ctx=ctx, license="x", use_cases=["general"],
        ggufs=[GgufFile(repo="local", file="t.gguf",
                        bytes=int(size_gb * 2**30), quant="Q6_K")])


def test_q8_cache_is_tried_before_giving_up_context():
    # 48L x 8kv x 128hd = 192 KiB/token at f16. An 11GB model on 16GB leaves
    # ~3GB, so 16k@f16 (3072MB) misses by ~50MB and used to fall back to 8k.
    # q8_0 halves the cache, which is far cheaper than losing half the window.
    from rigma import resolve
    spec = _dense(11.0)
    flags = resolve.fit_gguf(spec, spec.ggufs[0], _box(), 16384, [])
    assert flags is not None, "16k should fit once q8_0 is considered"
    assert flags.cache_type_k == "q8_0" and flags.cache_type_v == "q8_0"
    assert flags.ngl == 99, "weights must stay fully on the GPU"


def test_f16_is_still_preferred_when_it_fits():
    # quantising the cache is a trade, not a default — a small model that fits
    # at f16 must keep f16
    from rigma import resolve
    spec = _dense(4.0)
    flags = resolve.fit_gguf(spec, spec.ggufs[0], _box(), 8192, [])
    assert flags.cache_type_k == "f16"


def test_quantised_cache_beats_spilling_weights_to_ram():
    # the ordering bug: f16-with-partial-offload was chosen over
    # q8_0-fully-resident. Offloading weights costs real tokens/sec; the cache
    # quantisation costs ~8.5 effective bits.
    # 11.5GB @16k: f16 needs 14848MB (over the 14284 budget) but q8_0 needs
    # 13408MB, so ONLY the cache choice decides whether weights stay resident.
    from rigma import resolve
    spec = _dense(11.5)
    flags = resolve.fit_gguf(spec, spec.ggufs[0], _box(), 16384, [])
    assert flags is not None
    assert flags.ngl == 99, "must not spill layers while q8_0 would fit"
    assert flags.cache_type_k == "q8_0"
    # NOTE: when NEITHER cache type fits fully, f16-with-offload is still
    # returned before q8_0-with-offload. Picking the variant that keeps more
    # layers resident would be better, but that case doesn't arise for the
    # models in play here, so it's left alone rather than guessed at.
