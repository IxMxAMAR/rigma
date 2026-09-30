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


def test_a_lower_ram_tier_combo_is_used_and_named(tmp_path, monkeypatch):
    """AUDIT F15-4: the reference box measures 31.4 GB (tier 32) and the verified
    9070 XT combo is filed at ram-16. find_combo must fall back to it — and say
    in `explain` which tier it came from."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    p = HardwareProfile(gpus=[gpu], ram_mb=32143, ram_free_mb=20000,
                        cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)
    plan = resolve(p, Registry.load(), use_case="coding")
    assert plan.origin == \
        "combo:amd/amd-radeon-rx-9070-xt-16g/ram-16/coding.json"
    assert plan.flags.n_cpu_moe == 10
    assert any("ram-16 tier" in e for e in plan.explain)


def test_calculator_kicks_in_for_unknown_gpu(tmp_path, monkeypatch):
    # RIGMA_HOME isolated: the resolver now prefers ON-DISK quants, so with
    # the real home this test would see the owner's models and pick a
    # calibrated on-disk quant instead of the pure calculator answer
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
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


def test_a_smaller_cache_is_tried_before_giving_up_context():
    # 48L x 8kv x 128hd = 192 KiB/token at f16, so 16k costs 3,072MB of cache.
    # Sized so f16 misses the budget by ~60MB: the point is that the resolver
    # spends cache precision rather than half the context window.
    #
    # Was _dense(11.0) against a 900MB compute-buffer reserve. That reserve was
    # measured at 43-81MB on 2026-08-21 and cut to 400, which handed this
    # scenario enough room for f16 to simply fit — a better outcome, but no
    # longer a test of the ladder. Resized to keep exercising it.
    from rigma import resolve
    box = _box()
    # Size the model relative to the LIVE budget rather than hardcoding GB.
    # Pinned sizes had to be re-tuned every time a reserve constant moved
    # (900 -> 400 -> 150 as they were replaced by measurements), and each
    # re-tune silently stopped exercising the ladder for a while.
    usable, _ = resolve._budgets(box)
    kv_f16 = 16384 * resolve.kv_bytes_per_token(_dense(1.0), "f16", "f16") / 2**30
    spec = _dense(round(usable / 1024 - kv_f16 + 0.06, 2))   # f16 misses by ~60MB
    flags = resolve.fit_gguf(spec, spec.ggufs[0], box, 16384, [])
    assert flags is not None, "16k should fit once the cache can be quantised"
    assert flags.cache_type_k != "f16", "gave up context instead of cache bits"
    assert flags.cache_type_k == flags.cache_type_v
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
    box = _box()
    # Size the model relative to the LIVE budget rather than hardcoding GB.
    # Pinned sizes had to be re-tuned every time a reserve constant moved
    # (900 -> 400 -> 150 as they were replaced by measurements), and each
    # re-tune silently stopped exercising the ladder for a while.
    usable, _ = resolve._budgets(box)
    kv_f16 = 16384 * resolve.kv_bytes_per_token(_dense(1.0), "f16", "f16") / 2**30
    spec = _dense(round(usable / 1024 - kv_f16 + 0.06, 2))   # f16 misses by ~60MB
    flags = resolve.fit_gguf(spec, spec.ggufs[0], box, 16384, [])
    assert flags is not None
    assert flags.ngl == 99, "must not spill layers while q8_0 would fit"
    assert flags.cache_type_k == "q8_0"
    # NOTE: when NEITHER cache type fits fully, f16-with-offload is still
    # returned before q8_0-with-offload. Picking the variant that keeps more
    # layers resident would be better, but that case doesn't arise for the
    # models in play here, so it's left alone rather than guessed at.


def test_resolver_prefers_on_disk_quants(tmp_path, monkeypatch):
    # live 2026-07-20: `up -y` silently began a 26GB download while a 14.4GB
    # quant of the same model sat on disk. Any on-disk quant that fits beats
    # any remote one; remote is only considered when nothing local fits.
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                              HardwareProfile, ModelSpec)
    from rigma.registry import Registry
    from rigma.resolve import resolve
    big = GgufFile(repo="r", quant="Q5_K_P", file="m-q5.gguf",
                   bytes=26 * 2**30)
    small = GgufFile(repo="r", quant="IQ3_M", file="m-iq3.gguf",
                     bytes=14 * 2**30)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "m-iq3.gguf").write_text("x")   # only IQ3 on disk
    spec = ModelSpec(slug="m", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=32768, ggufs=[big, small],
                     use_cases=["general"], capabilities=["tools"],
                     cache_type_policy=CachePolicy())
    reg = Registry([], {"m": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=16000, backends=["vulkan"])
    prof = HardwareProfile(gpus=[gpu], ram_mb=32000, ram_free_mb=16000,
                           cpu=CpuInfo(cores=8), os="windows",
                           disk_free_gb=100.0)
    plan = resolve(prof, reg, model_override="m")
    assert plan.gguf.quant == "IQ3_M",         f"picked {plan.gguf.quant} — locality must beat quality"


# --- F23/F24: the curated-combo path had no fit check and could crash ---------

COMBO_REL = "amd/amd-radeon-rx-9070-xt-16g/ram-16/coding.json"


def _broken_combo_registry(**update):
    """The shipped registry with the ram-16 coding combo replaced."""
    from rigma.models import Combo
    r = Registry.load()
    base = r.combos[COMBO_REL]
    combos = dict(r.combos)
    combos[COMBO_REL] = Combo.model_validate(
        {**base.model_dump(), **update}) if update else base
    return Registry(r.gpus, r.models, combos, r.use_cases)


def _pressured(used_mb, ram_free=20000, ram_mb=16234):
    """The reference card with `used_mb` of its VRAM held by the desktop."""
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=ram_mb, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0, vram_used_mb=used_mb)


def test_a_combo_naming_an_unknown_model_raises_resolve_error():
    """`spec = registry.models[combo.model]` was an unguarded subscript, so a
    registry snapshot that dropped a model killed up/plan/sweep with a raw
    KeyError where every other failure raises ResolveError — and cli.py catches
    only ResolveError (AUDIT F24)."""
    import pytest

    from rigma.resolve import ResolveError
    broken = _broken_combo_registry(model="ghost-model")
    with pytest.raises(ResolveError) as e:
        resolve(_profile(ram_free=20000), broken, use_case="coding")
    msg = str(e.value)
    assert "ghost-model" in msg and "rigma update" in msg


def test_a_combo_naming_an_unknown_quant_raises_resolve_error():
    """The bare `next(...)` raised StopIteration, which is not even an
    Exception subclass a caller could have caught (AUDIT F24)."""
    import pytest

    from rigma.resolve import ResolveError
    broken = _broken_combo_registry(quant="NO_SUCH_QUANT")
    with pytest.raises(ResolveError) as e:
        resolve(_profile(ram_free=20000), broken, use_case="coding")
    msg = str(e.value)
    assert "NO_SUCH_QUANT" in msg and "does not list" in msg


def test_a_combo_verified_for_more_headroom_than_this_box_has_is_rejected():
    """The combo path returned its stored flags with no fit check at all — the
    combo's own declared `budget` was read nowhere in the codebase — so on a
    pressured desktop `rigma sweep` benchmarked an over-budget config and could
    crown a winner picked by paging noise (AUDIT F23).

    The ram-16 coding combo declares `budget.vram_mb = 15200`; a desktop
    holding 8.7GB of the card leaves 7.4GB usable."""
    r = Registry.load()
    assert r.combos[COMBO_REL].budget is not None, "fixture drifted"
    plan = resolve(_pressured(used_mb=8780), r, use_case="coding")
    assert plan.origin == "calculator", plan.explain
    assert plan.explain[0].startswith("registry"), plan.explain
    assert "NOT used" in plan.explain[0] and "15200" in plan.explain[0], \
        plan.explain[0]


def test_a_combo_the_machine_still_has_the_headroom_for_is_kept():
    r = Registry.load()
    plan = resolve(_pressured(used_mb=0), r, use_case="coding")
    assert plan.origin.startswith("combo:"), plan.explain


def _combo_with_env(env):
    """The shipped ram-16 coding combo with `env` added to its stored flags."""
    base = Registry.load().combos[COMBO_REL]
    return _broken_combo_registry(flags={**base.flags.model_dump(), "env": env})


def test_a_combo_env_quality_lever_is_dropped_with_a_visible_note():
    """w13b2 NIT 2: `resolve()` copied `combo.flags` onto the plan with no
    `_without_quality_env_levers` pass, so a hand-edited registry combo carrying
    `LLAMA_ATTN_ROT_DISABLE` (or a lowercase variant) reached a plain launch —
    the silent quality regression C11 closed for the calibration merge. The lever
    is dropped and the drop is named in `explain`; an unrelated env var rides
    along untouched."""
    for spelling in ("LLAMA_ATTN_ROT_DISABLE", "llama_attn_rot_disable"):
        plan = resolve(_pressured(used_mb=0),
                       _combo_with_env({spelling: "1",
                                        "RIGMA_UNRELATED": "keep"}),
                       use_case="coding")
        assert plan.origin.startswith("combo:"), (spelling, plan.explain)
        assert plan.flags.env == {"RIGMA_UNRELATED": "keep"}, \
            (spelling, plan.flags.env)
        assert any("LLAMA_ATTN_ROT_DISABLE" in e and "dropped" in e
                   for e in plan.explain), (spelling, plan.explain)


def test_a_combo_env_without_a_quality_lever_is_untouched():
    """The gate drops ONLY a `bench._QUALITY_ENV_LEVERS` name: a combo whose env
    carries an unrelated engine variable is copied through exactly as authored,
    and nothing is claimed to have been dropped."""
    plan = resolve(_pressured(used_mb=0),
                   _combo_with_env({"RIGMA_UNRELATED": "keep"}),
                   use_case="coding")
    assert plan.origin.startswith("combo:"), plan.explain
    assert plan.flags.env == {"RIGMA_UNRELATED": "keep"}, plan.flags.env
    assert not any("dropped" in e for e in plan.explain), plan.explain


def test_a_combo_whose_placement_no_longer_exists_falls_through():
    """Even with no declared budget the placement has to still EXIST: a combo
    verified at 32K is not usable on a box that can no longer place it at
    all (AUDIT F23)."""
    broken = _broken_combo_registry(budget=None)
    plan = resolve(_profile(vram=1024, ram_free=1000), broken,
                   use_case="coding")
    assert plan.origin == "calculator", plan.explain
    assert any("can no longer be placed" in line for line in plan.explain), \
        plan.explain
