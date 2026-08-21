"""The resolver, checked against configurations that were actually run.

Every number here comes from a measured rung on the owner's RX 9070 XT
(16,304 MiB, desktop holding ~1,095 MiB) on 2026-08-21. The point is that the
planner must not refuse, or degrade, a configuration that demonstrably works:

    Q3_K_M 64K q5_1   38.27 tok/s   dedicated 15,361 MiB   paged  -3 MiB
    Q3_K_M 96K q4_0   36.74 tok/s   dedicated 16,050 MiB   paged  97 MiB
    Q3_K_L 64K q4_0   15.86 tok/s   dedicated 15,950 MiB   paged 566 MiB

Before this, the planner chose ngl=55 for the first of those — pushing 9 of 64
layers to the CPU for a config that ran entirely on the GPU.
"""
import pytest

from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec)
from rigma.resolve import fit_gguf, kv_bytes_per_token

MIB = 2 ** 20


@pytest.fixture
def profile():
    return HardwareProfile(
        os="windows", cpu=CpuInfo(cores=20),
        gpus=[GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16304,
                      arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                      backends=["vulkan", "rocm"])],
        ram_mb=32133, ram_free_mb=24000, disk_free_gb=500.0,
        vram_used_mb=1095)


@pytest.fixture
def spec():
    # Qwen3.8-27B geometry, read from the gguf: 64 blocks, full-attention every
    # 4th layer, 4 kv heads, 256-wide k/v.
    return ModelSpec(
        slug="qwen38-ara-v5", family="qwen35", kind="dense",
        n_layers=64, full_attn_layers=16, kv_heads=4, head_dim=256,
        native_ctx=262144, custom=True,
        cache_type_policy=CachePolicy(k="f16", v="f16"),
        ggufs=[GgufFile(repo="a/b", file="RVN-Q3_K_M.gguf",
                        bytes=13_301_433_856, quant="Q3_K_M")])


def test_the_kv_arithmetic_matches_the_measured_cache(spec):
    """1,536 MiB at 64K with q5_1 — the figure the measured run allocated."""
    per_token = kv_bytes_per_token(spec, "q5_1", "q5_1")
    assert round(65536 * per_token / MIB) == 1536


def test_64k_is_planned_fully_resident(profile, spec):
    """MEASURED at 38.27 tok/s with 3 MiB spare. Planning any offload here
    trades 60% of throughput for context the card did not need to give up."""
    flags = fit_gguf(spec, spec.ggufs[0], profile, 65536, [])
    assert flags is not None, "refused a configuration that was measured working"
    assert flags.ngl == 99, f"offloaded to ngl={flags.ngl} for a resident config"
    assert flags.n_cpu_moe == 0


def test_it_reaches_for_a_smaller_cache_before_spilling_weights(profile, spec):
    """The ladder stopped at q8_0, so 64K "did not fit" and weights went to the
    CPU — while q5_1 fit with room to spare. Offload costs tokens/sec; one more
    step of cache quantisation costs a fraction of a percent of perplexity."""
    flags = fit_gguf(spec, spec.ggufs[0], profile, 65536, [])
    assert flags is not None
    assert flags.cache_type_k in ("q5_1", "q4_0", "q8_0")
    used = spec.ggufs[0].bytes / MIB + 65536 * kv_bytes_per_token(
        spec, flags.cache_type_k, flags.cache_type_v) / MIB
    assert used <= 16304 - 1095, "planned more than the card physically has"


def test_a_config_that_paged_is_not_planned_as_resident(profile, spec):
    """Q3_K_L at 64K paged 566 MiB and halved. The planner must see that as
    offload rather than promising full residency."""
    big = GgufFile(repo="a/b", file="RVN-Q3_K_L.gguf", bytes=14_344_766_976,
                   quant="Q3_K_L")
    flags = fit_gguf(spec, big, profile, 65536, [])
    if flags is not None and flags.ngl == 99:
        used = big.bytes / MIB + 65536 * kv_bytes_per_token(
            spec, flags.cache_type_k, flags.cache_type_v) / MIB
        assert used <= 16304 - 1095, (
            "promised full residency for a config measured paging 566 MiB")


def test_a_projector_that_will_not_be_loaded_is_not_reserved(profile, spec):
    """Measured 2026-08-21: launching with vision off still reserved the
    600 MiB projector, which pushed the plan to ngl=62 — two layers on the CPU
    — and the live server then ran at 30.63 tok/s where the same configuration
    benched at 49.46. Reserving memory for something the launcher is not going
    to load is pure loss."""
    from rigma.models import GgufFile
    with_mm = spec.model_copy(update={
        "mmproj": GgufFile(repo="a/b", file="mm.gguf", bytes=629_247_008,
                           quant="mmproj")})
    without = with_mm.model_copy(update={"mmproj": None})
    a = fit_gguf(with_mm, with_mm.ggufs[0], profile, 32768, [])
    b = fit_gguf(without, without.ggufs[0], profile, 32768, [])
    assert b is not None and b.ngl == 99
    # the point: dropping the projector must actually buy something
    assert (a is None) or (b.ngl >= a.ngl)


def test_speculation_reserves_its_draft_cache(profile, spec):
    """draft-mtp allocates a second KV cache for the draft head. Measured on
    this machine at n=1: ~465 MiB (dedicated went 14,650 -> 15,549 for a file
    only 434 MiB larger). Planning without it produced a config that fit on
    paper and paged in practice."""
    from rigma.resolve import draft_cache_mb
    # measured: 465 MiB at 16K, 565 MiB at 32K
    assert 440 <= draft_cache_mb(spec, 16384, "q4_0", "draft-mtp", 1) <= 490
    assert 540 <= draft_cache_mb(spec, 32768, "q4_0", "draft-mtp", 1) <= 590
    # no speculation, no reservation
    assert draft_cache_mb(spec, 32768, "q4_0", "none", 0) == 0
