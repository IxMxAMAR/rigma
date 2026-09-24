"""06-7: draft_cache_mb must not claim a KV-precision dependence it lacks.

It took a `kv` argument and never read it. The two constants are a measured
total (365 MiB fixed + 6.25 KiB/token, obtained by differencing), dominated by
the head's own nextn blocks and compute buffers, so scaling them by cache
precision would under-reserve for a quantised cache. The parameter is deleted
rather than implemented.
"""
import inspect

from rigma.models import CachePolicy, GgufFile, ModelSpec
from rigma.resolve import draft_cache_mb, with_launch_overheads


def _spec() -> ModelSpec:
    return ModelSpec(slug="s", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=32768,
                     ggufs=[GgufFile(repo="r", file="a.gguf", bytes=1,
                                     quant="Q4_K_M")],
                     cache_type_policy=CachePolicy())


def test_neither_function_takes_a_cache_type():
    assert "kv" not in inspect.signature(draft_cache_mb).parameters
    assert "kv" not in inspect.signature(with_launch_overheads).parameters


def test_the_measured_points_still_hold():
    spec = _spec()
    assert 440 <= draft_cache_mb(spec, 16384, "draft-mtp", 1) <= 490
    assert 540 <= draft_cache_mb(spec, 32768, "draft-mtp", 1) <= 590
    # no speculation, no reservation
    assert draft_cache_mb(spec, 32768, "none", 0) == 0


def test_the_overhead_slot_carries_the_draft_cache():
    spec = _spec()
    fit = with_launch_overheads(spec, vision=False, ctx=32768,
                                spec_type="draft-mtp", n_max=1)
    assert fit.mmproj is not None
    assert fit.mmproj.bytes == int(
        draft_cache_mb(spec, 32768, "draft-mtp", 1) * 2 ** 20)
