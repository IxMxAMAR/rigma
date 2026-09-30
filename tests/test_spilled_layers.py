"""`_spilled` must count the output layer the way `_cpu_layers` does.

`-ngl 58` on a 64-layer model leaves layers 0-6 on the CPU — seven — because ngl
counts the OUTPUT layer (llama-model.cpp:
`i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`, the output placed by the
same rule). `_cpu_layers` already said 7; `_spilled` said 6/64 and under-reported
the weight fraction by one layer, which is enough to flip the 0.15 speed bucket
near a boundary.
"""
import pytest

from rigma.models import CachePolicy, ComboFlags, GgufFile, MoESpec, ModelSpec
from rigma.resolve import _cpu_layers, _spilled


def _dense(n=64) -> ModelSpec:
    return ModelSpec(slug="d", family="qwen35", kind="dense", n_layers=n,
                     full_attn_layers=n, kv_heads=4, head_dim=256,
                     native_ctx=262144, cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="r", file="d.gguf", bytes=1,
                                     quant="Q4")])


def _moe(n=64) -> ModelSpec:
    spec = _dense(n)
    spec.moe = MoESpec(total_b=35.0, active_b=3.0, expert_weight_fraction=0.85)
    return spec


def test_spilled_counts_the_output_layer_cpu_layers_already_counts():
    spec = _dense(64)
    flags = ComboFlags(ctx=8192, ngl=58)
    assert _cpu_layers(spec, flags) == 7
    assert _spilled(spec, flags) == pytest.approx(7 / 64)
    assert _spilled(spec, flags) == pytest.approx(0.109, abs=0.001)


def test_spilled_is_clamped_at_one_and_zero():
    spec = _dense(64)
    # every layer on the GPU once ngl covers the output layer too
    assert _spilled(spec, ComboFlags(ctx=8192, ngl=65)) == 0.0
    assert _spilled(spec, ComboFlags(ctx=8192, ngl=99)) == 0.0
    # nothing on the GPU: 65/64 unclamped, so the fraction is exactly 1
    assert _spilled(spec, ComboFlags(ctx=8192, ngl=0)) == 1.0


def test_the_moe_branch_is_unchanged():
    spec = _moe(64)
    flags = ComboFlags(ctx=8192, ngl=99, n_cpu_moe=13)
    assert _spilled(spec, flags) == pytest.approx(13 / 64 * 0.85)
    # n_cpu_moe counts expert-offloaded layers; no output-layer off-by-one
    assert _spilled(spec, ComboFlags(ctx=8192, ngl=99, n_cpu_moe=0)) == 0.0
