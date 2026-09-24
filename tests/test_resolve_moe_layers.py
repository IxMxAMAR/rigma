"""06-1: a MoE header with no `block_count` must not kill the whole fit path.

`gguf_meta` reports `n_layers = 0` for a header that omits `block_count` while
still reporting `kind="moe"` from `expert_count`. The dense branch of
`_fit_with_cache` was hardened against exactly that; the MoE branch divided by
`spec.n_layers` before any guard, so `rigma up` printed a traceback and the
Models page swallowed the exception and rendered an empty fit dict.
"""
import io
import struct

from rigma.gguf_meta import inspect_gguf
from rigma.hangar import moe_from_probe, spec_fields_from_probe
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec, MoESpec)
from rigma.registry import Registry
from rigma.resolve import fit_gguf, resolve

T_U32, T_STR = 4, 8


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _kv_u32(key: bytes, val: int) -> bytes:
    return _s(key) + struct.pack("<I", T_U32) + struct.pack("<I", val)


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _s(key) + struct.pack("<I", T_STR) + _s(val)


def _gguf(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _moe_header_without_block_count() -> bytes:
    """A self-converted MoE gguf whose header never declared block_count."""
    return _gguf([
        _kv_str(b"general.architecture", b"moe"),
        _kv_str(b"general.name", b"No Block Count"),
        _kv_u32(b"moe.context_length", 8192),
        _kv_u32(b"moe.embedding_length", 1024),
        _kv_u32(b"moe.attention.head_count", 16),
        _kv_u32(b"moe.attention.head_count_kv", 2),
        _kv_u32(b"moe.attention.key_length", 64),
        _kv_u32(b"moe.expert_count", 8),
        _kv_u32(b"moe.expert_used_count", 2),
    ])


def _profile(vram=16368, ram_free=9100):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _probed_spec() -> ModelSpec:
    """The real probe -> hangar -> ModelSpec pipeline on the synthetic header."""
    info = inspect_gguf(io.BytesIO(_moe_header_without_block_count()))
    f = info.spec_fields
    return ModelSpec(
        slug="noblock", family="moe", kind=f["kind"],
        moe=moe_from_probe(f, 2 ** 33),
        ggufs=[GgufFile(repo="r", file="noblock.gguf", bytes=2 ** 33,
                        quant="Q4_K_M")],
        use_cases=["general"], cache_type_policy=CachePolicy(),
        **spec_fields_from_probe(f))


def test_probe_floors_a_header_that_omits_block_count():
    """Belt and braces: the hangar-side floor keeps a 0 out of every spec that
    the raw probe would otherwise hand to the fit math."""
    info = inspect_gguf(io.BytesIO(_moe_header_without_block_count()))
    assert info.spec_fields["n_layers"] == 0        # the header really is silent
    assert spec_fields_from_probe(info.spec_fields)["n_layers"] == 1


def test_moe_fit_returns_none_instead_of_dividing_by_zero():
    """The guard itself, on a spec that reaches it with n_layers = 0."""
    spec = ModelSpec(
        slug="zero", family="moe", kind="moe", n_layers=0, full_attn_layers=0,
        kv_heads=2, head_dim=64, native_ctx=8192,
        moe=MoESpec(total_b=16.0, active_b=3.0, expert_weight_fraction=0.85),
        ggufs=[GgufFile(repo="r", file="zero.gguf", bytes=2 ** 33,
                        quant="Q4_K_M")],
        use_cases=["general"], cache_type_policy=CachePolicy())
    # no exception: the model is simply unplaceable without a layer count
    assert fit_gguf(spec, spec.ggufs[0], _profile(), 8192, []) is None


def test_resolve_plans_a_moe_header_that_omits_block_count():
    """End to end: the same header through the real pipeline must yield a plan,
    not a traceback."""
    plan = resolve(_profile(), Registry([], {"noblock": _probed_spec()}, {}),
                   use_case="general")
    assert plan is not None and plan.model_slug == "noblock"
