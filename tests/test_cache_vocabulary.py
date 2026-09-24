"""06-2/06-3: ONE cache-type vocabulary, shared and complete.

`resolve.CACHE_BYTES` omitted bf16/q5_0/q4_1, which `quant_quality` treats as
real cache formats, so a policy pinning one of them raised KeyError out of the
whole fit path. Meanwhile `ComboFlags._symmetric_kv` ranked every unlisted type
0, so a more precise type lost to q4_0 and the pair was silently rewritten to
the least precise one. Both modules now read one table, and anything not in it
is rejected where the spec is parsed.
"""
import pytest
from pydantic import ValidationError

from rigma.models import CachePolicy, ComboFlags, CpuInfo, GgufFile, GpuInfo, \
    HardwareProfile, ModelSpec
from rigma.quant_quality import _KV_PPL
from rigma.registry import Registry
from rigma.resolve import CACHE_BYTES, fit_gguf, quant_verdicts, resolve


def _profile(vram=16368, ram_free=9100):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _spec(policy: CachePolicy) -> ModelSpec:
    return ModelSpec(slug="cachey", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=32768,
                     ggufs=[GgufFile(repo="r", file="cachey.gguf",
                                     bytes=2 ** 30, quant="Q4_K_M")],
                     use_cases=["general"], cache_type_policy=policy)


# --- the table itself --------------------------------------------------------

def test_cache_bytes_are_the_real_ggml_block_sizes():
    assert CACHE_BYTES["bf16"] == 2.0
    assert CACHE_BYTES["q5_0"] == 22 / 32
    assert CACHE_BYTES["q4_1"] == 20 / 32


def test_the_shared_table_covers_every_type_the_loss_table_knows():
    """The two vocabularies must not drift apart again: quant_quality publishes
    a reference loss for each of these, so each must be sizeable."""
    assert set(_KV_PPL) <= set(CACHE_BYTES)


def test_resolve_sizes_with_the_one_shared_table():
    import rigma.models as models
    import rigma.resolve as resolve_mod
    assert resolve_mod.CACHE_BYTES is models.CACHE_BYTES


# --- three policies x three entry points ------------------------------------

@pytest.mark.parametrize("kv", ["bf16", "q5_0", "q4_1"])
def test_a_policy_pinning_a_real_cache_type_works_at_every_entry_point(kv):
    spec = _spec(CachePolicy(k=kv, v=kv))
    prof = _profile()

    flags = fit_gguf(spec, spec.ggufs[0], prof, 8192, [])
    assert flags is not None and flags.cache_type_k == kv

    verdicts = quant_verdicts(spec, prof)
    assert verdicts and verdicts[0]["ok"] and verdicts[0]["kv"] == kv

    plan = resolve(prof, Registry([], {"cachey": spec}, {}), use_case="general")
    assert plan is not None and plan.flags.cache_type_k == kv


# --- five asymmetric pairs through ComboFlags directly -----------------------

@pytest.mark.parametrize("k,v,want", [
    ("q5_1", "q4_0", "q5_1"),   # q5_1 is more precise than q4_0
    ("bf16", "q4_0", "bf16"),
    ("q5_0", "q4_0", "q5_0"),
    ("q4_1", "q4_0", "q4_1"),
    ("q5_1", "q8_0", "q8_0"),   # q8_0 still wins when it really is more precise
])
def test_an_asymmetric_pair_normalises_to_the_more_precise_type(k, v, want):
    f = ComboFlags(ctx=4096, cache_type_k=k, cache_type_v=v)
    assert (f.cache_type_k, f.cache_type_v) == (want, want)


# --- anything else is rejected at parse time ---------------------------------

@pytest.mark.parametrize("field", ["k", "v"])
def test_a_policy_rejects_a_cache_type_not_in_the_table(field):
    with pytest.raises(ValidationError):
        CachePolicy(**{field: "q3_k_bogus"})


@pytest.mark.parametrize("field", ["cache_type_k", "cache_type_v"])
def test_flags_reject_a_cache_type_not_in_the_table(field):
    with pytest.raises(ValidationError):
        ComboFlags(ctx=4096, **{field: "q3_k_bogus"})
