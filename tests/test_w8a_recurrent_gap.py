"""A2d-gap: `rs=unknown` must fire where the count can actually be wrong.

A2d (6883f5f) makes an unrecognised recurrent geometry report `rs=unknown`
instead of a silent zero — but the label sat in the `elif` AFTER `if rs_mb:`.
So the one shape where the interval-derived count is least trustworthy still
printed a confident number: a header carrying BOTH an explicit
`attention.recurrent_layers` array (the upstream shape, ggml-org/llama.cpp
#28207, written alongside `full_attention_interval`) AND a complete `ssm.*`
set. `_inspect` derives `recurrent_layers` from the interval, flags the
geometry unknown, `recurrent_state_mb` returns a nonzero estimate, and the fit
charged it under a confident `rs=<N>MB`.

These headers are built for real and pushed through `inspect_gguf` and
`spec_fields_from_probe`, so the interval-derived count and the buffer size are
the probe's own — not a stubbed `ModelSpec`.
"""
from __future__ import annotations

import struct

from rigma.gguf_meta import inspect_gguf
from rigma.hangar import spec_fields_from_probe
from rigma.models import (LAUNCH_PARALLEL, CachePolicy, CpuInfo, GgufFile,
                          GpuInfo, HardwareProfile, ModelSpec)
from rigma.resolve import (_budget_rows, fit_gguf, recurrent_state_mb,
                           recurrent_state_unknown)

MIB = 2 ** 20

T_U32, T_BOOL, T_STR, T_ARR = 4, 7, 8, 9


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _kv_u32(key: bytes, val: int) -> bytes:
    return _s(key) + struct.pack("<I", T_U32) + struct.pack("<I", val)


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _s(key) + struct.pack("<I", T_STR) + _s(val)


def _kv_arr_bool(key: bytes, vals: list[bool]) -> bytes:
    body = b"".join(struct.pack("<B", 1 if v else 0) for v in vals)
    return (_s(key) + struct.pack("<I", T_ARR)
            + struct.pack("<I", T_BOOL) + struct.pack("<Q", len(vals)) + body)


def _gguf(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _write(tmp_path, kvs, name="hyb.gguf"):
    p = tmp_path / name
    p.write_bytes(_gguf(kvs))
    return p


BLOCKS, INTERVAL = 8, 4
# A complete ssm.* set: the buffer IS derivable from these; only the COUNT is
# the interval's guess, which is the whole point of the case.
SSM = ((b"ssm.state_size", 128), (b"ssm.inner_size", 4096),
       (b"ssm.conv_kernel", 4), (b"ssm.group_count", 1))


def _hybrid_kvs(declared: bool, ssm: bool = True) -> list[bytes]:
    kvs = [
        _kv_str(b"general.architecture", b"hyb3"),
        _kv_u32(b"hyb3.block_count", BLOCKS),
        _kv_u32(b"hyb3.context_length", 32768),
        _kv_u32(b"hyb3.embedding_length", 512),
        _kv_u32(b"hyb3.attention.head_count", 8),
        _kv_u32(b"hyb3.attention.head_count_kv", 2),      # scalar: interval shape
        _kv_u32(b"hyb3.full_attention_interval", INTERVAL),
    ]
    if ssm:
        kvs += [_kv_u32(b"hyb3." + key, val) for key, val in SSM]
    if declared:
        # the upstream shape: a per-layer bool array of length n_layers,
        # written ALONGSIDE full_attention_interval
        kvs.append(_kv_arr_bool(b"hyb3.attention.recurrent_layers",
                                [i % INTERVAL != 0 for i in range(BLOCKS)]))
    return kvs


def _spec(tmp_path, declared: bool, ssm: bool = True,
          name="hyb.gguf") -> ModelSpec:
    fields = inspect_gguf(_write(tmp_path, _hybrid_kvs(declared, ssm),
                                 name)).spec_fields
    return ModelSpec(slug="hyb", family="hyb3", kind="dense", custom=True,
                     cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="local", file=name,
                                     bytes=100 * MIB, quant="Q4_K_M")],
                     **spec_fields_from_probe(fields))


def _profile(vram=16368):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=24000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _rs_line(explain: list[str]) -> str:
    return next(ln for ln in explain if "rs=" in ln)


def test_the_both_present_shape_is_labelled_unknown_not_a_confident_number(tmp_path):
    """The hole A2d left: an explicit recurrent_layers array PLUS complete
    ssm.*. The count is the interval's guess, the size is real, so the fit used
    to print a confident `rs=<N>MB` for exactly the case that cannot stand
    behind the number."""
    spec = _spec(tmp_path, declared=True)
    # the interval-derived count and the buffer are REACHABLE, not stubbed:
    # both come out of the probe for this header
    assert BLOCKS - BLOCKS // INTERVAL == 6
    assert spec.recurrent_layers == 6
    assert spec.rs_geometry_unknown is True
    per_seq = recurrent_state_mb(spec)
    assert per_seq > 0, "the interval-derived count did not produce a buffer"
    assert recurrent_state_unknown(spec)

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _rs_line(explain)                      # what the user reads
    assert "rs=unknown" in line, line
    est = per_seq * LAUNCH_PARALLEL
    assert f"est {est:.0f}MB" in line, line       # charged, but as an estimate
    assert f"rs={est:.0f}MB " not in line, line   # never a confident number


def test_a_fully_declared_uniform_hybrid_still_reports_a_confident_number(tmp_path):
    """Negative control: the owner's real shape — the interval with NO explicit
    array — must keep its confident charge, or every hybrid loses it."""
    spec = _spec(tmp_path, declared=False)
    assert spec.recurrent_layers == BLOCKS - BLOCKS // INTERVAL
    assert spec.rs_geometry_unknown is False
    assert not recurrent_state_unknown(spec)

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _rs_line(explain)
    est = recurrent_state_mb(spec) * LAUNCH_PARALLEL
    assert f"rs={est:.0f}MB " in line, line
    assert "unknown" not in line, line


def test_a_dense_model_is_unaffected(tmp_path):
    kvs = [
        _kv_str(b"general.architecture", b"qwen3"),
        _kv_u32(b"qwen3.block_count", 8),
        _kv_u32(b"qwen3.context_length", 32768),
        _kv_u32(b"qwen3.embedding_length", 1024),
        _kv_u32(b"qwen3.attention.head_count", 16),
        _kv_u32(b"qwen3.attention.head_count_kv", 2),
    ]
    fields = inspect_gguf(_write(tmp_path, kvs, "dense.gguf")).spec_fields
    spec = ModelSpec(slug="dense", family="qwen3", kind="dense", custom=True,
                     cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="local", file="dense.gguf",
                                     bytes=100 * MIB, quant="Q4_K_M")],
                     **spec_fields_from_probe(fields))
    assert spec.recurrent_layers == 0
    assert not recurrent_state_unknown(spec)
    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    assert not any("rs=" in ln for ln in explain), explain


def test_the_budget_row_marks_the_unknown_recurrent_geometry(tmp_path):
    """A2d-budget: the fit's explain line says `rs=unknown(est N MB)` for the
    both-present shape, but `_budget_rows` — the arithmetic behind the Models
    page's "OVER by / headroom" line — emitted a bare numeric `rs_mb` with
    nothing to say the number is an interval estimate, not a measured one.

    The flag rides BESIDE the charge, never in place of it: dropping the term
    would free VRAM llama.cpp is about to allocate, which is the launch OOM the
    fit exists to prevent. The fix is the label, not the number."""
    spec = _spec(tmp_path, declared=True)
    assert recurrent_state_unknown(spec)
    rs_mb = recurrent_state_mb(spec) * LAUNCH_PARALLEL
    assert rs_mb > 0, "the interval-derived count did not produce a buffer"
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["rs_mb"] == round(rs_mb)      # still charged, unchanged
    assert row["rs_unknown"] is True         # and now labelled
    # ... and it survives the path the Models page actually reads:
    # quant_verdicts (hangar.list_models -> GET /api/models) puts this dict under
    # "budget" and JSONResponse serialises it untyped, so the new key is additive
    # there too — no consumer indexes the row positionally.
    from rigma.resolve import quant_verdicts
    verdicts = quant_verdicts(spec, _profile())
    assert verdicts[0]["budget"]["rs_unknown"] is True
    assert verdicts[0]["budget"]["rs_mb"] == round(rs_mb)


def test_a_zero_recurrent_charge_with_no_geometry_is_marked_unknown(tmp_path):
    """A2d's original shape: evidence of recurrent layers, no `ssm.*` to size
    them with. The row charges 0 AND says the 0 is an absence of evidence —
    without the flag a reader cannot tell it from a dense model's real zero."""
    spec = _spec(tmp_path, declared=False, ssm=False)
    assert spec.recurrent_layers > 0
    assert recurrent_state_mb(spec) == 0.0
    assert recurrent_state_unknown(spec)
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["rs_mb"] == 0
    assert row["rs_unknown"] is True


def test_a_fully_declared_uniform_hybrid_is_not_marked_unknown(tmp_path):
    """Negative control: the owner's real shape keeps its confident charge and
    must NOT be labelled — a flag that fires on every hybrid is noise."""
    spec = _spec(tmp_path, declared=False)
    assert not recurrent_state_unknown(spec)
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["rs_mb"] > 0
    assert row["rs_unknown"] is False
