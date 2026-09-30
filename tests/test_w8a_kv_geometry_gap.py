"""A2d-kv: `kv_mb` must not be a confident number when the geometry is missing.

The A2 family removed the same "silent confidence" for the split baseline, the
VRAM divergence and the recurrent geometry. The KV cache was the last term that
could read as measured while resting on a guess: `kv_bytes_per_token` returns 0
when `attention.head_count_kv` is absent, and `swa_kv_bytes` returns 0 when the
windowed layers were identified but their window length was not — and the fit
printed a bare `kv=0MB` (or a confident global-only number) with no provenance.

These headers are built for real and pushed through `inspect_gguf` and
`spec_fields_from_probe`, so the derived width, layer split and window term are
the probe's own — not a stubbed `ModelSpec`.
"""
from __future__ import annotations

import struct

from rigma.gguf_meta import inspect_gguf
from rigma.hangar import spec_fields_from_probe
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec)
from rigma.resolve import (_budget_rows, fit_gguf, kv_bytes_per_token,
                           quant_verdicts, swa_kv_bytes)

MIB = 2 ** 20

T_U32, T_BOOL, T_STR, T_ARR = 4, 7, 8, 9


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _kv_u32(key: bytes, val: int) -> bytes:
    return _s(key) + struct.pack("<I", T_U32) + struct.pack("<I", val)


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _s(key) + struct.pack("<I", T_STR) + _s(val)


def _kv_arr_u32(key: bytes, vals: list[int]) -> bytes:
    body = b"".join(struct.pack("<I", v) for v in vals)
    return (_s(key) + struct.pack("<I", T_ARR)
            + struct.pack("<I", T_U32) + struct.pack("<Q", len(vals)) + body)


def _kv_arr_bool(key: bytes, vals: list[bool]) -> bytes:
    body = b"".join(struct.pack("<B", 1 if v else 0) for v in vals)
    return (_s(key) + struct.pack("<I", T_ARR)
            + struct.pack("<I", T_BOOL) + struct.pack("<Q", len(vals)) + body)


def _gguf(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _write(tmp_path, kvs, name="m.gguf"):
    p = tmp_path / name
    p.write_bytes(_gguf(kvs))
    return p


# --- headers ------------------------------------------------------------------

def _dense_kvs(head_count_kv: bool = True) -> list[bytes]:
    """qwen3 dense: 8 layers, 16 heads, head_dim 64 (1024 // 16)."""
    kvs = [
        _kv_str(b"general.architecture", b"qwen3"),
        _kv_u32(b"qwen3.block_count", 8),
        _kv_u32(b"qwen3.context_length", 32768),
        _kv_u32(b"qwen3.embedding_length", 1024),
        _kv_u32(b"qwen3.attention.head_count", 16),
    ]
    if head_count_kv:
        kvs.append(_kv_u32(b"qwen3.attention.head_count_kv", 2))
    return kvs


def _swa_kvs(window: bool) -> list[bytes]:
    """gemma4-style SWA: 5 windowed layers, 1 global. `window=False` drops the
    `attention.sliding_window` key — the PARTIAL geometry the item names."""
    kvs = [
        _kv_str(b"general.architecture", b"gemma4"),
        _kv_u32(b"gemma4.block_count", 6),
        _kv_u32(b"gemma4.context_length", 262144),
        _kv_u32(b"gemma4.embedding_length", 2816),
        _kv_u32(b"gemma4.attention.head_count", 16),
        _kv_u32(b"gemma4.attention.key_length", 512),
        _kv_arr_u32(b"gemma4.attention.head_count_kv", [8, 8, 8, 8, 8, 2]),
        _kv_arr_bool(b"gemma4.attention.sliding_window_pattern",
                     [True, True, True, True, True, False]),
    ]
    if window:
        kvs.append(_kv_u32(b"gemma4.attention.sliding_window", 1024))
    return kvs


BLOCKS, INTERVAL = 8, 4
SSM = ((b"ssm.state_size", 128), (b"ssm.inner_size", 4096),
       (b"ssm.conv_kernel", 4), (b"ssm.group_count", 1))


def _hybrid_kvs(declared: bool) -> list[bytes]:
    """A scalar kv count plus `full_attention_interval` (the qwen35 shape). With
    `declared=True` the header ALSO carries an explicit recurrent-layers array,
    which is what makes the interval's full-attention split a guess."""
    kvs = [
        _kv_str(b"general.architecture", b"hyb3"),
        _kv_u32(b"hyb3.block_count", BLOCKS),
        _kv_u32(b"hyb3.context_length", 32768),
        _kv_u32(b"hyb3.embedding_length", 512),
        _kv_u32(b"hyb3.attention.head_count", 8),
        _kv_u32(b"hyb3.attention.head_count_kv", 2),      # scalar: interval shape
        _kv_u32(b"hyb3.full_attention_interval", INTERVAL),
    ]
    kvs += [_kv_u32(b"hyb3." + key, val) for key, val in SSM]
    if declared:
        kvs.append(_kv_arr_bool(b"hyb3.attention.recurrent_layers",
                                [i % INTERVAL != 0 for i in range(BLOCKS)]))
    return kvs


def _mamba_kvs() -> list[bytes]:
    """A pure-Mamba header: a complete ssm.* geometry and NO attention keys
    (mirrors tests/test_gguf_meta.py::_mamba_kvs). The probe reads it as
    `full_attn_layers == n_layers` — there is no pattern, so every layer falls
    through to full attention — with `kv_heads == 0` and `head_dim == 0`. That
    combination is what the derived clause must NOT flag."""
    return [
        _kv_str(b"general.architecture", b"mamba"),
        _kv_u32(b"mamba.block_count", 48),
        _kv_u32(b"mamba.context_length", 32768),
        _kv_u32(b"mamba.embedding_length", 2048),
        _kv_u32(b"mamba.ssm.state_size", 128),
        _kv_u32(b"mamba.ssm.inner_size", 4096),
        _kv_u32(b"mamba.ssm.conv_kernel", 4),
        _kv_u32(b"mamba.ssm.group_count", 1),
    ]


def _mistral_kvs() -> list[bytes]:
    """Mainstream Mistral: a scalar kv count plus `attention.sliding_window`,
    but NO `sliding_window_pattern`."""
    return [
        _kv_str(b"general.architecture", b"llama"),
        _kv_u32(b"llama.block_count", 32),
        _kv_u32(b"llama.context_length", 32768),
        _kv_u32(b"llama.embedding_length", 4096),
        _kv_u32(b"llama.attention.head_count", 32),
        _kv_u32(b"llama.attention.head_count_kv", 8),
        _kv_u32(b"llama.attention.sliding_window", 4096),
    ]


def _spec(tmp_path, kvs, name="m.gguf") -> ModelSpec:
    fields = inspect_gguf(_write(tmp_path, kvs, name)).spec_fields
    return ModelSpec(slug="m", family="qwen3", kind="dense", custom=True,
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


def _kv_line(explain: list[str]) -> str:
    return next(ln for ln in explain if "file=" in ln)


def _flag(spec) -> bool:
    """`kv_geometry_unknown` is new in A2d-kv. Imported here rather than at module
    scope so this file still COLLECTS against the pre-change source and every
    case fails on its own assertion instead of one collection error."""
    from rigma.resolve import kv_geometry_unknown as _kgu
    return _kgu(spec)


# --- the three shapes the flag must fire for ---------------------------------

def test_a_missing_head_count_kv_is_unknown_not_a_confident_zero(tmp_path):
    """The item's first shape: no `attention.head_count_kv`. `kv_bytes_per_token`
    returns 0, and the fit printed a bare `kv=0MB` — a number with no way to tell
    it from a model that genuinely holds no KV cache."""
    spec = _spec(tmp_path, _dense_kvs(head_count_kv=False))
    assert spec.kv_heads == 0
    assert spec.kv_geometry_unknown is True
    assert _flag(spec)

    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 0                  # the charge is unchanged...
    assert row["kv_unknown"] is True          # ...and now labelled

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=unknown " in line, line
    assert "kv=0MB" not in line, line


def test_a_partial_swa_geometry_is_unknown_not_a_silent_undercharge(tmp_path):
    """The item's second shape: the pattern identifies 5 windowed layers, but
    `attention.sliding_window` is missing. Those layers are EXCLUDED from the
    growing cache and their own cache is charged zero — an under-charge the fit
    printed as a confident global-only number."""
    spec = _spec(tmp_path, _swa_kvs(window=False))
    assert spec.swa_layers == 0 and spec.swa_window == 0   # zeroed by the guard
    assert spec.full_attn_layers == 1                      # only the global layer
    assert spec.kv_geometry_unknown is True
    assert _flag(spec)

    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 32                 # global layers only, unchanged
    assert row["kv_unknown"] is True

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=unknown(est 32MB)" in line, line


def test_a_declared_recurrent_split_over_a_scalar_kv_is_unknown(tmp_path):
    """The interaction the item names: an explicit `attention.recurrent_layers`
    array (a shape the probe does not model) written over a SCALAR kv count, so
    the full-attention layer count behind `kv_mb` is the interval's guess too."""
    spec = _spec(tmp_path, _hybrid_kvs(declared=True))
    assert spec.full_attention_interval == INTERVAL
    assert spec.kv_geometry_unknown is True
    assert _flag(spec)

    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 8
    assert row["kv_unknown"] is True

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    assert "kv=unknown(est 8MB)" in _kv_line(explain), explain


# --- negative controls: complete geometry keeps its confident charge ----------

def test_a_complete_dense_header_is_unflagged_and_byte_identical(tmp_path):
    """Complete geometry must not move: same number, same explain bytes, no flag.
    The prefix asserted here is exactly what the fit printed before A2d-kv."""
    spec = _spec(tmp_path, _dense_kvs())
    assert spec.kv_geometry_unknown is False
    assert not _flag(spec)
    assert kv_bytes_per_token(spec, "f16", "f16") == 4096   # 8*2*64*4

    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 32
    assert row["kv_unknown"] is False

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert line.startswith("Q4_K_M@ctx8192 kv=f16: file=100MB kv=32MB "), line
    assert "unknown" not in line, line


def test_a_complete_swa_geometry_charges_the_window_and_is_unflagged(tmp_path):
    """The owner's real shape: pattern AND window. All three swa_* numbers are
    present, so the second cache is charged and nothing is labelled."""
    spec = _spec(tmp_path, _swa_kvs(window=True))
    assert spec.swa_layers == 5 and spec.swa_kv_heads == 8
    assert spec.swa_window == 1024
    assert spec.kv_geometry_unknown is False
    assert not _flag(spec)
    # 8192 ctx * 1 global layer * 2 kv heads * 512 wide * 4 bytes = 32 MiB,
    # plus min(8192, 1024) * 5 * 8 * 512 * 4 = 80 MiB windowed
    assert swa_kv_bytes(spec, "f16", "f16", 8192) == 80 * MIB
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 112
    assert row["kv_unknown"] is False

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=112MB " in line, line
    assert "unknown" not in line, line


def test_the_interval_hybrid_without_the_array_keeps_its_confident_kv(tmp_path):
    """Negative control for the declared-recurrent shape: the same header with NO
    explicit array is the owner's real uniform hybrid and must keep its number."""
    spec = _spec(tmp_path, _hybrid_kvs(declared=False))
    assert spec.kv_geometry_unknown is False
    assert not _flag(spec)
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 8
    assert row["kv_unknown"] is False

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=8MB " in line, line
    assert "unknown" not in line, line


def test_a_real_mamba_header_is_not_flagged_for_kv(tmp_path):
    """Regression guard for the A2d-kv verifier FAIL: the derived clause flagged
    EVERY pure-Mamba spec `kv=unknown`, because the probe reads a Mamba header as
    `full_attn_layers == n_layers` (48) with `kv_heads == 0`, so the old
    `full_attn_layers > 0 and kv_heads <= 0` fired. `head_dim == 0` is the
    attention evidence that separates it from a dense header that merely omitted
    `attention.head_count_kv` (head_dim 64, which MUST stay flagged).

    Built from a REAL header, not a hand-made spec: the earlier control used
    `full_attn_layers=0`, a shape the probe never produces for a Mamba file, so
    it hid the bug."""
    f = inspect_gguf(_write(tmp_path, _mamba_kvs(), "mamba.gguf")).spec_fields
    assert f["kv_geometry_unknown"] is False        # the probe does not flag it
    assert f["full_attn_layers"] == 48              # no pattern: all "full"
    assert f["kv_heads"] == 0 and f["head_dim"] == 0
    spec = _spec(tmp_path, _mamba_kvs(), "mamba.gguf")
    assert spec.kv_geometry_unknown is False
    assert not _flag(spec)                          # ... and resolve must not

    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_mb"] == 0
    assert row["kv_unknown"] is False

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=0MB " in line, line                  # a confident (real) zero
    assert "kv=unknown" not in line, line
    assert "rs=unknown" in line, line               # the RS gap stays labelled


def test_a_window_without_a_pattern_is_not_flagged(tmp_path):
    """A2d-kv verifier nit, decided: a mainstream Mistral-style scalar header
    declares `attention.sliding_window` but no `sliding_window_pattern`.
    llama.cpp reads that as all layers windowed; this module cannot represent
    it, so it charges every layer at full ctx — a safe OVER-estimate, not an
    absence of evidence. Labelling a whole mainstream family `kv=unknown` is
    noise, so the window-without-pattern shape is deliberately not flagged."""
    f = inspect_gguf(_write(tmp_path, _mistral_kvs(), "mistral.gguf")).spec_fields
    assert f["swa_layers"] == 0 and f["swa_window"] == 0   # unmodelled, zeroed
    assert f["kv_geometry_unknown"] is False
    spec = _spec(tmp_path, _mistral_kvs(), "mistral.gguf")
    assert not _flag(spec)

    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    line = _kv_line(explain)
    assert "kv=unknown" not in line, line
    assert "kv=1024MB " in line, line   # full-ctx over-estimate, still confident


# --- propagation: probe -> spec -> fit/API -----------------------------------

def test_the_probe_flag_reaches_the_spec_through_the_shared_plumbing(tmp_path):
    from rigma.models import CachePolicy as _CP
    from rigma.models import GgufFile as _GF
    from rigma.models import ModelSpec as _MS

    f = inspect_gguf(_write(tmp_path, _dense_kvs(head_count_kv=False))).spec_fields
    assert f["kv_geometry_unknown"] is True
    spec = _MS(slug="m", family="qwen3", kind="dense",
               cache_type_policy=_CP(),
               ggufs=[_GF(repo="local", file="m.gguf", bytes=1, quant="Q4_K_M")],
               **spec_fields_from_probe(f))
    assert spec.kv_geometry_unknown is True
    assert _flag(spec)


def test_the_flag_survives_the_row_the_models_page_reads(tmp_path):
    """`quant_verdicts` (hangar.list_models -> GET /api/models) nests the row
    under "budget" and JSONResponse serialises it untyped, so the new key is
    additive: nothing indexes the row positionally."""
    spec = _spec(tmp_path, _dense_kvs(head_count_kv=False))
    verdicts = quant_verdicts(spec, _profile())
    assert verdicts[0]["budget"]["kv_unknown"] is True
    assert verdicts[0]["budget"]["kv_mb"] == 0


def test_a_spec_stored_before_the_flag_is_derived_unknown():
    """A spec written by an older probe carries no flag. Attention layers are
    declared but the growing cache's width is zero, which is the same statement
    — so the fit must not present that zero as measured. head_dim 64 is the
    attention evidence: a pure-Mamba spec has head_dim 0 and is excluded (see
    `test_a_real_mamba_header_is_not_flagged_for_kv`)."""
    spec = ModelSpec(slug="old", family="qwen3", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=0, head_dim=64,
                     native_ctx=8192, cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="local", file="o.gguf",
                                     bytes=100 * MIB, quant="Q4_K_M")])
    assert spec.kv_geometry_unknown is False       # not stored by the old probe
    assert _flag(spec)                             # but derived all the same
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_unknown"] is True


def test_a_mamba_shaped_stored_spec_is_not_derived_unknown():
    """The other half of the derived clause: the same missing-kv fields with
    head_dim 0 (what a Mamba header actually probes to) must NOT be flagged."""
    spec = ModelSpec(slug="m", family="mamba", kind="dense", n_layers=48,
                     full_attn_layers=48, kv_heads=0, head_dim=0,
                     native_ctx=32768, rs_geometry_unknown=True,
                     cache_type_policy=CachePolicy(),
                     ggufs=[GgufFile(repo="local", file="m.gguf",
                                     bytes=100 * MIB, quant="Q4_K_M")])
    assert spec.kv_geometry_unknown is False
    assert not _flag(spec)
    row = _budget_rows(spec, spec.ggufs[0], 0.0, 8192, 15000.0)
    assert row["kv_unknown"] is False
