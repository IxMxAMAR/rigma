"""Hangar: GGUF header parsing for custom-model import."""
import struct

import pytest

from rigma.gguf_meta import GgufParseError, inspect_gguf, read_metadata

T_U32, T_F32, T_BOOL, T_STR, T_ARR, T_U64 = 4, 6, 7, 8, 9, 10


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


def _kv_arr_str(key: bytes, vals: list[bytes]) -> bytes:
    body = b"".join(_s(v) for v in vals)
    return (_s(key) + struct.pack("<I", T_ARR)
            + struct.pack("<I", T_STR) + struct.pack("<Q", len(vals)) + body)


def _gguf(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _write(tmp_path, kvs, name="m.gguf"):
    p = tmp_path / name
    p.write_bytes(_gguf(kvs))
    return p


DENSE = [
    _kv_str(b"general.architecture", b"qwen3"),
    _kv_str(b"general.name", b"Spicy Tune 8B"),
    _kv_u32(b"qwen3.block_count", 8),
    _kv_u32(b"qwen3.context_length", 32768),
    _kv_u32(b"qwen3.embedding_length", 1024),
    _kv_u32(b"qwen3.attention.head_count", 16),
    _kv_u32(b"qwen3.attention.head_count_kv", 2),
    # huge-vocab stand-in: must be skipped, not held in memory
    _kv_arr_str(b"tokenizer.ggml.tokens", [b"a"] * 3000),
    _kv_str(b"tokenizer.chat_template",
            b"{% if tools %}...{% endif %}<think>"),
]


def test_read_metadata_skips_vocab_and_keeps_scalars(tmp_path):
    meta = read_metadata(_write(tmp_path, DENSE))
    assert meta["general.name"] == "Spicy Tune 8B"
    assert meta["qwen3.block_count"] == 8
    assert "tokenizer.ggml.tokens" not in meta   # skipped, not stored
    assert "tool" in meta["tokenizer.chat_template"]


def test_inspect_dense_spec_fields(tmp_path):
    info = inspect_gguf(_write(tmp_path, DENSE))
    assert not info.is_mmproj
    assert info.name == "Spicy Tune 8B"
    assert info.spec_fields["n_layers"] == 8
    assert info.spec_fields["full_attn_layers"] == 8
    assert info.spec_fields["kv_heads"] == 2
    assert info.spec_fields["head_dim"] == 64       # 1024 / 16
    assert info.spec_fields["native_ctx"] == 32768
    assert info.spec_fields["kind"] == "dense"
    assert set(info.capabilities) == {"tools", "thinking"}


def test_inspect_hybrid_kv_array_counts_full_attn_layers(tmp_path):
    kvs = [
        _kv_str(b"general.architecture", b"hyb"),
        _kv_u32(b"hyb.block_count", 6),
        _kv_u32(b"hyb.context_length", 8192),
        _kv_u32(b"hyb.embedding_length", 512),
        _kv_u32(b"hyb.attention.head_count", 8),
        # per-layer kv heads: zeros are linear-attention layers
        _kv_arr_u32(b"hyb.attention.head_count_kv", [0, 2, 0, 2, 0, 2]),
    ]
    info = inspect_gguf(_write(tmp_path, kvs))
    assert info.spec_fields["kv_heads"] == 2
    assert info.spec_fields["full_attn_layers"] == 3


def test_inspect_moe_detected(tmp_path):
    kvs = DENSE[:-2] + [_kv_u32(b"qwen3.expert_count", 64),
                        _kv_u32(b"qwen3.expert_used_count", 8)]
    info = inspect_gguf(_write(tmp_path, kvs))
    assert info.spec_fields["kind"] == "moe"


def test_inspect_mmproj_detected(tmp_path):
    kvs = [_kv_str(b"general.architecture", b"clip"),
           _kv_u32(b"clip.vision.image_size", 768)]
    info = inspect_gguf(_write(tmp_path, kvs))
    assert info.is_mmproj


def test_not_a_gguf_raises(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GgufParseError):
        read_metadata(p)


def test_corrupt_string_length_raises_instead_of_hanging(tmp_path):
    bad = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
           + struct.pack("<Q", 1)
           + struct.pack("<Q", 2**62) + b"xx")   # absurd key length
    p = tmp_path / "bad.gguf"
    p.write_bytes(bad)
    with pytest.raises(GgufParseError):
        read_metadata(p)


def test_nested_array_bomb_raises_instead_of_recursion_error(tmp_path):
    """Review 2026-07-17: arrays-of-arrays a few hundred deep must be a clean
    GgufParseError, not a RecursionError escaping install_model's handler."""
    depth = 400
    body = struct.pack("<I", T_ARR)              # value type: array
    for _ in range(depth - 1):                   # each level: elem=array, n=1
        body += struct.pack("<I", T_ARR) + struct.pack("<Q", 1)
    body += struct.pack("<I", T_U32) + struct.pack("<Q", 0)   # innermost: u32[0]
    kv = _s(b"bomb") + body
    p = tmp_path / "bomb.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                  + struct.pack("<Q", 1) + kv)
    with pytest.raises(GgufParseError, match="nesting"):
        read_metadata(p)


def _kv_arr_bool(key, vals):
    T_BOOL_ = 7
    body = b"".join(struct.pack("<B", 1 if v else 0) for v in vals)
    return (_s(key) + struct.pack("<I", T_ARR)
            + struct.pack("<I", T_BOOL_) + struct.pack("<Q", len(vals)) + body)


def test_sliding_window_counts_only_global_layers(tmp_path):
    """Live find 2026-07-18 (Gemma4 26B-A4B): SWA models keep a ctx-scaling KV
    only on their GLOBAL layers. Counting all layers overestimates KV ~6x and
    caps context far too low."""
    # 6 layers: every 6th is global (sliding_window_pattern False); global
    # layers have 2 kv heads, windowed have 8
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
        _kv_u32(b"gemma4.expert_count", 128),
    ]
    info = inspect_gguf(_write(tmp_path, kvs))
    assert info.spec_fields["full_attn_layers"] == 1     # only the 1 global layer
    assert info.spec_fields["kv_heads"] == 2             # global layer's kv heads
    assert info.spec_fields["head_dim"] == 512
    assert info.spec_fields["kind"] == "moe"


def test_no_swa_pattern_still_counts_nonzero_kv_layers(tmp_path):
    """A DeltaNet-style per-layer table without an SWA pattern keeps the old
    'zeros are linear-attention' behaviour."""
    kvs = [
        _kv_str(b"general.architecture", b"hyb"),
        _kv_u32(b"hyb.block_count", 6),
        _kv_u32(b"hyb.context_length", 8192),
        _kv_u32(b"hyb.embedding_length", 512),
        _kv_u32(b"hyb.attention.head_count", 8),
        _kv_arr_u32(b"hyb.attention.head_count_kv", [0, 2, 0, 2, 0, 2]),
    ]
    info = inspect_gguf(_write(tmp_path, kvs))
    assert info.spec_fields["full_attn_layers"] == 3
    assert info.spec_fields["kv_heads"] == 2
