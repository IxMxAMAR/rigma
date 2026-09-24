"""06-6: a corrupt numeric gguf header field fails as a GgufParseError.

The numeric conversions used a raw int()/max(), so a header whose
`attention.head_count` was an empty array, or whose `block_count` was declared
as a string, crashed with a bare ValueError that names neither the file nor the
key — and escapes every caller that catches GgufParseError, the module's
documented failure mode. Non-numeric values now raise GgufParseError naming the
key; an empty array is a clean zero that the downstream `<= 0` guards reject.
"""
import io
import struct

import pytest

from rigma.gguf_meta import GgufParseError, inspect_gguf

T_U32, T_STR, T_ARR = 4, 8, 9


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


def _gguf(kvs: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


BASE = [
    _kv_str(b"general.architecture", b"qwen3"),
    _kv_u32(b"qwen3.block_count", 8),
    _kv_u32(b"qwen3.context_length", 32768),
    _kv_u32(b"qwen3.embedding_length", 1024),
    _kv_u32(b"qwen3.attention.head_count", 16),
    _kv_u32(b"qwen3.attention.head_count_kv", 2),
]


def _replaced(key: bytes, kv: bytes) -> list[bytes]:
    """BASE with the entry for `key` replaced (or appended) by `kv`."""
    return [k for k in BASE if not k.startswith(_s(key))] + [kv]


def _inspect(kvs: list[bytes]):
    return inspect_gguf(io.BytesIO(_gguf(kvs)))


def test_a_valid_header_still_parses():
    info = _inspect(BASE)
    assert info.spec_fields["n_layers"] == 8
    assert info.spec_fields["native_ctx"] == 32768
    assert info.spec_fields["head_dim"] == 64


def test_an_empty_head_count_array_is_zero_not_a_crash():
    """`max()` on an empty array used to raise. Zero is the honest answer: the
    spec-level `head_dim <= 0` guard rejects the file with a real message."""
    info = _inspect(_replaced(b"qwen3.attention.head_count",
                              _kv_arr_u32(b"qwen3.attention.head_count", [])))
    assert info.spec_fields["head_dim"] == 0


@pytest.mark.parametrize("field", [
    "attention.head_count",
    "attention.head_count_kv",
    "block_count",
    "context_length",
    "embedding_length",
    "attention.sliding_window",
    "expert_count",
])
def test_a_non_numeric_header_field_raises_gguf_parse_error(field):
    key = f"qwen3.{field}".encode()
    with pytest.raises(GgufParseError, match=field.rsplit(".", 1)[-1]):
        _inspect(_replaced(key, _kv_str(key, b"not-a-number")))
