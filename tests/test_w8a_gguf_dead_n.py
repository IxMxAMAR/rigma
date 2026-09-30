"""A16e: the dead `n = 0` in `gguf_meta._read_tensors`.

The A16 verifier found an assignment no path can read: the per-tensor `try`
either sets `n` before its only later use, or returns early on GgufParseError.
It is REMOVED, not used — there is nothing to initialise, so keeping it would
only suggest a fallback value that never survives to the arithmetic.

Removing an unread assignment has no observable behaviour, so a true
before/after is not meaningful. The structural test below is the before/after
that IS available (the initialiser is gone); the behavioural pin beside it is
the loop arithmetic the assignment sat in, which must still multiply the tensor
dims out to the parameter count.
"""
from __future__ import annotations

import inspect
import struct

from rigma import gguf_meta
from rigma.gguf_meta import inspect_gguf

T_U32, T_STR = 4, 8


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _kv_u32(key: bytes, val: int) -> bytes:
    return _s(key) + struct.pack("<I", T_U32) + struct.pack("<I", val)


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _s(key) + struct.pack("<I", T_STR) + _s(val)


def _tensor(name: bytes, dims: list[int], ttype: int = 0) -> bytes:
    return (_s(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", ttype) + struct.pack("<Q", 0))


def _gguf(kvs: list[bytes], tensors: list[bytes]) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(tensors))


KV = [
    _kv_str(b"general.architecture", b"qwen3"),
    _kv_u32(b"qwen3.block_count", 1),
    _kv_u32(b"qwen3.context_length", 2048),
    _kv_u32(b"qwen3.embedding_length", 64),
    _kv_u32(b"qwen3.attention.head_count", 8),
    _kv_u32(b"qwen3.attention.head_count_kv", 1),
]


def test_read_tensors_no_longer_initialises_a_dead_n():
    src = inspect.getsource(gguf_meta._read_tensors)
    assert "n = 0" not in src, (
        "the dead `n = 0` before the per-tensor try is back; no path reads it")


def test_the_tensor_dims_still_multiply_out_to_the_parameter_count(tmp_path):
    tensors = [
        _tensor(b"blk.0.attn_q.weight", [4, 3]),             # 12
        _tensor(b"blk.0.ffn_down_exps.weight", [4, 3, 2]),   # 24, expert
    ]
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf(KV, tensors))
    fields = inspect_gguf(p).spec_fields
    assert fields["params"] == 36
    assert fields["expert_params"] == 24
