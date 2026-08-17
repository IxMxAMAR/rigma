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


def _tensor(name: bytes, dims: list[int], ttype: int = 0) -> bytes:
    """One tensor-info record: name, rank, dims, ggml type, data offset."""
    return (_s(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", ttype) + struct.pack("<Q", 0))


def _gguf(kvs: list[bytes], tensors: list[bytes] | None = None) -> bytes:
    tensors = tensors or []
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(tensors))


def _write(tmp_path, kvs, name="m.gguf", tensors=None):
    p = tmp_path / name
    p.write_bytes(_gguf(kvs, tensors))
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


def _qwen35(interval, blocks=65):
    """Qwen3.5/3.8 shape: SSM layers interleaved with full-attention ones, and
    the pattern declared as ONE NUMBER — with a SCALAR kv head count, not the
    per-layer table the other hybrids use."""
    kvs = [
        _kv_str(b"general.architecture", b"qwen35"),
        _kv_u32(b"qwen35.block_count", blocks),
        _kv_u32(b"qwen35.context_length", 262144),
        _kv_u32(b"qwen35.embedding_length", 5120),
        _kv_u32(b"qwen35.attention.head_count", 24),
        _kv_u32(b"qwen35.attention.head_count_kv", 4),      # scalar!
        _kv_u32(b"qwen35.attention.key_length", 256),
        _kv_u32(b"qwen35.ssm.state_size", 128),
    ]
    if interval:
        kvs.append(_kv_u32(b"qwen35.full_attention_interval", interval))
    return kvs


def test_full_attention_interval_is_honoured_with_a_scalar_kv_count(tmp_path):
    """Live 2026-07-30: Qwen3.8-27B read as 65-of-65 full-attention layers,
    overestimating its KV cache 4x and pinning it to 8K on a 16GB card. The
    hybrid branches above only fire when head_count_kv is a LIST; this model
    ships a scalar plus `full_attention_interval`, so it fell through to
    "every layer is full attention"."""
    info = inspect_gguf(_write(tmp_path, _qwen35(interval=4)))
    assert info.spec_fields["n_layers"] == 65
    assert info.spec_fields["full_attn_layers"] == 16      # 65 // 4, not 65
    assert info.spec_fields["kv_heads"] == 4
    assert info.spec_fields["head_dim"] == 256


def test_a_scalar_kv_count_without_an_interval_is_still_all_full_attention(
        tmp_path):
    # the ordinary dense case must not regress into claiming a hybrid
    info = inspect_gguf(_write(tmp_path, _qwen35(interval=None)))
    assert info.spec_fields["full_attn_layers"] == 65


@pytest.mark.parametrize("interval,blocks,want", [
    (1, 40, 40),      # interval 1 = every layer, i.e. not hybrid at all
    (2, 40, 20),
    (4, 64, 16),
    (6, 60, 10),
    (8, 4, 1),        # never zero, or KV per token becomes 0 and everything "fits"
])
def test_interval_layer_counts(tmp_path, interval, blocks, want):
    info = inspect_gguf(_write(tmp_path, _qwen35(interval, blocks)))
    assert info.spec_fields["full_attn_layers"] == want


def test_the_interval_fix_shrinks_the_kv_cache_it_was_overstating(tmp_path):
    """The fix is only worth anything if it moves the number that gates
    context. 4x less cache per token is 4x more context in the same VRAM."""
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    from rigma.resolve import kv_bytes_per_token

    def spec_for(kvs):
        f = inspect_gguf(_write(tmp_path, kvs)).spec_fields
        return ModelSpec(slug="s", family="f", kind="dense",
                         n_layers=f["n_layers"],
                         full_attn_layers=f["full_attn_layers"],
                         kv_heads=f["kv_heads"], head_dim=f["head_dim"],
                         native_ctx=f["native_ctx"],
                         cache_type_policy=CachePolicy(),
                         ggufs=[GgufFile(repo="r", file="a.gguf", bytes=1,
                                         quant="Q4_K_M")])
    fixed = kv_bytes_per_token(spec_for(_qwen35(4)), "q8_0", "q8_0")
    naive = kv_bytes_per_token(spec_for(_qwen35(None)), "q8_0", "q8_0")
    assert naive / fixed == pytest.approx(65 / 16, rel=0.01)


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


# --- MTP: the tensor table decides, not the header ---------------------------
_MTP_KVS = [
    _kv_str(b"general.architecture", b"qwen35moe"),
    _kv_str(b"general.name", b"Test MTP"),
    _kv_u32(b"qwen35moe.block_count", 5),          # 4 real layers + 1 MTP block
    _kv_u32(b"qwen35moe.context_length", 4096),
    _kv_u32(b"qwen35moe.embedding_length", 512),
    _kv_u32(b"qwen35moe.attention.head_count", 8),
    _kv_u32(b"qwen35moe.attention.head_count_kv", 2),
    _kv_u32(b"qwen35moe.attention.key_length", 64),
    _kv_u32(b"qwen35moe.expert_count", 8),
    _kv_u32(b"qwen35moe.expert_used_count", 2),
    _kv_u32(b"qwen35moe.nextn_predict_layers", 1),
    _kv_str(b"tokenizer.chat_template", b"tool <think>"),
]
_MTP_TENSORS = [
    _tensor(b"blk.0.attn_q.weight", [512, 512]),
    _tensor(b"blk.4.attn_q.weight", [512, 512]),
    _tensor(b"blk.4.nextn.eh_proj.weight", [1024, 512]),
]


def test_mtp_verified_from_tensor_table(tmp_path):
    info = inspect_gguf(_write(tmp_path, _MTP_KVS, tensors=_MTP_TENSORS))
    assert info.spec_fields["mtp"] is True
    assert "mtp" in info.capabilities
    assert "tools" in info.capabilities        # existing detection untouched


def test_header_claims_mtp_but_tensors_do_not_carry_it(tmp_path):
    """nextn_predict_layers is copied from the source config and survives a
    conversion that dropped the tensors. Trusting it promises llama.cpp a draft
    head that isn't there — which resets the GPU driver instead of erroring."""
    stripped = [t for t in _MTP_TENSORS if b"nextn" not in t]
    info = inspect_gguf(_write(tmp_path, _MTP_KVS, tensors=stripped))
    assert info.spec_fields["mtp"] is False
    assert "mtp" not in info.capabilities      # the file overrules the header


def test_mtp_block_is_not_counted_as_a_decoder_layer(tmp_path):
    """block_count includes the MTP block. Counting it made one model read as
    two: the same architecture reported 65 layers where the MTP block was kept
    and 64 where it was dropped."""
    info = inspect_gguf(_write(tmp_path, _MTP_KVS, tensors=_MTP_TENSORS))
    assert info.spec_fields["n_layers"] == 4       # 5 blocks - 1 MTP block
    assert info.spec_fields["mtp_layers"] == 1


def test_no_tensor_table_leaves_mtp_unknown_not_absent(tmp_path):
    """A ranged read cut short cannot tell 'no MTP' from 'not read that far'."""
    import io
    blob = _gguf(_MTP_KVS, _MTP_TENSORS)
    truncated = blob[:len(blob) - 20]
    info = inspect_gguf(io.BytesIO(truncated))
    assert info.spec_fields["mtp"] is None
    assert info.tensors_truncated is True
    assert "mtp" in info.capabilities          # falls back to the header claim


def test_parameter_and_expert_counts_come_from_tensor_dims(tmp_path):
    tensors = [
        _tensor(b"blk.0.attn_q.weight", [10, 10]),        # 100
        _tensor(b"blk.0.ffn_down_exps.weight", [10, 10, 4]),   # 400 expert
    ]
    info = inspect_gguf(_write(tmp_path, _MTP_KVS, tensors=tensors))
    assert info.spec_fields["params"] == 500
    assert info.spec_fields["expert_params"] == 400


# --- an absent chat template is not a finding about the model ----------------
def test_missing_chat_template_is_reported_not_silently_empty(tmp_path):
    kvs = [k for k in _MTP_KVS if b"chat_template" not in k]
    info = inspect_gguf(_write(tmp_path, kvs, tensors=_MTP_TENSORS))
    assert info.spec_fields["has_template"] is False
    assert "tools" not in info.capabilities
    assert "thinking" not in info.capabilities


def test_template_present_but_featureless_is_distinguishable(tmp_path):
    kvs = [k for k in _MTP_KVS if b"chat_template" not in k]
    kvs.append(_kv_str(b"tokenizer.chat_template", b"{{ messages }}"))
    info = inspect_gguf(_write(tmp_path, kvs, tensors=_MTP_TENSORS))
    assert info.spec_fields["has_template"] is True   # evidence exists...
    assert "tools" not in info.capabilities           # ...and says no tools


@pytest.mark.parametrize("template,cap", [
    (b"<think>", "thinking"),
    (b"reasoning_content", "thinking"),
    (b"enable_thinking", "thinking"),
    (b"<|channel|>analysis", "thinking"),
    (b"tool_calls", "tools"),
    (b"function_call", "tools"),
])
def test_capability_needles_cover_real_template_dialects(tmp_path, template, cap):
    kvs = [k for k in _MTP_KVS if b"chat_template" not in k]
    kvs.append(_kv_str(b"tokenizer.chat_template", template))
    info = inspect_gguf(_write(tmp_path, kvs, tensors=_MTP_TENSORS))
    assert cap in info.capabilities
