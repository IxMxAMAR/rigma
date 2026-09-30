"""Read GGUF header metadata without loading tensors.

A dropped fine-tune self-describes: layers, KV heads, context length, MoE,
chat template. Hangar turns that into a ModelSpec so custom models get the
same fit math as registry ones — no hand-written JSON.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# type id -> (struct fmt, size); 8=string and 9=array handled separately
_SIMPLE = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
           4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_MAX_STR = 10_000_000      # metadata strings top out ~100KB (chat templates)
_MAX_KEPT_ARRAY = 4096     # per-layer arrays are tiny; vocab arrays are not

# AUDIT 06R3-8: the layouts this reader implements, exactly. The check used to
# be `version < 2` — a floor — so a header declaring v4, v5 or 4294967295 was
# parsed as if it were v2/v3 and every field (n_layers, kv_heads,
# context_length) was silently wrong, which is how a wrong number becomes "a
# model that will not load". A later version is NOT a newer file this reader
# understands: both engines Rigma launches define GGUF_VERSION 3 — mainline
# `b9867` (ggml/include/gguf.h and gguf-py/gguf/constants.py) and PrismML
# `87268f77` (ggml/include/gguf.h) — so an allow-list trades a wrong number for
# a refusal that names the version.
_KNOWN_GGUF_VERSIONS = (2, 3)

# AUDIT 06R3-9: an upper bound for the tensor table's own arithmetic. Every
# other count the parser reads has one (`n_kv`, `n_tensors`, `n_dims`, string
# lengths); the product of a tensor's dims did not. The largest real model is
# ~1e12 parameters, so a file declaring more than 1e15 cannot be real.
_MAX_PARAMS = 10**15


class GgufParseError(ValueError):
    pass


def _read(f, fmt: str, size: int):
    raw = f.read(size)
    if len(raw) != size:
        raise GgufParseError("truncated gguf header")
    return struct.unpack(fmt, raw)[0]


def _read_str(f) -> str:
    n = _read(f, "<Q", 8)
    if n > _MAX_STR:
        raise GgufParseError(f"implausible string length {n}")
    raw = f.read(n)
    if len(raw) != n:   # a string cut at a range boundary must not silently
        raise GgufParseError("truncated gguf header")   # lose template info
    return raw.decode("utf-8", "replace")


def _read_value(f, t: int, keep: bool, depth: int = 0):
    """Read (and if `keep`, return) one value; always consumes its bytes."""
    if t == 8:
        s = _read_str(f)
        return s if keep else None
    if t == 9:
        if depth >= 8:   # real ggufs are flat; nested-array bombs are not
            raise GgufParseError("array nesting too deep")
        et = _read(f, "<I", 4)
        n = _read(f, "<Q", 8)
        keep_items = keep and n <= _MAX_KEPT_ARRAY
        out = [] if keep_items else None
        for _ in range(n):
            v = _read_value(f, et, keep_items, depth + 1)
            if keep_items:
                out.append(v)
        return out
    if t not in _SIMPLE:
        raise GgufParseError(f"unknown gguf value type {t}")
    fmt, size = _SIMPLE[t]
    v = _read(f, fmt, size)
    return (bool(v) if t == 7 else v) if keep else None


def _as_int(value, key: str) -> int:
    """A header integer, or a GgufParseError naming the key it came from.

    AUDIT F06-6: every other malformed-header path in this module raises
    GgufParseError and callers catch that type, but the numeric conversions used
    a raw int()/max() — so a header whose `attention.head_count` was an empty
    array, or whose `block_count` was declared as a string, crashed with a bare
    ValueError ("max() iterable argument is empty", "invalid literal for int()")
    that names neither the file nor the key and escapes every caller's except.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GgufParseError(f"{key} is not a number: {value!r}") from exc


def read_metadata(src) -> dict:
    """Header KVs, skipping bulky tokenizer arrays (vocab, merges, scores).

    `src` is a path or a binary file-like (e.g. BytesIO of a ranged HTTP read
    of a remote gguf — headers fit in the first few MB)."""
    if hasattr(src, "read"):
        return _read_meta(src)[0]
    with open(src, "rb") as f:
        return _read_meta(f)[0]


def _read_meta(f) -> tuple[dict, int]:
    """(metadata, tensor_count). The file is left positioned at the start of
    the tensor-info table, which is what `_read_tensors` continues from."""
    if f.read(4) != b"GGUF":
        raise GgufParseError("not a GGUF file")
    version = _read(f, "<I", 4)
    if version not in _KNOWN_GGUF_VERSIONS:
        raise GgufParseError(
            f"unsupported gguf version {version} (this reader implements "
            + " and ".join(str(v) for v in _KNOWN_GGUF_VERSIONS) + ")")
    n_tensors = _read(f, "<Q", 8)
    n_kv = _read(f, "<Q", 8)
    if n_kv > 100_000:
        raise GgufParseError("implausible metadata count")
    meta: dict = {}
    for _ in range(n_kv):
        key = _read_str(f)
        t = _read(f, "<I", 4)
        keep = (not key.startswith("tokenizer.")
                or key == "tokenizer.chat_template")
        v = _read_value(f, t, keep)
        if keep:
            meta[key] = v
    return meta, n_tensors


# Speculative-decoding heads carried INSIDE a model file. llama.cpp names the
# Qwen3-Next / DeepSeek-V3 / GLM-4.x multi-token-prediction projections
# "blk.<N>.nextn.*"; nothing else in a gguf uses that segment.
_MTP_TENSOR = (".nextn.", ".mtp.")
# MoE expert stacks: "blk.N.ffn_{gate,up,down}_exps.weight". Their share of the
# weights is what --n-cpu-moe actually moves to system RAM.
_EXPERT_TENSOR = "_exps."


@dataclass
class TensorIndex:
    """What the tensor table says, as opposed to what the header claims.

    The header is a set of assertions copied from the source config; the tensor
    table is the file's own inventory. When they disagree the table wins, which
    is the whole reason this is read at all — `nextn_predict_layers` survives a
    conversion that dropped the tensors it describes.
    """
    n_tensors: int = 0
    params: int = 0
    expert_params: int = 0
    mtp_params: int = 0
    mtp_blocks: int = 0
    max_block: int = -1
    truncated: bool = False   # ranged read ended mid-table: absence proves nothing
    # R3-ENG-2: ggml type id -> how many tensors use it.
    #
    # The type id was ALREADY being read and thrown away (it sits between the dims
    # and the offset, so skipping it is unavoidable to reach the offset). Capturing
    # it therefore costs nothing — no extra byte is read — and it is the one piece of
    # information that predicts whether an engine can load this file at all.
    #
    # The motivating case: Ternary-Bonsai-2-27B-PQ2_0 uses ggml type 142, which is
    # PRIVATE to a third-party llama.cpp fork (PrismML-Eng, GGML_TYPE_PQ2_0). Mainline
    # numbers stop at GGML_TYPE_Q2_0 = 42. Rigma planned that model, launched the
    # pinned mainline build, and the engine died at load with an opaque
    # `invalid ggml type 142. should be in [0, 42)` — a fact that was sitting in the
    # file's own header the whole time.
    #
    # Only meaningful when the table was read whole; see `types_complete`.
    type_counts: dict = field(default_factory=dict)

    @property
    def types_complete(self) -> bool:
        """True when `type_counts` covers every tensor.

        A ranged read can stop mid-table, and a partial histogram is actively
        misleading here: it would suggest a file uses only the types seen so far.
        `truncated` already carries this distinction for the same reason.
        """
        return not self.truncated

    @property
    def ggml_types(self) -> list[int] | None:
        """The distinct ggml type ids in the file, or None if not fully read."""
        if self.truncated:
            return None
        return sorted(self.type_counts)

    @property
    def has_mtp(self) -> bool | None:
        """True/False when the table was read whole; None when it was cut off —
        a truncated read cannot distinguish "no MTP" from "not read that far"."""
        if self.truncated:
            return None
        return self.mtp_blocks > 0


def _read_tensors(f, n_tensors: int) -> TensorIndex:
    """Walk the tensor-info table that follows the KV block.

    Cheap: names + dims only, no tensor DATA is touched, so this is the same
    few hundred KB the header read already paid for. Over a ranged HTTP read the
    table can be cut short — that is recorded, never guessed at."""
    idx = TensorIndex(n_tensors=n_tensors)
    if n_tensors > 1_000_000:
        raise GgufParseError("implausible tensor count")
    for _ in range(n_tensors):
        try:
            name = _read_str(f)
            n_dims = _read(f, "<I", 4)
            if n_dims > 8:            # GGML_MAX_DIMS is 4; 8 is generous
                raise GgufParseError(f"implausible tensor rank {n_dims}")
            n = 1
            for _ in range(n_dims):
                n *= _read(f, "<Q", 8)
            gtype = _read(f, "<I", 4)  # ggml type — kept, see TensorIndex.type_counts
            _read(f, "<Q", 8)         # offset
        except GgufParseError:
            idx.truncated = True
            return idx
        # AUDIT 06R3-9: the dims are multiplied with no plausibility bound, and
        # Python has no overflow — rank 8 with every dim 2**64-1 is not an
        # error, it is a silently 3.4e153-element tensor. That count then
        # survives every downstream ratio: moe_from_probe keeps a
        # plausible-looking expert share and measured_bpw turns it into a
        # -100% "drift" that reads as a real finding. The largest real model is
        # ~1e12 parameters, so 1e15 is already beyond corrupt. The check is
        # OUTSIDE the try: a GgufParseError raised inside it is caught as a
        # truncated ranged read, which is the opposite verdict.
        if n > _MAX_PARAMS or idx.params + n > _MAX_PARAMS:
            raise GgufParseError(
                f"implausible tensor size: {name!r} declares {n} elements, "
                f"which puts the file over {_MAX_PARAMS} parameters")
        idx.params += n
        idx.type_counts[gtype] = idx.type_counts.get(gtype, 0) + 1
        if name.startswith("blk."):
            head = name.split(".", 2)
            if len(head) > 1 and head[1].isdigit():
                idx.max_block = max(idx.max_block, int(head[1]))
        if any(m in name for m in _MTP_TENSOR):
            idx.mtp_params += n
            idx.mtp_blocks += 1
        elif _EXPERT_TENSOR in name:
            idx.expert_params += n
    return idx


def read_tensor_index(src) -> TensorIndex:
    """Tensor inventory for a gguf path or ranged-read file-like."""
    if hasattr(src, "read"):
        return _read_tensors(src, _read_meta(src)[1])
    with open(src, "rb") as f:
        return _read_tensors(f, _read_meta(f)[1])


@dataclass
class GgufInfo:
    name: str
    arch: str
    is_mmproj: bool
    capabilities: list[str] = field(default_factory=list)
    spec_fields: dict = field(default_factory=dict)
    # the tensor table was cut short by a ranged read — the caller may want to
    # fetch more before trusting an ABSENCE (presence is always trustworthy)
    tensors_truncated: bool = False


def inspect_gguf(src, fallback_name: str = "") -> GgufInfo:
    if hasattr(src, "read"):
        return _inspect(src, fallback_name or "model")
    with open(src, "rb") as f:
        return _inspect(f, fallback_name or Path(src).stem)


def _inspect(f, fallback: str) -> GgufInfo:
    meta, n_tensors = _read_meta(f)
    arch = str(meta.get("general.architecture", ""))
    name = str(meta.get("general.name", "") or fallback)
    if arch == "clip" or any(k.startswith("clip.") for k in meta):
        return GgufInfo(name=name, arch=arch or "clip", is_mmproj=True)
    # continues from where the KV block ended — same handle, no second read
    tx = _read_tensors(f, n_tensors)

    def g(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    # block_count INCLUDES the multi-token-prediction block. MTP is a whole
    # extra decoder layer (blk.<last> carries attn_* and ffn_* like any other,
    # plus the nextn projections) that plain decoding never runs — it exists to
    # draft tokens for speculation. Counting it as a normal layer made one model
    # read as two: unsloth's Qwen3.8-27B ships the MTP block and reported 65
    # layers, while jaromer's identical-architecture re-quant dropped it and
    # reported 64. Same weights, same geometry, two different specs — and every
    # per-layer figure derived from them (KV per token, dense offload, the MoE
    # expert share) silently disagreed by one layer's worth.
    blocks = _as_int(g("block_count", 0), f"{arch}.block_count")
    mtp_layers = _as_int(g("nextn_predict_layers", 0) or 0,
                         f"{arch}.nextn_predict_layers")
    if not mtp_layers and tx.mtp_blocks:
        # tensors carry MTP the header forgot to declare: trust the file
        mtp_layers = 1
    n_layers = max(1, blocks - mtp_layers) if blocks else 0
    heads = g("attention.head_count", 0)
    if isinstance(heads, list):
        # default=0: an empty array is corrupt, not a reason to crash unnamed
        heads = max((_as_int(h, f"{arch}.attention.head_count") for h in heads),
                    default=0)
    else:
        heads = _as_int(heads, f"{arch}.attention.head_count")
    kv = g("attention.head_count_kv", 0)
    # sliding-window attention (Gemma 2/3/4, …): only the GLOBAL layers keep a
    # KV cache that grows with context; the windowed layers are bounded by the
    # window, so counting all layers as full-attention overestimates KV ~6x and
    # makes long contexts read as "won't fit" (live repro 2026-07-18: Gemma4
    # 26B-A4B capped at 8K when it runs at 256K).
    swa = g("attention.sliding_window_pattern")   # True = windowed, False = global
    # AUDIT F19: docs/audit-2026-09-04-full.md — the windowed layers were
    # identified only to be EXCLUDED, and the geometry thrown away. llama.cpp
    # allocates them a second, window-sized KV cache (~400 MiB on a 27B-class
    # SWA model), which nothing downstream could budget because nothing carried
    # the numbers. Reported here so the resolver can charge for it; a file with
    # no window reports zeros and is charged nothing.
    swa_window = _as_int(g("attention.sliding_window", 0) or 0,
                         f"{arch}.attention.sliding_window")
    swa_layers = swa_kv_heads = 0
    # A2d-kv: positive evidence that a KV cache exists whose WIDTH or LAYER
    # SPLIT this module cannot derive. The failure mode is the same silent
    # confidence A2d removed for the recurrent state: `kv_mb` computes to a
    # confident number — often exactly 0 — with no way to tell it is a guess.
    kv_geometry_unknown = False
    # How many layers hold a FIXED recurrent state (Mamba/DeltaNet) rather than a
    # growing KV cache. Positive evidence only, from the same attention pattern
    # that decides `full_attn_layers`: a scalar kv count with a hybrid interval,
    # or a per-layer table whose zero-kv-head rows are linear attention. A
    # sliding-window model's windowed layers are NOT recurrent — they hold a
    # windowed KV cache — so that branch leaves this 0.
    recurrent_layers = 0
    if isinstance(kv, list):
        if isinstance(swa, list) and len(swa) == len(kv):
            gi = [i for i, w in enumerate(swa) if not w]   # global-layer indices
            wi = [i for i, w in enumerate(swa) if w]       # windowed-layer ones
            if wi:
                swa_layers = len(wi)
                swa_kv_heads = max(_as_int(kv[i], f"{arch}.attention.head_count_kv")
                                   for i in wi)
                # A2d-kv: the windowed layers are EXCLUDED from the growing
                # cache, so without a window length their own cache is charged
                # zero — an under-charge with no note (partial SWA geometry).
                if swa_window <= 0:
                    kv_geometry_unknown = True
            if gi:
                full_attn = len(gi)
                kv_heads = max(_as_int(kv[i], f"{arch}.attention.head_count_kv")
                               for i in gi)
            else:                                          # all windowed (rare)
                full_attn = sum(1 for h in kv
                                if _as_int(h, f"{arch}.attention.head_count_kv") > 0)
                kv_heads = max((_as_int(h, f"{arch}.attention.head_count_kv")
                                for h in kv), default=0)
                # AUDIT F19: with no global layers, `full_attn` above already
                # charges every layer at the full context. Reporting the same
                # layers as windowed too would bill them twice. Drop the
                # windowed term rather than the full one: over-charging is the
                # safe direction here, and this keeps the rare branch exactly
                # as it behaved before the windowed term existed.
                swa_layers = swa_kv_heads = 0
        else:
            # per-layer table without an SWA pattern: zeros are linear-attention
            # (DeltaNet) layers that hold no KV cache
            full_attn = sum(1 for h in kv
                            if _as_int(h, f"{arch}.attention.head_count_kv") > 0)
            kv_heads = max((_as_int(h, f"{arch}.attention.head_count_kv")
                            for h in kv), default=0)
            recurrent_layers = max(0, n_layers - full_attn)
            # A2d-kv: a window length with no usable layer pattern leaves the
            # split (WHICH layers are windowed) underivable, so the second
            # cache cannot be sized.
            if swa_window > 0:
                kv_geometry_unknown = True
    else:
        # A SCALAR kv head count does not mean every layer is full attention.
        # Qwen3.5/3.8 interleave SSM (linear-attention) layers and declare the
        # pattern as ONE NUMBER instead of a per-layer table:
        #     qwen35.full_attention_interval = 4
        #     qwen35.attention.head_count_kv = 4      <- scalar
        #     qwen35.ssm.state_size          = 128    <- the other 3-in-4
        # Only every Nth layer keeps a cache that grows with context; an SSM
        # layer's state is a fixed size. Reading the scalar as "all 65 layers
        # are full attention" overestimated KV by 4x and pinned Qwen3.8-27B at
        # 8K on a 16GB card when its real appetite allows far more (owner
        # report 2026-07-30 — the same failure Gemma's sliding-window pattern
        # got fixed for above, arriving in a different shape).
        interval = _as_int(g("full_attention_interval", 0) or 0,
                           f"{arch}.full_attention_interval")
        if interval > 1 and n_layers > 0:
            full_attn = max(1, n_layers // interval)
            recurrent_layers = max(0, n_layers - full_attn)
        else:
            full_attn = n_layers
        kv_heads = _as_int(kv, f"{arch}.attention.head_count_kv")
        # A2d-kv: a window length with no per-layer pattern leaves the split
        # underivable, exactly as in the table branch above.
        if swa_window > 0:
            kv_geometry_unknown = True
    head_dim = _as_int(g("attention.key_length", 0),
                       f"{arch}.attention.key_length") or (
        _as_int(g("embedding_length", 0), f"{arch}.embedding_length") // heads
        if heads else 0)
    caps, has_template = _capabilities(meta, tx)
    experts = _as_int(g("expert_count", 0) or 0, f"{arch}.expert_count")
    # Recurrent-state geometry. llama.cpp allocates a per-sequence RS buffer for
    # every recurrent layer, sized from these four keys (llama-hparams.cpp):
    #     n_embd_s() = ssm_d_state * ssm_d_inner
    #     n_embd_r() = (ssm_d_conv - 1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)
    # A dense file has none of them and reports zeros, so it is charged nothing.
    # `time_step_rank` is NOT read: it cancels out of n_embd_s, which the source
    # writes as state * inner.
    ssm_state = _as_int(g("ssm.state_size", 0) or 0, f"{arch}.ssm.state_size")
    ssm_inner = _as_int(g("ssm.inner_size", 0) or 0, f"{arch}.ssm.inner_size")
    ssm_conv = _as_int(g("ssm.conv_kernel", 0) or 0, f"{arch}.ssm.conv_kernel")
    ssm_groups = _as_int(g("ssm.group_count", 0) or 0,
                         f"{arch}.ssm.group_count")
    # A2d: positive evidence of a recurrent geometry whose COUNT or SIZE this
    # module cannot derive. The failure mode is a confident wrong number, so each
    # shape below is reported as unknown rather than charged zero:
    #
    #   * a pure-Mamba header has `ssm.*` and no attention pattern at all, so
    #     there is nothing to count recurrent layers from — the old code derived
    #     0, read the file as dense with a zero-width KV cache and charged no RS;
    #   * an explicit `attention.recurrent_layers` array is a shape this module
    #     does not model (indices? booleans? a count?), so it must not invent a
    #     number from it — it used to ignore it entirely;
    #   * a PARTIAL `ssm.*` set sizes S but silently drops R (conv_kernel and
    #     group_count default to 0), an under-charge with no note.
    #
    # `rs_geometry_unknown` is what `resolve.recurrent_state_unknown` reads, so
    # the fit says `rs=unknown` and the explorer table shows the same.
    declared_recurrent = g("attention.recurrent_layers")
    rs_geometry_unknown = declared_recurrent is not None
    if recurrent_layers == 0 and any((ssm_state, ssm_inner, ssm_conv, ssm_groups)):
        rs_geometry_unknown = True
    if recurrent_layers > 0 and not all((ssm_state, ssm_inner, ssm_conv,
                                         ssm_groups)):
        rs_geometry_unknown = True
    # A2d-kv: attention is declared but the growing cache's width derives to
    # zero — a missing or zero `attention.head_count_kv`, or no head width to
    # multiply it by. `kv_bytes_per_token` returns 0 and the fit read that as a
    # measured zero rather than as an absence of evidence.
    if heads > 0 and (kv_heads <= 0 or head_dim <= 0):
        kv_geometry_unknown = True
    # A2d-kv: an explicit `attention.recurrent_layers` array is a layer split
    # this module does not model (A2d). With a SCALAR kv count the
    # full-attention layer count behind `kv_mb` is the interval's guess, so the
    # KV charge is a guess too; with a per-layer kv table the split comes from
    # the table itself and the declared array does not move it.
    if declared_recurrent is not None and not isinstance(kv, list):
        kv_geometry_unknown = True
    fields = {"n_layers": n_layers, "full_attn_layers": full_attn,
              "kv_heads": kv_heads, "head_dim": head_dim,
              "native_ctx": _as_int(g("context_length", 0),
                                    f"{arch}.context_length"),
              "kind": "moe" if experts else "dense",
              # geometry needed to re-derive the above without the file in hand,
              # so a spec written by an older probe can be healed on read
              "full_attention_interval": _as_int(
                  g("full_attention_interval", 0) or 0,
                  f"{arch}.full_attention_interval"),
              # sliding-window geometry: how many layers hold a cache bounded
              # by the window rather than by ctx, how wide they are, and how
              # long the window is. All three or nothing — a zero window means
              # "no evidence of SWA", not "a window of zero".
              "swa_layers": swa_layers if swa_window else 0,
              "swa_kv_heads": swa_kv_heads if swa_window else 0,
              "swa_window": swa_window if swa_layers else 0,
              # recurrent-state geometry: how many layers, and the four ssm.*
              # numbers that size the buffer llama.cpp allocates for them.
              "recurrent_layers": recurrent_layers,
              "ssm_state_size": ssm_state,
              "ssm_inner_size": ssm_inner,
              "ssm_conv_kernel": ssm_conv,
              "ssm_group_count": ssm_groups,
              # A2d: the header gives evidence of recurrent state but the count
              # or the buffer could not be derived. The fit reports `rs=unknown`
              # instead of charging a zero it cannot stand behind.
              "rs_geometry_unknown": rs_geometry_unknown,
              # A2d-kv: the header gives evidence of a KV cache whose width or
              # layer split could not be derived, so the fit reports
              # `kv=unknown` instead of charging a confident zero (or a
              # confident estimate) it cannot stand behind.
              "kv_geometry_unknown": kv_geometry_unknown,
              "mtp_layers": mtp_layers,
              # the file's own inventory, not the header's claims
              "mtp": tx.has_mtp,
              "params": tx.params,
              "expert_params": tx.expert_params,
              "expert_used": _as_int(g("expert_used_count", 0) or 0,
                                     f"{arch}.expert_used_count"),
              "experts": experts,
              "has_template": has_template,
              # the model's OWN recommended sampling, which it ships and Rigma
              # was not reading. Qwen3.8 gguf files carry general.sampling.*
              # = top_k 20, top_p 0.95, temp 1.0 — exactly the published
              # thinking-mode preset. Guessing at these when the file states
              # them is the same mistake as reading a quant off its filename.
              "sampling": _declared_sampling(meta)}
    return GgufInfo(name=name, arch=arch, is_mmproj=False,
                    capabilities=caps, spec_fields=fields,
                    tensors_truncated=tx.truncated)


# A chat template is a Jinja program, so capability detection is necessarily a
# search for the strings that template uses to talk about a feature. Keep the
# needles broad: over-detecting costs a toggle the user can turn off, while
# under-detecting silently removes the feature from the UI with no explanation.
_TOOL_MARKS = ("tool", "function_call", "functioncall")
_THINK_MARKS = ("<think>", "</think>", "thinking", "<|thinking|>", "[think]",
                "reasoning_content", "reasoning_effort", "<|channel|>",
                "◁think▷")


_SAMPLING_KEYS = (("temp", "temperature"), ("top_p", "top_p"),
                  ("top_k", "top_k"), ("min_p", "min_p"))


def _declared_sampling(meta: dict) -> dict:
    """Sampling the gguf states for itself, under general.sampling.*."""
    out = {}
    for key, name in _SAMPLING_KEYS:
        v = meta.get(f"general.sampling.{key}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[name] = float(v)
    return out


def _capabilities(meta: dict, tx: TensorIndex) -> tuple[list[str], bool]:
    """Capabilities, plus whether the file carried a chat template at all.

    NO TEMPLATE IS NOT NO CAPABILITIES. A quantiser that drops
    tokenizer.chat_template leaves a file that reports neither tools nor
    thinking — which is indistinguishable, in a list of capabilities, from a
    model that genuinely has neither. jaromer's Qwen3.8-27B re-quant ships no
    template and so listed nothing at all, while unsloth's build of the same
    base model listed tools, thinking and vision (owner report, 2026-08-18).
    The empty list was not a finding about the model; it was the absence of the
    evidence. Callers get `has_template` so they can say so instead of
    presenting silence as an answer.
    """
    raw = meta.get("tokenizer.chat_template")
    has_template = bool(raw)
    template = str(raw or "").lower()
    caps = []
    if any(m in template for m in _TOOL_MARKS):
        caps.append("tools")
    if any(m in template for m in _THINK_MARKS):
        caps.append("thinking")
    # MTP: the tensor table decides. `nextn_predict_layers` is copied from the
    # source config and survives a conversion that dropped the tensors, so the
    # header alone can promise a draft head that isn't in the file — and asking
    # llama.cpp for draft-mtp without the tensors is a documented Vulkan
    # driver-reset loop, not a clean error. Verified presence wins; verified
    # absence wins; only when the table could not be read whole (a ranged
    # remote read cut short) do we fall back to what the header claims.
    declared = int(meta.get(
        f"{meta.get('general.architecture', '')}.nextn_predict_layers", 0) or 0) > 0
    verified = tx.has_mtp
    if verified if verified is not None else declared:
        caps.append("mtp")
    return caps, has_template
