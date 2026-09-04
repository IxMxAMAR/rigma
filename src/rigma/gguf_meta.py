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
    if version < 2:
        raise GgufParseError(f"gguf v{version} is too old")
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
            _read(f, "<I", 4)         # ggml type
            _read(f, "<Q", 8)         # offset
        except GgufParseError:
            idx.truncated = True
            return idx
        idx.params += n
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
    blocks = int(g("block_count", 0))
    mtp_layers = int(g("nextn_predict_layers", 0) or 0)
    if not mtp_layers and tx.mtp_blocks:
        # tensors carry MTP the header forgot to declare: trust the file
        mtp_layers = 1
    n_layers = max(1, blocks - mtp_layers) if blocks else 0
    heads = g("attention.head_count", 0)
    heads = max(int(h) for h in heads) if isinstance(heads, list) else int(heads)
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
    swa_window = int(g("attention.sliding_window", 0) or 0)
    swa_layers = swa_kv_heads = 0
    if isinstance(kv, list):
        if isinstance(swa, list) and len(swa) == len(kv):
            gi = [i for i, w in enumerate(swa) if not w]   # global-layer indices
            wi = [i for i, w in enumerate(swa) if w]       # windowed-layer ones
            if wi:
                swa_layers = len(wi)
                swa_kv_heads = max(int(kv[i]) for i in wi)
            if gi:
                full_attn = len(gi)
                kv_heads = max(int(kv[i]) for i in gi)
            else:                                          # all windowed (rare)
                full_attn = sum(1 for h in kv if int(h) > 0)
                kv_heads = max((int(h) for h in kv), default=0)
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
            full_attn = sum(1 for h in kv if int(h) > 0)
            kv_heads = max((int(h) for h in kv), default=0)
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
        interval = int(g("full_attention_interval", 0) or 0)
        if interval > 1 and n_layers > 0:
            full_attn = max(1, n_layers // interval)
        else:
            full_attn = n_layers
        kv_heads = int(kv)
    head_dim = int(g("attention.key_length", 0)) or (
        int(g("embedding_length", 0)) // heads if heads else 0)
    caps, has_template = _capabilities(meta, tx)
    experts = int(g("expert_count", 0) or 0)
    fields = {"n_layers": n_layers, "full_attn_layers": full_attn,
              "kv_heads": kv_heads, "head_dim": head_dim,
              "native_ctx": int(g("context_length", 0)),
              "kind": "moe" if experts else "dense",
              # geometry needed to re-derive the above without the file in hand,
              # so a spec written by an older probe can be healed on read
              "full_attention_interval": int(g("full_attention_interval", 0) or 0),
              # sliding-window geometry: how many layers hold a cache bounded
              # by the window rather than by ctx, how wide they are, and how
              # long the window is. All three or nothing — a zero window means
              # "no evidence of SWA", not "a window of zero".
              "swa_layers": swa_layers if swa_window else 0,
              "swa_kv_heads": swa_kv_heads if swa_window else 0,
              "swa_window": swa_window if swa_layers else 0,
              "mtp_layers": mtp_layers,
              # the file's own inventory, not the header's claims
              "mtp": tx.has_mtp,
              "params": tx.params,
              "expert_params": tx.expert_params,
              "expert_used": int(g("expert_used_count", 0) or 0),
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
