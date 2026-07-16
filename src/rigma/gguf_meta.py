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
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t: int, keep: bool):
    """Read (and if `keep`, return) one value; always consumes its bytes."""
    if t == 8:
        s = _read_str(f)
        return s if keep else None
    if t == 9:
        et = _read(f, "<I", 4)
        n = _read(f, "<Q", 8)
        keep_items = keep and n <= _MAX_KEPT_ARRAY
        out = [] if keep_items else None
        for _ in range(n):
            v = _read_value(f, et, keep_items)
            if keep_items:
                out.append(v)
        return out
    if t not in _SIMPLE:
        raise GgufParseError(f"unknown gguf value type {t}")
    fmt, size = _SIMPLE[t]
    v = _read(f, fmt, size)
    return (bool(v) if t == 7 else v) if keep else None


def read_metadata(path: Path) -> dict:
    """Header KVs, skipping bulky tokenizer arrays (vocab, merges, scores)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GgufParseError(f"{Path(path).name} is not a GGUF file")
        version = _read(f, "<I", 4)
        if version < 2:
            raise GgufParseError(f"gguf v{version} is too old")
        _read(f, "<Q", 8)                       # tensor_count
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
        return meta


@dataclass
class GgufInfo:
    name: str
    arch: str
    is_mmproj: bool
    capabilities: list[str] = field(default_factory=list)
    spec_fields: dict = field(default_factory=dict)


def inspect_gguf(path: Path) -> GgufInfo:
    meta = read_metadata(path)
    arch = str(meta.get("general.architecture", ""))
    name = str(meta.get("general.name", "") or Path(path).stem)
    if arch == "clip" or any(k.startswith("clip.") for k in meta):
        return GgufInfo(name=name, arch=arch or "clip", is_mmproj=True)

    def g(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n_layers = int(g("block_count", 0))
    heads = g("attention.head_count", 0)
    heads = max(int(h) for h in heads) if isinstance(heads, list) else int(heads)
    kv = g("attention.head_count_kv", 0)
    if isinstance(kv, list):
        # per-layer table: zeros are linear-attention layers with no KV cache
        full_attn = sum(1 for h in kv if int(h) > 0)
        kv_heads = max((int(h) for h in kv), default=0)
    else:
        full_attn, kv_heads = n_layers, int(kv)
    head_dim = int(g("attention.key_length", 0)) or (
        int(g("embedding_length", 0)) // heads if heads else 0)
    template = str(meta.get("tokenizer.chat_template", ""))
    caps = []
    if "tool" in template:
        caps.append("tools")
    if "<think>" in template or "thinking" in template:
        caps.append("thinking")
    fields = {"n_layers": n_layers, "full_attn_layers": full_attn,
              "kv_heads": kv_heads, "head_dim": head_dim,
              "native_ctx": int(g("context_length", 0)),
              "kind": "moe" if int(g("expert_count", 0) or 0) else "dense"}
    return GgufInfo(name=name, arch=arch, is_mmproj=False,
                    capabilities=caps, spec_fields=fields)
