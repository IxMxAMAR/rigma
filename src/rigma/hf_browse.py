"""Browse Hugging Face for gguf models and fit-check them BEFORE download.

The trick: HF serves ranged reads, so the first few MB of any remote gguf
give us the same header gguf_meta parses locally — layers, KV geometry,
context, template capabilities. That feeds the real fit calculator, so every
quant in a repo gets an honest "fits at ~N ctx / too big" verdict costing
megabytes, not a 20 GB download.
"""
from __future__ import annotations

import io
import os
import re

import httpx

from .gguf_meta import GgufParseError, inspect_gguf
from .hangar import HangarError, _quant_from_name, _slugify, _write_spec
from .models import GgufFile, ModelSpec, MoESpec

HF = "https://huggingface.co"
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
_RANGE_STEPS_MB = (8, 32, 64)   # escalate when a huge vocab pads the header


def _headers() -> dict:
    tok = os.environ.get("HF_TOKEN", "")
    return {"authorization": f"Bearer {tok}"} if tok else {}


def _get_json(path: str, params: dict | None = None):
    try:
        r = httpx.get(HF + path, params=params, headers=_headers(),
                      timeout=20, follow_redirects=True)
    except httpx.HTTPError as e:
        raise HangarError(f"Hugging Face unreachable: {e}") from e
    if r.status_code in (401, 403):
        raise HangarError("that repo is gated — accept its license on "
                          "huggingface.co and set HF_TOKEN")
    if r.status_code == 404:
        raise HangarError("not found on Hugging Face")
    r.raise_for_status()
    return r.json()


def _fetch_head(repo: str, file: str, cap_mb: int) -> bytes:
    """First cap_mb MB of a repo file (Range survives the CDN redirect)."""
    try:
        r = httpx.get(f"{HF}/{repo}/resolve/main/{file}",
                      headers={**_headers(),
                               "range": f"bytes=0-{cap_mb * 2**20 - 1}"},
                      timeout=90, follow_redirects=True)
    except httpx.HTTPError as e:
        raise HangarError(f"Hugging Face unreachable: {e}") from e
    if r.status_code in (401, 403):
        raise HangarError("that repo is gated — accept its license on "
                          "huggingface.co and set HF_TOKEN")
    if r.status_code not in (200, 206):
        raise HangarError(f"couldn't read {file} (HTTP {r.status_code})")
    return r.content


def search(query: str, limit: int = 12) -> list[dict]:
    rows = _get_json("/api/models", {"filter": "gguf", "search": query,
                                     "sort": "downloads", "direction": -1,
                                     "limit": limit})
    return [{"repo": m.get("id", ""), "downloads": m.get("downloads", 0),
             "likes": m.get("likes", 0),
             "updated": str(m.get("lastModified", ""))[:10]}
            for m in rows if m.get("id")]


def repo_files(repo: str) -> dict:
    tree = _get_json(f"/api/models/{repo}/tree/main")
    ggufs, mmprojs, split = [], [], 0
    for f in tree:
        p = str(f.get("path", ""))
        if not p.lower().endswith(".gguf"):
            continue
        if _SPLIT_RE.search(p):
            split += 1
            continue
        entry = {"file": p, "bytes": int(f.get("size", 0) or 0)}
        (mmprojs if "mmproj" in p.lower() else ggufs).append(entry)
    ggufs.sort(key=lambda g: -g["bytes"])    # registry order: largest first
    mm = None
    if mmprojs:   # f16 is the usual quality/size sweet spot; else biggest
        mm = next((m for m in mmprojs if "f16" in m["file"].lower()),
                  max(mmprojs, key=lambda m: m["bytes"]))
    return {"ggufs": ggufs, "mmproj": mm, "split_skipped": split}


def remote_inspect(repo: str, file: str):
    """gguf_meta over a ranged read; escalates if the header is padded out
    by a giant vocab array."""
    for cap in _RANGE_STEPS_MB:
        blob = _fetch_head(repo, file, cap)
        try:
            return inspect_gguf(io.BytesIO(blob), fallback_name=file)
        except GgufParseError as e:
            if "truncated" in str(e) and cap != _RANGE_STEPS_MB[-1] \
                    and len(blob) >= cap * 2**20:
                continue   # header really is bigger than this range
            raise HangarError(f"couldn't parse {file}: {e}") from e


def _spec_from_repo(repo: str) -> tuple[ModelSpec, dict]:
    rf = repo_files(repo)
    if not rf["ggufs"]:
        extra = (f" ({rf['split_skipped']} split .gguf parts skipped — "
                 "split files aren't supported yet)"
                 if rf["split_skipped"] else "")
        raise HangarError(f"no single-file gguf in that repo{extra}")
    probe = min(rf["ggufs"], key=lambda g: g["bytes"])   # cheapest header
    info = remote_inspect(repo, probe["file"])
    if info.is_mmproj:
        raise HangarError("that repo only holds a vision projector")
    f = info.spec_fields
    if f["n_layers"] <= 0 or f["kv_heads"] <= 0 or f["head_dim"] <= 0:
        raise HangarError("the gguf header lacks attention metadata — "
                          "Rigma can't compute memory fit for it")
    caps = sorted(set(info.capabilities)
                  | ({"vision"} if rf["mmproj"] else set()))
    moe = None
    if f["kind"] == "moe":
        big = rf["ggufs"][0]["bytes"]
        est_b = max(1.0, round(big / 2**30 * 2, 1))
        moe = MoESpec(total_b=est_b, active_b=max(0.5, round(est_b * 0.1, 1)),
                      expert_weight_fraction=0.85)
    mm = None
    if rf["mmproj"]:
        mm = GgufFile(repo=repo, file=rf["mmproj"]["file"],
                      bytes=rf["mmproj"]["bytes"],
                      quant=_quant_from_name(rf["mmproj"]["file"]))
    spec = ModelSpec(
        slug=_slugify(info.name), family=info.arch or "custom",
        kind=f["kind"], n_layers=f["n_layers"],
        full_attn_layers=f["full_attn_layers"], kv_heads=f["kv_heads"],
        head_dim=f["head_dim"], native_ctx=max(2048, f["native_ctx"]),
        ggufs=[GgufFile(repo=repo, file=g["file"], bytes=g["bytes"],
                        quant=_quant_from_name(g["file"]))
               for g in rf["ggufs"]],
        moe=moe, mmproj=mm, license="see model card", use_cases=["general"],
        capabilities=caps, custom=True,
        sources=[f"{HF}/{repo}"])
    return spec, rf


def inspect_repo(repo: str, registry=None, profile=None) -> dict:
    """Everything the browser UI needs: header-derived facts + a fit verdict
    per quant against THIS machine, before any download."""
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import _grow_ctx, fit_gguf
    spec, rf = _spec_from_repo(repo)
    reg = registry if registry is not None else Registry.load()
    prof = profile if profile is not None else probe_hardware(reg.gpus)
    quants = []
    for g in spec.ggufs:
        flags = None
        for ctx in (8192, 4096, 2048):
            flags = fit_gguf(spec, g, prof, ctx, [])
            if flags:
                flags = _grow_ctx(spec, g, prof, flags, [])
                break
        quants.append({"file": g.file, "quant": g.quant, "bytes": g.bytes,
                       "fit": ({"ok": True, "ctx": flags.ctx,
                                "n_cpu_moe": flags.n_cpu_moe}
                               if flags else {"ok": False})})
    return {"repo": repo, "name": spec.slug, "family": spec.family,
            "kind": spec.kind, "native_ctx": spec.native_ctx,
            "capabilities": spec.capabilities,
            "already": spec.slug in reg.models,
            "mmproj": rf["mmproj"], "split_skipped": rf["split_skipped"],
            "ggufs": quants}


def add_model(repo: str, registry=None) -> ModelSpec:
    """Register the repo as a library model (no download yet — quants pull
    on demand through the normal Models-tab buttons)."""
    from .registry import Registry
    spec, _ = _spec_from_repo(repo)
    reg = registry if registry is not None else Registry.load()
    if spec.slug in reg.models:
        raise HangarError(f"{spec.slug} is already in your library")
    _write_spec(spec)
    return spec
