"""Model manager: install custom ggufs, list/delete models, edit capabilities.

A dropped fine-tune is inspected (gguf_meta), moved into ~/.rigma/models, and
registered as a custom spec under ~/.rigma/custom/models — from then on it
resolves, fits, and switches exactly like a registry model. Registry slugs
always win collisions; installs refuse duplicates up front.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path

from .gguf_meta import GgufParseError, inspect_gguf
from .models import GgufFile, ModelSpec, MoESpec
from .runtime import rigma_home

VALID_CAPS = ("tools", "vision", "thinking")
_QUANT_RE = re.compile(
    r"(UD-)?(I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32|MXFP4(?:_[A-Z0-9]+)*)",
    re.IGNORECASE)


class HangarError(RuntimeError):
    pass


def models_dir() -> Path:
    p = rigma_home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def custom_dir() -> Path:
    return rigma_home() / "custom" / "models"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.")
    return s or "custom-model"


def _quant_from_name(fname: str) -> str:
    m = _QUANT_RE.search(fname)
    return m.group(0).upper() if m else "GGUF"


def _write_spec(spec: ModelSpec) -> None:
    d = custom_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{spec.slug}.json.tmp"
    tmp.write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, d / f"{spec.slug}.json")


def _load_custom(slug: str) -> ModelSpec | None:
    f = custom_dir() / f"{slug}.json"
    if not f.is_file():
        return None
    return ModelSpec.model_validate_json(f.read_text(encoding="utf-8"))


def _move(src: Path, dest: Path) -> None:
    try:
        os.replace(src, dest)          # same-drive: instant rename
    except OSError:
        shutil.move(str(src), str(dest))   # cross-drive: real copy


def install_model(path: str | Path, attach_to: str | None = None) -> ModelSpec:
    """Install a local .gguf. Inspect first, move only after checks pass —
    a failed install must never eat the user's file."""
    src = Path(path).expanduser()
    if not src.is_file():
        raise HangarError(f"no such file: {src}")
    if src.suffix.lower() != ".gguf":
        raise HangarError("only .gguf files can be installed")
    try:
        info = inspect_gguf(src)
    except GgufParseError as e:
        raise HangarError(str(e)) from e

    if info.is_mmproj:
        if not attach_to:
            raise HangarError(
                "this file is a vision projector (mmproj) — install the "
                "model first, then attach this to it")
        spec = _load_custom(attach_to)
        if spec is None:
            raise HangarError(f"{attach_to} is not a custom model — "
                              "projectors can only attach to custom models")
        dest = models_dir() / src.name
        if dest.exists():
            raise HangarError(f"{dest.name} already exists in Rigma's models "
                              "folder — rename the file and try again")
        mm = GgufFile(repo="local", file=dest.name,
                      bytes=src.stat().st_size, quant="LOCAL")
        updated = spec.model_copy(update={
            "mmproj": mm,
            "capabilities": sorted(set(spec.capabilities) | {"vision"})})
        # spec first, then move: a spec-write failure leaves the user's file
        # untouched at its source (never silently swallowed)
        _write_spec(updated)
        try:
            _move(src, dest)
        except OSError:
            _write_spec(spec)   # roll the mmproj back off the model
            raise
        return updated

    from .registry import Registry
    slug = _slugify(info.name)
    if slug in Registry.load().models:
        raise HangarError(f"a model named {slug} already exists")
    f = info.spec_fields
    if f["n_layers"] <= 0 or f["kv_heads"] <= 0 or f["head_dim"] <= 0:
        raise HangarError(
            "gguf header is missing attention metadata — Rigma can't "
            "compute memory fit for this file")
    size = src.stat().st_size
    moe = None
    if f["kind"] == "moe":
        est_b = max(1.0, round(size / 2**30 * 2, 1))   # ~2B params/GB at Q4
        moe = MoESpec(total_b=est_b, active_b=max(0.5, round(est_b * 0.1, 1)),
                      expert_weight_fraction=0.85)
    dest = models_dir() / src.name
    if dest.exists():
        # same filename may back a DIFFERENT model (generic quantizer names) —
        # overwriting could destroy the running engine's weights
        raise HangarError(f"{dest.name} already exists in Rigma's models "
                          "folder — rename the file and try again")
    spec = ModelSpec(
        slug=slug, family=info.arch or "custom", kind=f["kind"],
        n_layers=f["n_layers"], full_attn_layers=f["full_attn_layers"],
        kv_heads=f["kv_heads"], head_dim=f["head_dim"],
        native_ctx=max(2048, f["native_ctx"]),
        ggufs=[GgufFile(repo="local", file=dest.name, bytes=size,
                        quant=_quant_from_name(dest.name))],
        moe=moe, license="custom import", use_cases=["general"],
        capabilities=sorted(info.capabilities), custom=True)
    # spec first, then move: if the move fails, drop the orphan spec so the
    # library never lists a model whose file isn't there
    _write_spec(spec)
    try:
        _move(src, dest)
    except OSError:
        (custom_dir() / f"{slug}.json").unlink(missing_ok=True)
        raise
    return spec


def _running_files(state: dict | None, reg) -> set[str]:
    """Files the live engine holds open: the loaded quant and its mmproj."""
    if not state:
        return set()
    spec = reg.models.get(state.get("model", ""))
    if spec is None:
        return set()
    out = {g.file for g in spec.ggufs if g.quant == state.get("quant")}
    if spec.mmproj is not None:
        out.add(spec.mmproj.file)
    return out


def list_models(registry=None) -> dict:
    from . import state as st
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    state = st.read_state()
    mdir = models_dir()
    models, used = [], 0
    for slug in sorted(reg.models):
        spec = reg.models[slug]
        quants = []
        for g in spec.ggufs:
            on_disk = (mdir / g.file).exists()
            used += g.bytes if on_disk else 0
            quants.append({"file": g.file, "quant": g.quant,
                           "bytes": g.bytes, "on_disk": on_disk,
                           # HF-added models have a real repo (downloadable);
                           # drag-dropped ones are "local" (only exist here)
                           "pullable": g.repo != "local",
                           "pull": _PULLS.get(f"{slug}::{g.file}")})
        mm = None
        if spec.mmproj is not None:
            mm_on = (mdir / spec.mmproj.file).exists()
            used += spec.mmproj.bytes if mm_on else 0
            mm = {"file": spec.mmproj.file, "bytes": spec.mmproj.bytes,
                  "on_disk": mm_on}
        models.append({
            "slug": slug, "family": spec.family, "kind": spec.kind,
            "custom": spec.custom, "capabilities": spec.capabilities,
            "native_ctx": spec.native_ctx, "quants": quants, "mmproj": mm,
            "running": bool(state and state.get("model") == slug)})
    du = shutil.disk_usage(mdir)
    return {"models": models,
            "disk": {"free_gb": round(du.free / 2**30, 1),
                     "models_gb": round(used / 2**30, 1),
                     "dir": str(mdir)}}


def delete_file(slug: str, file: str, registry=None) -> None:
    from . import state as st
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    spec = reg.models.get(slug)
    if spec is None:
        raise HangarError(f"unknown model: {slug}")
    known = {g.file for g in spec.ggufs}
    if spec.mmproj is not None:
        known.add(spec.mmproj.file)
    if file not in known:
        raise HangarError(f"{file} does not belong to {slug}")
    state = st.read_state()
    if state and state.get("model") == slug and \
            file in _running_files(state, reg):
        raise HangarError("that file is running right now — stop or "
                          "switch models first")
    target = models_dir() / file
    if not target.exists():
        raise HangarError(f"{file} is not on disk")
    target.unlink()


def delete_model(slug: str, registry=None) -> None:
    from . import state as st
    spec = _load_custom(slug)
    if spec is None:
        raise HangarError("only custom models can be removed — registry "
                          "models just have their files deleted")
    state = st.read_state()
    if state and state.get("model") == slug:
        raise HangarError(f"{slug} is running — stop or switch models first")
    for g in spec.ggufs:
        (models_dir() / g.file).unlink(missing_ok=True)
    if spec.mmproj is not None:
        (models_dir() / spec.mmproj.file).unlink(missing_ok=True)
    (custom_dir() / f"{slug}.json").unlink(missing_ok=True)


def patch_capabilities(slug: str, caps: list[str]) -> ModelSpec:
    spec = _load_custom(slug)
    if spec is None:
        raise HangarError("capabilities can only be edited on custom models")
    bad = [c for c in caps if c not in VALID_CAPS]
    if bad:
        raise HangarError(f"unknown capability: {', '.join(bad)}")
    if "vision" in caps and spec.mmproj is None:
        raise HangarError("vision needs an mmproj file — attach one first")
    spec = spec.model_copy(update={"capabilities": sorted(set(caps))})
    _write_spec(spec)
    return spec


# ---- background quant pulls (registry models) -------------------------------

_PULLS: dict[str, dict] = {}
_PULL_LOCK = threading.Lock()


def start_pull(slug: str, file: str, registry=None) -> dict:
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    spec = reg.models.get(slug)
    gguf = next((g for g in (spec.ggufs if spec else []) if g.file == file),
                None)
    if gguf is None and spec is not None and spec.mmproj is not None \
            and spec.mmproj.file == file:
        gguf = spec.mmproj
    if gguf is None:
        raise HangarError(f"{file} is not a known quant of {slug}")
    if gguf.repo == "local":
        raise HangarError("custom files can't be re-downloaded — they only "
                          "exist on this machine")
    key = f"{slug}::{file}"
    with _PULL_LOCK:
        if key in _PULLS and _PULLS[key]["status"] == "downloading":
            return _PULLS[key]
        _PULLS[key] = {"status": "downloading", "total": gguf.bytes,
                       "error": None}

    def _run():
        from .runtime import ensure_model
        try:
            ensure_model(gguf)
            _PULLS[key]["status"] = "done"
        except Exception as e:   # surfaced via /api/models, not lost in a thread
            _PULLS[key].update(status="error", error=str(e).splitlines()[0])

    threading.Thread(target=_run, daemon=True, name=f"pull:{file}").start()
    return _PULLS[key]


def pull_progress(file: str, total: int) -> int:
    """Bytes on disk so far: the final file, or hf_hub's .incomplete part."""
    final = models_dir() / file
    if final.exists():
        return min(final.stat().st_size, total)
    part_dir = models_dir() / ".cache" / "huggingface" / "download"
    best = 0
    if part_dir.is_dir():
        for p in part_dir.glob(f"*{file}*.incomplete"):
            best = max(best, p.stat().st_size)
    return min(best, total)
