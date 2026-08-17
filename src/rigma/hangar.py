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

VALID_CAPS = ("tools", "vision", "thinking", "mtp")
_QUANT_RE = re.compile(
    r"(UD-)?(I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32|MXFP4(?:_[A-Z0-9]+)*)",
    re.IGNORECASE)


DOWNLOAD_ATTEMPTS = 6   # multi-GB pulls WILL drop; resume and retry


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


def _distinct_quants(files: list[str]) -> list[str]:
    """A label per gguf that actually distinguishes them.

    _quant_from_name looks for Q4_K_M/IQ3_M-style tags. Repos that name their
    variants some other way — SC117's APEX ships I-Compact / I-Quality /
    I-Balanced — all collapse to "GGUF", so the picker showed three identical
    rows AND flagged every one of them as recommended (the badge compares on
    this label). Fall back to whatever part of the filename actually differs."""
    import os as _os
    from pathlib import Path as _P
    labels = [_quant_from_name(f) for f in files]
    if len(set(labels)) == len(labels):
        return labels                       # real quant tags: leave them alone
    stems = [_P(f).stem for f in files]
    pre = _os.path.commonprefix(stems)
    suf = _os.path.commonprefix([s[::-1] for s in stems])[::-1]
    out = []
    for stem, fallback in zip(stems, labels):
        core = stem[len(pre):len(stem) - len(suf)] if len(pre) + len(suf) < len(stem) \
            else stem
        core = core.strip("-_. ").upper()
        if fallback == "GGUF":
            # no real tag anywhere in the name (APEX's I-Compact / I-Quality):
            # the differing part of the filename IS the label
            out.append(core[:24] or fallback)
            continue
        # There IS a tag, and two files share it (jaromer ships both
        # RVN-Q4_K_M.gguf and Qwen3.8-27B-Heretic-Q4_K_M.gguf). Keep the TAG at
        # the FRONT and hang the distinguishing part off it: truncating a bare
        # core to 24 chars ate the "_M" off "QWEN3.8-27B-HERETIC-Q4_K_M", which
        # left the row unpriceable as well as unreadable.
        extra = core.replace(fallback, "").strip("-_. ")
        out.append(f"{fallback} ({extra[:14]})" if extra else fallback)
    if len(set(out)) == len(out):
        return out
    # still ambiguous (same stem in different subdirs): keep the path, which is
    # the only thing left that differs
    return [str(_P(f).with_suffix("")).replace("\\", "/").upper()[-24:]
            for f in files]


def _write_spec(spec: ModelSpec) -> None:
    d = custom_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{spec.slug}.json.tmp"
    tmp.write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, d / f"{spec.slug}.json")


def inherit_family_defaults(spec: ModelSpec) -> ModelSpec:
    """Custom imports inherit the registry sibling's card sampling and KV
    policy. An installed fine-tune of a registry model otherwise ran on raw
    llama-server defaults — no DRY, no model-card temperature, and no
    DeltaNet-aware q8_0 cache policy (audit 2026-07-21, limiting-setting #3).

    Matching is by ARCHITECTURAL FINGERPRINT (kind + layer geometry), not by
    family name: gguf arch strings ("qwen35moe") never equal registry family
    names ("qwen3.6"), but a fine-tune of the same base model shares its
    exact attention geometry."""
    try:
        from .registry import Registry
        from .models import CachePolicy
        blank = CachePolicy()
        for m in Registry.load().models.values():
            if m.custom:
                continue
            if (m.kind, m.n_layers, m.full_attn_layers, m.kv_heads,
                    m.head_dim) != (spec.kind, spec.n_layers,
                                    spec.full_attn_layers, spec.kv_heads,
                                    spec.head_dim):
                continue
            update = {}
            if not spec.default_params and m.default_params:
                update["default_params"] = dict(m.default_params)
            if spec.cache_type_policy == blank and \
                    m.cache_type_policy != blank:
                update["cache_type_policy"] = \
                    m.cache_type_policy.model_copy()
            return spec.model_copy(update=update) if update else spec
    except Exception:
        pass          # inheritance is a nicety — never block an install
    return spec


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
    spec = inherit_family_defaults(spec)
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


def list_models(registry=None, profile=None, *, kv: str = "",
                vision: bool = True, grow: str = "speed") -> dict:
    from . import state as st
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    state = st.read_state()
    mdir = models_dir()
    models, used = [], 0
    for slug in sorted(reg.models):
        spec = reg.models[slug]
        # Labels are derived HERE, not trusted from the spec. A spec written
        # before _distinct_quants existed has "GGUF" baked into every entry —
        # the APEX model shipped three rows all reading "GGUF", indistinguishable
        # (owner report 2026-07-30). Deriving on read heals those in place
        # instead of needing every stored spec rewritten.
        labels = _distinct_quants([g.file for g in spec.ggufs])
        # fit verdict per quant, against THIS machine. Without it the page
        # offered 21 quants with nothing but a size to choose between them.
        fits: list[dict] = [{} for _ in spec.ggufs]
        if profile is not None:
            try:
                from .resolve import quant_verdicts
                fits = quant_verdicts(spec, profile, kv=kv, vision=vision,
                                      grow=grow)
            except Exception:
                pass         # a fit we can't compute must not blank the page
        from .quant_quality import quality_of, total_loss
        quants = []
        for g, label, fit in zip(spec.ggufs, labels, fits):
            on_disk = (mdir / g.file).exists()
            used += g.bytes if on_disk else 0
            # reference quality for the FORMAT — None when the repo uses its own
            # naming (I-Compact etc.), because no published figure exists for
            # those and inventing one from a file size would be fabrication.
            # `total` folds in the KV cache actually chosen, so the number is
            # against the true reference: BF16 weights + f16 cache.
            k = fit.get("kv") or spec.cache_type_policy.k
            quants.append({"file": g.file, "quant": label,
                           "bytes": g.bytes, "on_disk": on_disk,
                           # HF-added models have a real repo (downloadable);
                           # drag-dropped ones are "local" (only exist here)
                           "pullable": g.repo != "local",
                           "fit": fit,
                           "quality": quality_of(label),
                           "total": total_loss(label, k),
                           "pull": _PULLS.get(f"{slug}::{g.file}")})
        mm = None
        if spec.mmproj is not None:
            mm_on = (mdir / spec.mmproj.file).exists()
            used += spec.mmproj.bytes if mm_on else 0
            mm = {"file": spec.mmproj.file, "bytes": spec.mmproj.bytes,
                  "on_disk": mm_on, "pullable": spec.mmproj.repo != "local",
                  "pull": _PULLS.get(f"{slug}::{spec.mmproj.file}")}
        best = None
        if profile is not None:
            try:
                from .resolve import recommended_quant
                best = recommended_quant(quants)
            except Exception:
                pass
        # Where it came from. The slug is the gguf's OWN general.name, which is
        # often nothing like the repo you typed ("Qwen38 Ara v5" for
        # 0bserverx/Qwen3.8-27B-Heretic-...), so without this the card cannot
        # be traced back to what was added.
        source = next((g.repo for g in spec.ggufs
                       if g.repo and g.repo != "local"), "")
        models.append({
            "slug": slug, "family": spec.family, "kind": spec.kind,
            "custom": spec.custom, "capabilities": spec.capabilities,
            "native_ctx": spec.native_ctx, "quants": quants, "mmproj": mm,
            "recommended": best, "source": source,
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
    repo, want = gguf.repo, gguf.bytes
    with _PULL_LOCK:
        if key in _PULLS and _PULLS[key]["status"] == "downloading":
            return _PULLS[key]
        _PULLS[key] = {"status": "downloading", "total": want, "done": 0,
                       "error": None}

    def _run():
        try:
            _download_file(repo, file, models_dir() / file,
                           lambda n: _PULLS[key].update(done=n))
            _PULLS[key]["status"] = "done"
        except Exception as e:   # surfaced via /api/models, not lost in a thread
            _PULLS[key].update(status="error", error=str(e).splitlines()[0])

    threading.Thread(target=_run, daemon=True, name=f"pull:{file}").start()
    return _PULLS[key]


def _download_file(repo: str, file: str, dest, report) -> int:
    """Stream a HF file straight to `dest` with resume + live byte reporting.

    Direct httpx (not hf_hub_download) on purpose: we get the exact byte count
    for a real progress bar, resume works off a plain `.part` file, and it
    never touches the xet backend that hangs on this box."""
    import httpx
    if dest.exists():
        report(dest.stat().st_size)
        return dest.stat().st_size
    import time
    tok = os.environ.get("HF_TOKEN", "")
    part = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
    # A multi-GB download WILL have the connection dropped ("peer closed
    # connection without sending complete message body"). Resume existed but
    # nothing retried, so one drop threw away the transfer. Retry from the
    # .part file instead — that's what resume is for.
    last = ""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        headers = {"authorization": f"Bearer {tok}"} if tok else {}
        have = part.stat().st_size if part.exists() else 0
        if have:
            headers["range"] = f"bytes={have}-"
        try:
            with httpx.stream("GET", url, headers=headers,
                              follow_redirects=True,
                              timeout=httpx.Timeout(60.0, read=120.0)) as r:
                if r.status_code == 416:   # range past EOF: partial is complete
                    os.replace(part, dest)
                    report(dest.stat().st_size)
                    return dest.stat().st_size
                if r.status_code in (401, 403):
                    raise HangarError("that repo is gated — accept its license "
                                      "on huggingface.co and set HF_TOKEN")
                resumed = r.status_code == 206
                if r.status_code not in (200, 206):
                    raise HangarError(
                        f"Hugging Face returned HTTP {r.status_code}")
                if not resumed:
                    have = 0               # server ignored Range — start over
                with open(part, "ab" if resumed and have else "wb") as f:
                    report(have)
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        report(have)
            os.replace(part, dest)
            report(have)
            return have
        except HangarError:
            raise                          # gated/HTTP errors are terminal
        except Exception as e:             # transport drop: resume and retry
            last = str(e)[:200]
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                break
            time.sleep(min(2 ** attempt, 20))
    done = part.stat().st_size if part.exists() else 0
    raise HangarError(
        f"download kept dropping after {DOWNLOAD_ATTEMPTS} attempts "
        f"({done:,} bytes saved — press Download again to resume from there). "
        f"Last error: {last}")


def pull_progress(file: str, total: int) -> int:
    """Live bytes for an in-flight pull (tracked in-process), else the final
    file's size on disk."""
    for key, st in list(_PULLS.items()):
        if key.rsplit("::", 1)[-1] == file and st.get("status") == "downloading":
            return min(int(st.get("done", 0)), total)
    final = models_dir() / file
    return min(final.stat().st_size, total) if final.exists() else 0
