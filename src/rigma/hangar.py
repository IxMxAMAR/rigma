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

# How many ggufs a repo may hold before `reprobe` stops asking every file
# whether it carries the MTP draft head.
#
# MTP is a per-FILE fact, and reading one file left every other row as None —
# which the Models page rendered identically to a verified "no". On a repo whose
# every quant carries the head (SC117's ...-MTP-APEX-GGUF) that reads as "only
# this one supports it", and steers the choice for a reason that is not true.
#
# The limit exists because a ranged header read escalates to 8, 32 or 64MB and
# 0bserverx/Qwen3.8-27B-Heretic publishes 103 ggufs. Small repos are worth the
# handful of requests; that one is not, and its rows stay honestly unknown.
MTP_PROBE_LIMIT = 8
# Bump when the probe learns something a stored spec would have got wrong.
#   1  layer geometry from gguf_meta, capabilities from the chat template
#   2  hybrid attention read from full_attention_interval (Qwen3.5/3.8)
#   3  MTP block excluded from n_layers; MTP verified from the tensor table;
#      exact parameter/expert counts; "no chat template" told apart from
#      "no capabilities"
PROBE_VERSION = 3
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


# A model is keyed by the `general.name` inside its gguf, which is what makes
# two mirrors of one model dedupe instead of filling the library with copies.
# Some quantisers never set it and the conversion default survives: the Apriel
# 1.6 decensored build calls itself "base-model", so it sits in the library as
# `base-model` with nothing to connect it to what was added. Worse, the name is
# not unique — the NEXT model that self-describes the same way is refused as a
# duplicate of a model it has nothing to do with.
_GENERIC_NAMES = {
    "base-model", "base", "model", "models", "gguf", "ggml-model", "output",
    "unnamed", "merged", "merged-model", "merge", "finetune", "fine-tune",
    "checkpoint", "pytorch-model", "converted", "llama", "mistral", "unsloth",
    # what _slugify itself returns when the name had nothing usable in it
    "custom-model",
}


def model_slug(gguf_name: str, fallback: str) -> str:
    """Slug from the gguf's own name, unless that name says nothing.

    The fallback is whatever the user actually pointed at — the repo id, or the
    filename for a local install. Never silently: `_already_msg` explains the
    slug/repo relationship when a collision is reported."""
    s = _slugify(gguf_name)
    if s in _GENERIC_NAMES or len(s) < 3:
        return _slugify(fallback) or s
    return s


def _quant_from_name(fname: str) -> str:
    m = _QUANT_RE.search(fname)
    return m.group(0).upper() if m else "GGUF"


_TOKEN_SPLIT = re.compile(r"[-_.]+")


def quant_variants(files: list[str]) -> list[tuple[str, list[str]]]:
    """Split each gguf name into (base label, variant tags).

    Big repos do not ship a flat list of quants — they ship a GRID. 0bserverx's
    Qwen3.8-27B publishes 103 ggufs that are really 22 quants crossed with
    -multilingual, -mtp and -vision. Flat, that is a 103-row wall nobody can
    read; as a grid it is 23 rows and three checkboxes (owner, 2026-08-21).

    A variant is a token that FOLLOWS the quant tag. Everything before the tag
    is the uploader's prefix ("RVN-"), which is not an axis — it is the same on
    nearly every file. When a repo mixes prefixes, the minority ones are named
    in full so two files sharing a quant tag stay distinguishable.
    """
    import os as _os
    from collections import Counter
    from pathlib import Path as _P
    stems = [_P(f).stem for f in files]
    tags = [_quant_from_name(f) for f in files]
    # For names carrying no quant tag at all (SC117's APEX ships I-Compact /
    # I-Quality / I-Balanced) there is nothing to anchor on, so the label is
    # whatever part of the filename actually differs from its siblings.
    pre = _os.path.commonprefix(stems)
    suf = _os.path.commonprefix([x[::-1] for x in stems])[::-1]
    heads, variants = [], []
    for stem, tag in zip(stems, tags):
        cut = stem.upper().rfind(tag) if tag != "GGUF" else -1
        head, tail = (stem, "") if cut < 0 else (stem[:cut],
                                                 stem[cut + len(tag):])
        heads.append(head.strip("-_. "))
        variants.append([t.lower() for t in _TOKEN_SPLIT.split(tail) if t])
    seen = Counter(h for h in heads if h)
    main = seen.most_common(1)[0][0] if seen else ""
    bases = []
    for stem, tag, head in zip(stems, tags, heads):
        if tag == "GGUF":
            core = (stem[len(pre):len(stem) - len(suf)]
                    if len(pre) + len(suf) < len(stem) else stem)
            bases.append(core.strip("-_. ").upper() or "GGUF")
        elif head and head != main:
            bases.append(f"{tag} ({head})")     # minority prefix, kept in full
        else:
            bases.append(tag)
    return list(zip(bases, variants))


def _distinct_quants(files: list[str]) -> list[str]:
    """A label per gguf that actually distinguishes them.

    _quant_from_name looks for Q4_K_M/IQ3_M-style tags. Repos that name their
    variants some other way — SC117's APEX ships I-Compact / I-Quality /
    I-Balanced — all collapse to "GGUF", so the picker showed three identical
    rows AND flagged every one of them as recommended (the badge compares on
    this label). Fall back to whatever part of the filename actually differs."""
    from pathlib import Path as _P
    labels = [_quant_from_name(f) for f in files]
    if len(set(labels)) == len(labels):
        return labels                       # real quant tags: leave them alone
    # There IS a tag and two files share it, so something else in the name has
    # to earn its place. quant_variants knows which part: whatever follows the
    # tag. "RVN-Q3_K_M-multilingual-mtp" becomes "Q3_K_M · multilingual+mtp"
    # — the tag stays at the FRONT, where it is readable and priceable, and
    # nothing is truncated. The old code cut `extra` to 14 chars, which made
    # "RVN--MULTILINGUAL" and "RVN--MULTILINGUAL-MTP" collide (2026-08-21).
    # ASCII on purpose: these labels are echoed by the CLI, and a Windows
    # console on cp437 raises UnicodeEncodeError on anything else.
    out = [f"{base} [{'+'.join(v)}]" if v else base
           for base, v in quant_variants(files)]
    if len(set(out)) == len(out):
        return out
    # Same stem in different subdirs, or names differing only past the tag.
    # Keep the HEAD and elide the middle: the previous version sliced the LAST
    # 24 characters, rendering "RVN-IQ3_M-multilingual-mtp" as
    # "N-IQ3_M-MULTILINGUAL-MTP". A label starting mid-word is unreadable in a
    # way a long one is not (owner, 2026-08-21).
    paths = [str(_P(f).with_suffix("")).replace(chr(92), "/").upper()
             for f in files]
    return [q if len(q) <= 30 else f"{q[:16]}...{q[-11:]}" for q in paths]


def moe_from_probe(f: dict, biggest_bytes: int) -> MoESpec | None:
    """MoE sizing, measured from the tensor table when the table was readable.

    The old figures were a rule of thumb over the file size — `size_gb * 2` for
    total parameters, a flat 10% of that for active, and a hardcoded 0.85 expert
    weight fraction. On the APEX 35B that produced 48.4B total (real: 35.5B, 36%
    over), 4.8B active (real: 3.5B — the model's own name says A3B) and 0.85
    against a measured 0.93. That fraction is not cosmetic: `_spilled` and the
    offload math multiply by it, so every --n-cpu-moe decision inherited the
    error. Sum the expert tensors instead; the estimate stays only as the
    fallback for a remote probe whose tensor table was cut off.
    """
    if f.get("kind") != "moe":
        return None
    params = int(f.get("params", 0) or 0)
    experts = int(f.get("expert_params", 0) or 0)
    used, total = int(f.get("expert_used", 0) or 0), int(f.get("experts", 0) or 0)
    if params > 0 and experts > 0 and total > 0:
        active = params - experts + experts * (used / total)
        return MoESpec(total_b=round(params / 1e9, 2),
                       active_b=round(max(active, 0.0) / 1e9, 2),
                       expert_weight_fraction=round(experts / params, 3))
    est_b = max(1.0, round(biggest_bytes / 2**30 * 2, 1))
    return MoESpec(total_b=est_b, active_b=max(0.5, round(est_b * 0.1, 1)),
                   expert_weight_fraction=0.85)


def spec_fields_from_probe(f: dict) -> dict:
    """The probed facts every ModelSpec carries, in one place so install,
    remote-add and healing cannot drift apart."""
    return {"n_layers": f["n_layers"],
            "full_attn_layers": f["full_attn_layers"],
            "kv_heads": f["kv_heads"], "head_dim": f["head_dim"],
            "native_ctx": max(2048, f["native_ctx"]),
            "params": int(f.get("params", 0) or 0),
            "mtp_layers": int(f.get("mtp_layers", 0) or 0),
            "full_attention_interval":
                int(f.get("full_attention_interval", 0) or 0),
            # AUDIT F19: docs/audit-2026-09-04-full.md — without these three the
            # windowed KV term stays zero for every model that can exist, and the
            # fit silently overcommits on a Gemma-style import.
            "swa_layers": int(f.get("swa_layers", 0) or 0),
            "swa_kv_heads": int(f.get("swa_kv_heads", 0) or 0),
            "swa_window": int(f.get("swa_window", 0) or 0),
            "has_template": bool(f.get("has_template", True)),
            "probe_version": PROBE_VERSION}


def template_override(slug: str) -> bool:
    """Is a repaired chat template installed for this model?

    Both `cli.up` and `server_ops.switch_model` pass templates/<slug>.jinja to
    llama-server with --chat-template-file when it exists. The Models page did
    not look, so a model whose gguf ships no template kept being described as
    "running on the engine's fallback format" long after a proper one had been
    dropped in for it — a warning that had become false (owner, 2026-08-19).
    """
    try:
        return (rigma_home() / "templates" / f"{slug}.jinja").is_file()
    except OSError:
        return False


def _write_spec(spec: ModelSpec) -> None:
    d = custom_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{spec.slug}.json.tmp"
    tmp.write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, d / f"{spec.slug}.json")


# Repetition control every model gets, on top of whatever sampling it declares.
#
# Qwen3.8's PUBLISHED thinking preset is temp 1.0 / top_p 0.95 / top_k 20 with
# presence_penalty 0.0 and repetition_penalty 1.0 — deliberately no repetition
# control at all (huggingface.co/Qwen/Qwen3.8-27B, and Unsloth's local-run
# guide agrees). That is a fine choice for short answers and a loaded gun for a
# long one. Live 2026-08-18: a dense 27B import wrote 81,544 characters in one
# write_file call, the same ~280-character stanza about 200 times.
#
# Qwen's own lever for this is presence_penalty (they suggest 0-2, and use 1.5
# in the instruct preset), but it penalises ANY repeated token — character
# names, place names — and they warn it causes language mixing. DRY penalises
# repeated SEQUENCES, which is precisely "stop reciting that stanza" without
# taxing ordinary prose, and it is what every curated model in this registry
# already uses.
#
# dry_penalty_last_n matters as much as the multiplier: the observed cycle was
# ~70 tokens against llama.cpp's default 64-token repeat window, so a
# token-level penalty could not have seen it even switched on.
#
# dry_allowed_length is the number that matters, and 2 (llama.cpp's default) is
# actively harmful here. It penalises ANY repeated run longer than two tokens,
# and a filename is 8-12: once Chapter_02_Shubhashini.txt is in context, the
# model is punished for re-typing it and emits a MUTATION instead. Live
# 2026-08-19 that produced Shubhash -> Shabhash -> Shaubha and .txt -> .ttf, and
# tool calling fell apart within a turn. Paths, character names and quoted lines
# are all repeated on purpose; only a LOOP is repetition.
#
# 16 leaves those alone and still catches the failure DRY is here for by a wide
# margin — that was a ~70-token stanza repeated ~200 times, where the penalty
# grows exponentially with match length past the allowance. RUN_PARAMS keeps 2
# because it also pins temperature to 0.3 ("the call syntax must be boring"); a
# writing chat cannot do that, so the allowance has to carry it alone.
_DRY_BASELINE = {"dry_multiplier": 0.8, "dry_base": 1.75,
                 "dry_allowed_length": 16.0, "dry_penalty_last_n": 4096.0}


def params_from_probe(f: dict | None = None) -> dict:
    """The model's own declared sampling, plus the repetition control its
    published preset leaves out."""
    out = dict((f or {}).get("sampling") or {})
    out.update(_DRY_BASELINE)
    return out


def inherit_family_defaults(spec: ModelSpec,
                            probe: dict | None = None) -> ModelSpec:
    """Custom imports inherit the registry sibling's card sampling and KV
    policy. An installed fine-tune of a registry model otherwise ran on raw
    llama-server defaults — no DRY, no model-card temperature, and no
    DeltaNet-aware q8_0 cache policy (audit 2026-07-21, limiting-setting #3).

    Matching is by ARCHITECTURAL FINGERPRINT (kind + layer geometry), not by
    family name: gguf arch strings ("qwen35moe") never equal registry family
    names ("qwen3.6"), but a fine-tune of the same base model shares its
    exact attention geometry.

    When nothing matches, the model does NOT fall through to nothing — it gets
    its own declared sampling plus _DRY_BASELINE. The empty case was the whole
    bug: an import matching no curated geometry ran with no repetition control
    whatsoever."""
    update: dict = {}
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
            if not spec.default_params and m.default_params:
                update["default_params"] = dict(m.default_params)
            if spec.cache_type_policy == blank and \
                    m.cache_type_policy != blank:
                update["cache_type_policy"] = \
                    m.cache_type_policy.model_copy()
            break
    except Exception:
        pass          # inheritance is a nicety — never block an install
    if not (update.get("default_params") or spec.default_params):
        update["default_params"] = params_from_probe(probe)
    return spec.model_copy(update=update) if update else spec


def heal_spec(spec: ModelSpec) -> ModelSpec:
    """Re-derive a stored spec's probed facts when the probe has since improved.

    Quant LABELS already heal on read (_distinct_quants). Geometry did not, so a
    model added before a fix kept the wrong numbers forever and the only cure was
    for the user to delete and re-add it — which they cannot be expected to know.
    The APEX 35B on this machine was stored as 41-of-41 full-attention layers;
    the file says 10-of-40, a 4x overstatement of its KV cache that had been
    quietly capping its context since it was added.

    Only re-probes when the gguf is actually on disk, and only for custom specs
    (registry entries are hand-authored and researched — never overwrite those).
    Capabilities are ADDED, never removed, except `mtp`, which the tensor table
    can positively disprove; a capability the user set by hand must survive.
    """
    if not spec.custom:
        return spec
    # Repair that needs NO file first: a spec with no sampling defaults runs on
    # llama-server's bare defaults, which carry no repetition control at all.
    # Both models the owner added from Hugging Face stored {}, and one has no
    # quant downloaded — so this must not wait on a probe.
    if not spec.default_params:
        spec = spec.model_copy(update={"default_params": params_from_probe()})
        try:
            _write_spec(spec)
        except OSError:
            pass
    mdir = models_dir()
    stale = spec.probe_version < PROBE_VERSION
    # a quant downloaded after the last heal still needs its own MTP answer
    unprobed = [g for g in spec.ggufs
                if g.mtp is None and (mdir / g.file).is_file()]
    if not stale and not unprobed:
        return spec                        # nothing to do: the common path
    probed = {}
    targets = {g.file for g in unprobed}
    if stale:
        local = next((g for g in spec.ggufs if (mdir / g.file).is_file()), None)
        if local is not None:
            targets.add(local.file)
    for name in targets:
        try:
            probed[name] = inspect_gguf(mdir / name)
        except (GgufParseError, OSError, ValueError):
            continue               # unreadable file must not break the library
    if not probed:
        return spec
    fields = {name: got.spec_fields for name, got in probed.items()}
    info = next((probed[g.file] for g in spec.ggufs if g.file in probed), None)
    f = info.spec_fields if info else None
    if stale and info and f and f.get("n_layers", 0) > 0 \
            and f.get("kv_heads", 0) > 0:
        healed = _with_probe(spec, info, fields)     # geometry + caps + files
    else:
        healed = spec.model_copy(update={"ggufs": [   # just the per-file answer
            g.model_copy(update={"mtp": fields[g.file].get("mtp")})
            if g.file in fields else g for g in spec.ggufs]})
    if healed == spec:
        return spec
    try:
        _write_spec(healed)        # persist so the next read is free
    except OSError:
        pass                       # a read-only home still gets healed in memory
    return healed


def reprobe(slug: str, *, allow_remote: bool = True,
            refresh_files: bool = True) -> ModelSpec:
    """Re-derive a spec's probed facts on demand, reading the repo if needed.

    `heal_spec` runs on every registry load and so must never touch the network;
    it can only work from a gguf already on disk. That leaves a model added but
    not yet downloaded stuck with whatever an older probe wrote — Qwen3.8-27B
    kept its 65-layer geometry (the MTP block counted as a decoder layer) with
    no way to correct it short of removing and re-adding the repo. This is the
    explicit version: the user asks, so the ranged header read is theirs to
    spend. Local file first, because it is free and it is the better evidence.
    """
    spec = _load_custom(slug)
    if spec is None:
        raise HangarError(f"{slug} is not a custom model — registry specs are "
                          "hand-authored and are not re-probed")
    healed = heal_spec(spec.model_copy(update={"probe_version": 0}))
    remote = next((g for g in spec.ggufs if g.repo and g.repo != "local"), None)
    # The INVENTORY is a separate question from the GEOMETRY, and refreshing it
    # must not be gated on the geometry being stale: the owner's model had
    # current geometry and a file list 76 ggufs out of date, so the freshness
    # check below returned before anything looked at the repo (2026-08-21).
    if refresh_files and allow_remote and remote is not None:
        from .hf_browse import repo_files
        merged = merge_repo_files(healed, repo_files(remote.repo))
        # Compare labels too, not just filenames: `_distinct_quants` derives a
        # label from the whole set, so a refresh that adds no file can still
        # legitimately rename rows — and a write gated on filenames alone left
        # 103 stale labels on disk (2026-08-21).
        if [(g.file, g.quant) for g in merged.ggufs]                 != [(g.file, g.quant) for g in healed.ggufs]                 or merged.mmproj != healed.mmproj:
            healed = merged
            _write_spec(healed)
    if healed.probe_version >= PROBE_VERSION:
        return healed
    if not allow_remote:
        raise HangarError(f"nothing of {slug} is downloaded, so there is no "
                          "file to read")
    if remote is None:
        raise HangarError(f"{slug} was installed from a local file that is no "
                          "longer on disk — nothing left to read")
    from .hf_browse import remote_inspect
    info = remote_inspect(remote.repo, remote.file)
    f = info.spec_fields
    if not f or f.get("n_layers", 0) <= 0 or f.get("kv_heads", 0) <= 0:
        raise HangarError(f"{remote.file} carries no usable model metadata")
    fields = {remote.file: f}
    # Every other file's MTP answer, while we are already talking to the repo.
    # Only ones we have never read: a second reprobe of the same library costs
    # nothing. One unreadable file must not cost the answers for the rest.
    others = [g for g in healed.ggufs
              if g.file != remote.file and g.mtp is None
              and g.repo and g.repo != "local"]
    if others and len(healed.ggufs) <= MTP_PROBE_LIMIT:
        for g in others:
            try:
                fields[g.file] = remote_inspect(g.repo, g.file).spec_fields
            except (HangarError, OSError, ValueError):
                continue
    updated = _with_probe(healed, info, fields)
    _write_spec(updated)
    return updated


def merge_repo_files(spec: ModelSpec, rf: dict, *,
                     on_disk: set[str] | None = None) -> ModelSpec:
    """Fold a fresh repo listing into a spec's gguf list.

    `reprobe` re-reads a model's GEOMETRY; this re-reads its INVENTORY. They are
    different questions and a repo answers the second one long after you added
    it: 0bserverx/Qwen3.8-27B-Heretic went from 27 ggufs to 103 by publishing
    -mtp and -multilingual builds, and none of them were reachable, because
    `add_model` refuses a repo already in the library (2026-08-21).

    Four rules, each one a way this could quietly lose something:

    * a gguf ON DISK is never dropped, even if the repo deleted it — forgetting
      it orphans gigabytes and cuts the running server loose from its own spec;
    * one that was never downloaded IS dropped, so the list tracks the repo;
    * `mtp` flags already probed are carried over. Each cost a ranged header
      read, and re-reading 103 of them to learn what we know is minutes of
      network for nothing. A NEW file stays None — unknown, never assumed,
      because llama.cpp resets the Vulkan driver if asked for draft-mtp on a
      file without the tensors;
    * labels are re-derived over the WHOLE new set, since `_distinct_quants`
      names a file by what distinguishes it from its siblings — adding a file
      can legitimately rename one that was already there.
    """
    if on_disk is None:
        # AUDIT F28: docs/audit-2026-09-04-full.md
        # `repo_files` lists the tree recursively because many repos nest
        # quants in subdirs, so a spec's `file` can be "Q4_K_M/model.gguf".
        # A name-only glob never matched those, so every nested quant read as
        # "not downloaded" and the rule above dropped it from the spec.
        # Both spellings on purpose. The relative path is what a nested spec
        # entry looks like; the bare name is what a FLAT entry looks like whose
        # file has since been stored in a subdirectory. Matching only the first
        # would read that file as absent and the rule above would drop it —
        # trading the old bug for the exact loss this function exists to
        # prevent. "On disk is never dropped" wins over precise attribution.
        mdir = models_dir()
        found = list(mdir.rglob("*.gguf"))
        on_disk = ({p.relative_to(mdir).as_posix() for p in found}
                   | {p.name for p in found})
    repo = next((g.repo for g in spec.ggufs if g.repo and g.repo != "local"), "")
    known = {g.file: g for g in spec.ggufs}
    listed = [g["file"] for g in (rf.get("ggufs") or [])]
    sizes = {g["file"]: g["bytes"] for g in (rf.get("ggufs") or [])}
    kept = [f for f in known
            if f not in sizes and (f in on_disk or known[f].repo == "local")]
    files = listed + kept
    ggufs = []
    for fname, label in zip(files, _distinct_quants(files)):
        prev = known.get(fname)
        if prev is not None:
            ggufs.append(prev.model_copy(update={
                "quant": label, "bytes": sizes.get(fname, prev.bytes)}))
        else:
            ggufs.append(GgufFile(repo=repo, file=fname,
                                  bytes=sizes.get(fname, 0), quant=label))
    update: dict = {"ggufs": ggufs}
    mm = rf.get("mmproj")
    if mm and (spec.mmproj is None or spec.mmproj.file != mm["file"]):
        update["mmproj"] = GgufFile(repo=repo, file=mm["file"],
                                    bytes=mm["bytes"],
                                    quant=_quant_from_name(mm["file"]))
        update["capabilities"] = sorted(set(spec.capabilities) | {"vision"})
    return spec.model_copy(update=update)


def _with_probe(spec: ModelSpec, info, probed: dict) -> ModelSpec:
    """Fold a fresh probe into a spec. Capabilities are ADDED, never removed,
    except `mtp`, which a read tensor table can positively disprove — a
    capability the user set by hand has to survive a re-probe."""
    f = info.spec_fields
    caps = set(spec.capabilities) | set(info.capabilities)
    if f.get("mtp") is False:
        caps.discard("mtp")
    update = dict(spec_fields_from_probe(f))
    update["capabilities"] = sorted(caps)
    update["ggufs"] = [g.model_copy(update={"mtp": probed[g.file].get("mtp")})
                       if g.file in probed else g for g in spec.ggufs]
    if spec.kind == "moe":
        moe = moe_from_probe(f, max((g.bytes for g in spec.ggufs), default=0))
        if moe is not None:
            update["moe"] = moe
    return spec.model_copy(update=update)


def rename_model(slug: str, new_slug: str) -> ModelSpec:
    """Rename a custom model, carrying everything else keyed by its slug.

    A slug is not just a label: the repaired chat template lives at
    templates/<slug>.jinja and calibration rows are keyed "<slug>:<quant>:
    <backend>". Renaming by hand — the only option before this — silently
    orphaned both, so a model that had been tuned and given a working template
    came back untuned and on its embedded template.
    """
    from . import state as st
    from .registry import Registry
    spec = _load_custom(slug)
    if spec is None:
        raise HangarError("only custom models can be renamed")
    new_slug = _slugify(new_slug)
    if not new_slug:
        raise HangarError("that name has no usable characters in it")
    if new_slug == slug:
        return spec
    if new_slug in Registry.load().models:
        raise HangarError(f"a model named {new_slug} already exists")
    state = st.read_state()
    if state and state.get("model") == slug:
        raise HangarError(f"{slug} is running — stop or switch models first")
    home = rigma_home()
    renamed = spec.model_copy(update={"slug": new_slug})
    _write_spec(renamed)
    tmpl = home / "templates" / f"{slug}.jinja"
    if tmpl.is_file():
        os.replace(tmpl, home / "templates" / f"{new_slug}.jinja")
    calib = home / "calibration.json"
    if calib.is_file():
        try:
            import json
            rows = json.loads(calib.read_text(encoding="utf-8"))
            moved = {(f"{new_slug}:{k.split(':', 1)[1]}"
                      if k.startswith(f"{slug}:") else k): v
                     for k, v in rows.items()}
            if moved != rows:
                tmp = calib.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(moved, indent=1), encoding="utf-8")
                os.replace(tmp, calib)
        except (OSError, ValueError):
            pass          # a lost calibration row costs one re-tune, not data
    (custom_dir() / f"{slug}.json").unlink(missing_ok=True)
    return renamed


def file_has_mtp(gguf: GgufFile) -> bool | None:
    """Does THIS quant carry the draft head? True/False/None (can't tell yet).

    Reads the file when it is on disk and the spec has not recorded an answer —
    which is exactly the situation at launch, the one moment the question has to
    be right. Asking llama.cpp for draft-mtp against a file without the tensors
    is a Vulkan driver reset, so "we never checked" must not read as yes.
    """
    if gguf.mtp is not None:
        return gguf.mtp
    path = models_dir() / gguf.file
    if not path.is_file():
        return None
    try:
        return inspect_gguf(path).spec_fields.get("mtp")
    except (GgufParseError, OSError, ValueError):
        return None


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
    slug = model_slug(info.name, src.stem)
    if slug in Registry.load().models:
        raise HangarError(f"a model named {slug} already exists")
    f = info.spec_fields
    if f["n_layers"] <= 0 or f["kv_heads"] <= 0 or f["head_dim"] <= 0:
        raise HangarError(
            "gguf header is missing attention metadata — Rigma can't "
            "compute memory fit for this file")
    size = src.stat().st_size
    moe = moe_from_probe(f, size)
    dest = models_dir() / src.name
    if dest.exists():
        # same filename may back a DIFFERENT model (generic quantizer names) —
        # overwriting could destroy the running engine's weights
        raise HangarError(f"{dest.name} already exists in Rigma's models "
                          "folder — rename the file and try again")
    spec = ModelSpec(
        slug=slug, family=info.arch or "custom", kind=f["kind"],
        ggufs=[GgufFile(repo="local", file=dest.name, bytes=size,
                        quant=_quant_from_name(dest.name), mtp=f.get("mtp"))],
        moe=moe, license="custom import", use_cases=["general"],
        capabilities=sorted(info.capabilities), custom=True,
        **spec_fields_from_probe(f))
    spec = inherit_family_defaults(spec, f)
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
    # The FILE is the identity. A quant LABEL is derived from the whole file
    # list, so publishing siblings can rename a row that did not change — and
    # matching on it un-marked the model the engine was actually holding
    # (2026-08-21). State written before this carries no file: fall back.
    want = state.get("gguf") or ""
    out = {g.file for g in spec.ggufs if g.file == want} if want else set()
    if not out:
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
    held = _running_files(state, reg)
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
        # The grid behind the list: 103 files that are really 26 quants crossed
        # with -multilingual/-mtp/-vision. The UI groups on `base` and filters
        # on `variants`, so the card is 26 rows and three chips instead of a
        # 103-row wall (owner, 2026-08-21).
        axes = quant_variants([g.file for g in spec.ggufs])
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
        from .quant_quality import (label_overstates, measured_bpw, quality_of,
                                    total_loss)
        quants = []
        for g, label, (base, variants), fit in zip(spec.ggufs, labels,
                                                   axes, fits):
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
                           # per FILE, not per model: whether this artefact
                           # carries the draft head. None = not read yet.
                           "mtp": g.mtp,
                           # measured, so a repo that names its files
                           # I-Compact still says what it spends
                           "bpw": measured_bpw(g.bytes, spec.params),
                           "label_drift": label_overstates(label, g.bytes,
                                                           spec.params),
                           # WHICH quant is loaded, not just which model —
                           # with two on disk the card could not tell them
                           # apart. Keyed on the FILE: labels are derived from
                           # the whole set and move when the repo publishes
                           # siblings, which silently un-marked the running
                           # row (2026-08-21). `held` covers the fallback for
                           # state written before the file was recorded.
                           "running": g.file in held,
                           # what this row IS, split from what makes it a
                           # variant of its siblings
                           "base": base,
                           "variants": variants,
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
            "params": spec.params,
            "n_layers": spec.n_layers,
            "full_attn_layers": spec.full_attn_layers,
            "mtp_layers": spec.mtp_layers,
            # False => the capability list above is missing evidence, not a
            # finding. The UI must say so rather than render an empty row.
            "has_template": spec.has_template,
            # a repaired template standing in for a missing one
            "template_override": template_override(slug),
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
    _discard_partial(target)   # AUDIT F27: docs/audit-2026-09-04-full.md


def delete_model(slug: str, registry=None) -> None:
    from . import state as st
    spec = _load_custom(slug)
    if spec is None:
        raise HangarError("only custom models can be removed — registry "
                          "models just have their files deleted")
    state = st.read_state()
    if state and state.get("model") == slug:
        raise HangarError(f"{slug} is running — stop or switch models first")
    # AUDIT F27: docs/audit-2026-09-04-full.md
    # The resume files go with the model. Nothing else reaped them, so a
    # cancelled multi-GB pull outlived the model it belonged to, invisible in
    # the library because glob("*.gguf") does not match ".part".
    for g in spec.ggufs:
        path = models_dir() / g.file
        path.unlink(missing_ok=True)
        _discard_partial(path)
    if spec.mmproj is not None:
        path = models_dir() / spec.mmproj.file
        path.unlink(missing_ok=True)
        _discard_partial(path)
    (custom_dir() / f"{slug}.json").unlink(missing_ok=True)


def set_launch_defaults(slug: str, registry=None, **fields) -> ModelSpec:
    """Store how this model should come up. Pass only what you mean to pin.

    Passing None for a field CLEARS it back to "no opinion", which is how the
    UI removes a default without needing a separate verb.
    """
    from .models import LaunchDefaults
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    spec = reg.models.get(slug)
    if spec is None:
        raise HangarError(f"unknown model: {slug}")
    if not spec.custom:
        raise HangarError(f"{slug} is a registry model — its launch settings "
                          "are hand-authored and are not overwritten here")
    current = (spec.launch or LaunchDefaults()).model_dump()
    allowed = set(LaunchDefaults.model_fields)
    unknown = set(fields) - allowed
    if unknown:
        raise HangarError(f"unknown launch setting(s): {', '.join(sorted(unknown))}")
    for key, value in fields.items():
        current[key] = value
    launch = LaunchDefaults(**current)
    # Everything cleared means no opinion at all; store None rather than an
    # empty object, so a spec that pins nothing reads as pinning nothing.
    updated = spec.model_copy(update={
        "launch": launch if launch.as_overrides() else None})
    _write_spec(updated)
    return updated


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
            # AUDIT F27: docs/audit-2026-09-04-full.md
            # What the registry says this file IS. gguf.bytes was in scope all
            # along and drove nothing but the progress bar; gguf.sha256 was
            # declared in models.py and read nowhere at all.
            _download_file(repo, file, models_dir() / file,
                           lambda n: _PULLS[key].update(done=n),
                           expect_bytes=want, sha256=gguf.sha256)
            _PULLS[key]["status"] = "done"
        except Exception as e:   # surfaced via /api/models, not lost in a thread
            _PULLS[key].update(status="error", error=str(e).splitlines()[0])

    threading.Thread(target=_run, daemon=True, name=f"pull:{file}").start()
    return _PULLS[key]


# AUDIT F27: docs/audit-2026-09-04-full.md
# A resume file is keyed on the DESTINATION name, which is not an identity.
# `mmproj-F16.gguf` ships in the registry twice over and quantiser filenames
# collide routinely, so a cancelled pull of one repo left a .part that the
# next repo's same-named file resumed onto: the new object's tail appended to
# the old object's head, landing on EXACTLY the expected byte count. The note
# beside the .part records which (repo, file) wrote those bytes; a partial
# that cannot prove what it is gets dropped rather than resumed. Partials
# written before this existed have no note and are therefore discarded once,
# which costs one restarted download and no wrong installs.
_PART_SUFFIX = ".part"
_PART_NOTE_SUFFIX = ".part.id"


def _resume_files(dest) -> tuple[Path, Path]:
    """The .part beside `dest` and the note saying which file wrote it."""
    return (dest.with_name(dest.name + _PART_SUFFIX),
            dest.with_name(dest.name + _PART_NOTE_SUFFIX))


def _part_ident(repo: str, file: str) -> str:
    return f"{repo}\n{file}"


def _claim_partial(dest, repo: str, file: str) -> None:
    """Record which (repo, file) the .part beside `dest` is accumulating."""
    _resume_files(dest)[1].write_text(_part_ident(repo, file), encoding="utf-8")


def _part_owner(dest) -> str:
    """Who wrote the .part beside `dest`, or "" when it cannot say."""
    try:
        return _resume_files(dest)[1].read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _discard_partial(dest) -> None:
    """Throw away a resume file and its note. Best effort on purpose: a stale
    partial we cannot remove must not sink the delete that found it."""
    for p in _resume_files(dest):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _object_size(status: int, headers) -> int:
    """The WHOLE object's length as the server states it, 0 if it does not.

    On a 206 or 416 that is content-range's total — content-length there
    describes the slice, not the file — and on a 200 it is content-length."""
    if status in (206, 416):
        m = re.search(r"/\s*(\d+)\s*$", headers.get("content-range") or "")
        return int(m.group(1)) if m else 0
    try:
        return int(headers.get("content-length") or 0)
    except (TypeError, ValueError):
        return 0


def _sha256_of(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _ShortBody(Exception):
    """A body that ended cleanly but early — a transport drop by another name.
    Raised so the retry loop resumes instead of installing a truncated file."""


def _install_download(dest, have: int, size: int, sha256: str | None) -> None:
    """Check a finished .part against what it should be, then move it in.

    Size cannot be the only gate: the splice this guards against lands on the
    exact expected length. sha256 is the check that proves identity, and it
    runs at 4.0 GB/s here (SHA-NI), so an 11GB quant verifies in about three
    seconds against a download measured in minutes — cheap enough to always do
    when the registry carries a hash. It usually does not, which is why the
    resume note above has to stand on its own."""
    part, note = _resume_files(dest)
    if size and have != size:
        if have > size:
            _discard_partial(dest)
            raise HangarError(
                f"{dest.name} came back {have:,} bytes where {size:,} were "
                f"expected — those are not this file's bytes, so they were "
                f"discarded rather than installed. Re-probe the model to "
                f"refresh its file list, then press Download again.")
        raise _ShortBody(f"body ended at {have:,} of {size:,} bytes")
    if sha256:
        got = _sha256_of(part)
        if got.lower() != sha256.lower():
            _discard_partial(dest)
            raise HangarError(
                f"{dest.name} failed its sha256 check (got {got[:12]}, "
                f"expected {sha256[:12].lower()}) — discarded rather than "
                f"installed. Press Download again to refetch it.")
    os.replace(part, dest)
    note.unlink(missing_ok=True)


def _download_file(repo: str, file: str, dest, report, *,
                   expect_bytes: int = 0, sha256: str | None = None) -> int:
    """Stream a HF file straight to `dest` with resume + live byte reporting.

    Direct httpx (not hf_hub_download) on purpose: we get the exact byte count
    for a real progress bar, resume works off a plain `.part` file, and it
    never touches the xet backend that hangs on this box.

    `expect_bytes`/`sha256` are the registry's record of what this file is
    (GgufFile.bytes and .sha256). They are the fallback: what the server itself
    declares wins, because a repo that re-uploads a file leaves the registry
    stale and a stale number must not make a good file undownloadable."""
    import httpx
    if dest.exists():
        report(dest.stat().st_size)
        return dest.stat().st_size
    import time
    tok = os.environ.get("HF_TOKEN", "")
    part, _note = _resume_files(dest)
    try:
        # AUDIT F28: docs/audit-2026-09-04-full.md
        # `repo_files` lists the tree recursively because many repos nest
        # quants in subdirs, so `file` can be "Q4_K_M/model.gguf". Without
        # this the open below raised FileNotFoundError INSIDE the retry loop,
        # which spent 31 seconds "resuming" and then blamed the connection for
        # a directory nobody had created. `rigma up` on the same model worked,
        # because hf_hub_download makes the subdirectory.
        dest.parent.mkdir(parents=True, exist_ok=True)
        # AUDIT F27: bytes that cannot prove they came from this (repo, file)
        # are not a resume point.
        if part.exists() and _part_owner(dest) != _part_ident(repo, file):
            _discard_partial(dest)
    except OSError as e:
        raise HangarError(f"cannot write into {dest.parent}: {e}") from e
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
                size = _object_size(r.status_code, r.headers) or expect_bytes
                if r.status_code == 416:
                    # Range past EOF. Renaming the partial in as "complete" is
                    # right only when it is exactly the object's length, which
                    # the server states in content-range even on a 416. Any
                    # other size means the .part outlived some other transfer,
                    # so drop it and re-issue with no Range header at all.
                    if size and have == size:
                        _install_download(dest, have, size, sha256)
                        report(have)
                        return have
                    _discard_partial(dest)
                    continue
                if r.status_code in (401, 403):
                    raise HangarError("that repo is gated — accept its license "
                                      "on huggingface.co and set HF_TOKEN")
                resumed = r.status_code == 206
                if r.status_code not in (200, 206):
                    raise HangarError(
                        f"Hugging Face returned HTTP {r.status_code}")
                if not resumed:
                    have = 0               # server ignored Range — start over
                if not (resumed and have):
                    _claim_partial(dest, repo, file)     # a fresh .part is ours
                with open(part, "ab" if resumed and have else "wb") as f:
                    report(have)
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        report(have)
            _install_download(dest, have, size, sha256)
            report(have)
            return have
        except HangarError:
            raise                          # gated/HTTP/identity errors terminal
        except OSError as e:
            # AUDIT F28: not a transport drop. httpx wraps network failures in
            # its own exception tree, so an OSError here came from the local
            # file — and retrying one six times is what reported "download
            # kept dropping (0 bytes saved)" for a directory that never
            # existed, over 31 seconds, forever.
            raise HangarError(f"could not write {part.name}: {e}") from e
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
