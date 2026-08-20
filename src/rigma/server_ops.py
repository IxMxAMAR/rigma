from __future__ import annotations

import json
import os
import platform

import psutil

from .runtime import rigma_home


def ram_snapshot() -> dict:
    m = psutil.virtual_memory()
    return {"ram_free_mb": int(m.available / 2**20),
            "ram_total_mb": int(m.total / 2**20)}


def engine_version() -> str:
    try:
        from .runtime import _engines_manifest
        return str(_engines_manifest().get("version", ""))
    except Exception:
        return ""


def expected_tg(model: str, quant: str, backend: str) -> float | None:
    """Calibrated decode speed for the running combo, if bench ever ran.

    The number lives under "measured" — that is the shape `bench.save_calibration`
    writes, and it is the only writer of this file. This read used a flat
    `["tg_tps"]` that no writer has ever produced, so it returned None for every
    real calibration on disk and the engine-room verdict was permanently
    "unknown". The unit test passed throughout, because it hand-wrote the flat
    shape instead of calling the writer.

    The flat branch is kept for entries written before the nesting existed;
    without it, fixing the read would silently retire every old calibration.
    """
    try:
        cal = json.loads((rigma_home() / "calibration.json")
                         .read_text(encoding="utf-8"))
        entry = cal[f"{model}:{quant}:{backend}"]
        got = entry.get("measured", entry).get("tg_tps")
        if got is None:
            return None
        # Speed on a fixed model+quant+engine is not a constant: it depends on
        # whether the weights fit in VRAM, and on Windows that depends on what
        # ELSE holds VRAM. Measured 2026-08-21, the same combo ran at 9.95 and
        # 37.59 tok/s purely because the desktop's footprint changed by 2.8GB.
        # An expectation carried across that is worse than no expectation.
        from .bench import calibration_stale
        from .probe import gpu_used_mb
        if calibration_stale(entry, gpu_used_mb()) is not None:
            return None
        return float(got)
    except Exception:
        return None


def verdict(last_tg: float | None, exp: float | None) -> str:
    if last_tg is None or exp is None:
        return "unknown"
    return "degraded" if last_tg < 0.6 * exp else "healthy"


def log_tail(lines: int = 200) -> str:
    logs = sorted((rigma_home() / "logs").glob("server-*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return ""
    lines = max(10, min(int(lines), 1000))
    text = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _model_on_disk(gguf) -> bool:
    return (rigma_home() / "models" / gguf.file).exists()


def _calib_marker_path():
    return rigma_home() / "calibrating.json"


def read_calib_marker() -> dict | None:
    """First-load tuning progress, if a calibration is running right now. Stale
    markers (a crash mid-tune) are ignored after 20 min so the UI never sticks
    on 'optimizing' forever."""
    p = _calib_marker_path()
    try:
        import time
        if time.time() - p.stat().st_mtime > 1200:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_calib_marker(model: str, step: str) -> None:
    p = _calib_marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"model": model, "step": step}), encoding="utf-8")
    tmp.replace(p)


def _clear_calib_marker() -> None:
    try:
        _calib_marker_path().unlink()
    except FileNotFoundError:
        pass


def _free_current(profile, state: dict, reg):
    """A copy of `profile` with the CURRENTLY-running model's RAM footprint
    added back — perform_switch/ctx-change kill that engine before launching,
    so fitting against live free RAM is wrong. Without this, a ctx change on
    the running model probes against its own occupied RAM and reports an
    absurdly small ceiling (live repro 2026-07-18: 35B ctx change said 'tops
    out at 8,192' while it was running fine at 32K).

    VRAM is credited the same way now that the budget accounts for what the
    desktop actually holds (2026-08-21). Without it the outgoing engine's own
    13GB would count as "someone else's", and a ctx change on the running model
    would plan against a card that looks entirely full."""
    if not state:
        return profile
    spec = reg.models.get(state.get("model", ""))
    if spec is None or not spec.ggufs:
        return profile
    # The file the engine ACTUALLY holds. `state["gguf"]` records it since
    # 2026-08-21; the largest-gguf guess below is the fallback for state written
    # before that. The guess was harmless while a model listed a handful of
    # files and became dangerous the moment one listed 103: this model's repo
    # includes a 50GB BF16, so crediting the largest handed back three times the
    # card's capacity, made the GPU look empty, and put the planner straight
    # back to assuming VRAM that does not exist.
    running = next((g for g in spec.ggufs if g.file == state.get("gguf")), None)
    freed_mb = ((running.bytes if running is not None
                 else max(g.bytes for g in spec.ggufs)) / 2**20)
    if spec.mmproj and not state.get("no_vision"):
        freed_mb += spec.mmproj.bytes / 2**20
    update = {"ram_free_mb": profile.ram_free_mb + int(freed_mb)}
    # The engine we are about to kill still holds its VRAM. Credit it back, or
    # the measured "desktop" footprint includes our own model and the plan
    # shrinks to fit a card that is about to be freed. Never below zero, and
    # never below what an unloaded desktop would read.
    if profile.vram_used_mb is not None and not state.get("unloaded"):
        # VRAM must never be over-credited the way RAM safely can: the whole
        # point of measuring it is to stop planning against memory the card
        # does not have. Cap the credit at what is actually held.
        update["vram_used_mb"] = max(0.0, profile.vram_used_mb
                                     - min(freed_mb, profile.vram_used_mb))
    return profile.model_copy(update=update)


def _resolve_for(slug: str, state: dict, registry, profile,
                 backend: str | None = None):
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import resolve
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
    p = _free_current(p, state, reg)   # count the outgoing engine as freed
    return resolve(p, reg, use_case=state.get("use_case", "general"),
                   model_override=slug, backend_override=backend), reg, p


def switch_options(state: dict, registry=None, profile=None) -> list[dict]:
    """Alternative plans limited to models already on disk (no downloads).

    Resolves each model against ONLY its on-disk quants — the resolver may
    prefer a quant that was never downloaded (e.g. under RAM pressure), and
    filtering on that preference hid genuinely usable local models."""
    from .probe import probe_hardware
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
    out = []
    for slug in sorted(reg.models):
        if slug == state.get("model"):
            continue
        spec = reg.models[slug]
        on_disk = [g for g in spec.ggufs if _model_on_disk(g)]
        if not on_disk:
            continue
        trimmed = Registry(reg.gpus,
                           {**reg.models,
                            slug: spec.model_copy(update={"ggufs": on_disk})},
                           reg.combos, reg.use_cases)
        try:
            rp, _, _ = _resolve_for(slug, state, trimmed, p)
        except Exception:
            continue
        if rp.model_slug != slug or not _model_on_disk(rp.gguf):
            continue
        reason = (f"{max(1, rp.flags.ctx // 1024)}K context, "
                  f"{rp.gguf.quant} on disk")
        if rp.backend == "cpu":
            reason += " — CPU fallback, will be slow (free RAM for GPU)"
        out.append({"model": rp.model_slug, "quant": rp.gguf.quant,
                    "ctx": rp.flags.ctx, "backend": rp.backend,
                    "reason": reason})
    out.sort(key=lambda o: -o["ctx"])
    return out


KV_CACHE_TYPES = ("f16", "q8_0", "q5_1", "q4_0")


def vram_snapshot(registry=None) -> dict | None:
    """Card capacity, what the desktop holds, and what is left for a model.

    Exists because the alternative is invisible. llama.cpp reports every layer
    as "on GPU" whenever the driver accepted the allocation, and on Windows the
    driver accepts allocations it intends to page — so a model can be 29% in
    system RAM with nothing anywhere saying so (measured 2026-08-21).
    """
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import COMPUTE_BUFFER_MB, VRAM_RESERVE_MB, _budgets
    reg = registry if registry is not None else Registry.load()
    prof = probe_hardware(reg.gpus)
    total = sum(g.vram_mb for g in prof.gpus)
    if not total:
        return None
    usable, _ = _budgets(prof)
    floor = VRAM_RESERVE_MB[prof.os]
    desktop = prof.vram_used_mb
    return {
        "total_mb": total,
        # None when unmeasurable — the UI must not render "0 MB held" then
        "desktop_mb": None if desktop is None else round(desktop),
        "usable_mb": round(usable),
        "assumed_mb": floor + COMPUTE_BUFFER_MB,
        # only worth telling the user about when it is worse than assumed
        "pressured": desktop is not None and desktop > floor,
    }


def available_backends(registry=None) -> list[dict]:
    """Backends this GPU can run, with whether the engine build is on disk.

    The gpu table has always listed more than one for AMD and NVIDIA cards —
    the resolver simply took the first. Surfacing the rest is what makes the
    choice real, and `ready` is what stops the UI offering ROCm without saying
    it costs a ~1.2GB download first.
    """
    import platform

    from .probe import probe_hardware
    from .registry import Registry
    from .runtime import _engines_manifest, rigma_home
    reg = registry if registry is not None else Registry.load()
    gpu = probe_hardware(reg.gpus).primary_gpu
    names = list(gpu.backends) if gpu and gpu.backends else []
    if "cpu" not in names:
        names.append("cpu")          # always available, always last resort
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    try:
        man = _engines_manifest()
    except Exception:
        return [{"name": n, "ready": False, "buildable": False} for n in names]
    root = rigma_home() / "engines" / man["version"]
    return [{"name": n,
             # a build we have no pinned asset for cannot be offered at all
             "buildable": f"{os_name}/{n}" in man.get("assets", {}),
             "ready": (root / n / ".ready").exists()}
            for n in names]


def perform_switch(model: str, registry=None, profile=None,
                   ctx: int | None = None, force_calibrate: bool = False,
                   kv: str | None = None, vision: bool | None = None,
                   quant: str | None = None,
                   backend: str | None = None) -> dict:
    """Stop the running engine and launch `model` in its place; with `ctx`,
    relaunch (same model allowed) at a requested context size; with `kv`,
    force the KV-cache quantisation (f16/q8_0/q5_1/q4_0). Growing the cache
    (q8_0 -> f16) may not fit — the launch failure path reports it honestly
    and leaves the UI manageable.

    `vision=False` runs a vision model text-only. The projector is loaded
    alongside the weights and cannot be offloaded, so it costs VRAM for the
    whole session whether or not an image is ever sent — 888MB on
    Qwen3.8-27B, which on a 16GB card is 4x the context window. Choosing to
    drop it is the single largest context lever such a model has.

    Raises RuntimeError with a user-facing message on any failure; a failure
    after the old engine died clears state (one requested plan, one honest
    result — the fallback ladder stays a CLI behavior)."""
    from . import runtime
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if kv is not None and kv not in KV_CACHE_TYPES:
        raise RuntimeError(f"kv must be one of {', '.join(KV_CACHE_TYPES)}")
    # `quant` joins ctx/kv as a reason to relaunch the SAME model: swapping
    # between two downloaded quants is the whole point of asking for one.
    # `backend` joins ctx/kv/quant as a reason to relaunch the SAME model —
    # switching Vulkan<->ROCm is exactly a same-model relaunch.
    if (model == s.get("model") and not s.get("unloaded") and ctx is None
            and kv is None and quant is None and backend is None
            and not force_calibrate):
        raise RuntimeError(f"{model} is already running")
    from .registry import Registry
    reg_full = registry if registry is not None else Registry.load()
    spec_full = reg_full.models.get(model)
    if spec_full is None:
        raise RuntimeError(f"unknown model: {model} — run: rigma update")
    on_disk = [g for g in spec_full.ggufs if _model_on_disk(g)]
    if not on_disk:
        raise RuntimeError(
            f"{model} is not downloaded — run: rigma up --model {model}")
    # resolve against on-disk quants only — the resolver may prefer a quant
    # that was never downloaded (live repro 2026-07-17: switch-back refused
    # while a perfectly usable quant sat on disk)
    # An explicit quant narrows the choice to exactly one file. Without this the
    # resolver picked, so a user with two quants on disk could not ask for the
    # smaller one — the trade you make when you want more context (owner,
    # 2026-08-19). Matched leniently: the label shown in the UI carries the
    # quanter's decoration, e.g. "Q3_K_M (RVN)".
    if quant:
        want = str(quant).strip().lower()
        picked = [g for g in on_disk if g.quant.lower() == want]
        if not picked:
            picked = [g for g in on_disk if want in g.quant.lower()
                      or g.quant.lower() in want]
        if not picked:
            have = ", ".join(sorted(g.quant for g in on_disk)) or "none"
            raise RuntimeError(
                f"{model} has no downloaded quant matching {quant!r} — "
                f"on disk: {have}")
        on_disk = picked[:1]
    trimmed = Registry(reg_full.gpus,
                       {**reg_full.models,
                        model: spec_full.model_copy(update={"ggufs": on_disk})},
                       reg_full.combos, reg_full.use_cases)
    rp, _, p = _resolve_for(model, s, trimmed, profile, backend)
    if rp.model_slug != model or not _model_on_disk(rp.gguf):
        raise RuntimeError(f"{model} does not fit this machine right now")
    if ctx is not None:
        # honest relaunch at a requested context: real fit math, not hope.
        # rp.flags.ctx is the calculator's grow-to-fit maximum for this quant.
        from .resolve import fit_gguf
        want = max(2048, min(int(ctx), spec_full.native_ctx))
        flags = fit_gguf(spec_full, rp.gguf, p, want, [])
        if flags is None:
            raise RuntimeError(
                f"ctx {want:,} doesn't fit — {model} ({rp.gguf.quant}) tops "
                f"out around {rp.flags.ctx:,} on this machine")
        rp.flags = rp.flags.model_copy(update={
            "ctx": flags.ctx, "n_cpu_moe": flags.n_cpu_moe, "ngl": flags.ngl,
            "cache_type_k": flags.cache_type_k,
            "cache_type_v": flags.cache_type_v})
    if kv is not None:
        rp.flags = rp.flags.model_copy(update={"cache_type_k": kv,
                                               "cache_type_v": kv})
    # vision projector: attach it if it's on disk, otherwise run text-only
    # rather than refusing — a vision model still works for text, and the user
    # can download the projector separately to turn vision on
    # `vision` is remembered across relaunches: a ctx change must not silently
    # switch vision back on and eat the VRAM the user just freed.
    if vision is None:
        vision = not bool((st.read_state() or {}).get("no_vision"))
    mm = getattr(reg_full.models.get(model), "mmproj", None)
    extra = (["--mmproj", str(rigma_home() / "models" / mm.file)]
             if vision and mm is not None and _model_on_disk(mm) else None)
    # Some ggufs ship a chat template that llama.cpp cannot use — Apriel-1.6's
    # decensored build self-assigns `{%- set messages = messages ... -%}`, which
    # minja evaluates as unbounded recursion and the process dies with
    # STATUS_STACK_BUFFER_OVERRUN before it ever allocates a context. A repaired
    # template dropped next to the model overrides the embedded one.
    tmpl = rigma_home() / "templates" / f"{model}.jinja"
    if tmpl.is_file():
        extra = (extra or []) + ["--chat-template-file", str(tmpl)]
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = rigma_home() / "models" / rp.gguf.file
    port = int(s["public_port"]) - 1
    st.kill_pid(int(s.get("engine_pid", -1)))
    _await_port_free(port)   # Windows TIME_WAIT grace
    # First load of a never-seen model+quant: with the old engine already dead,
    # VRAM is free — auto-tune the hardware-specific toggles ONCE, then launch
    # the winner. Cached forever after. Skipped for ctx-relaunches (a deliberate
    # reconfigure, not a fresh load), CPU, and when disabled.
    from .bench import auto_calibrate, is_calibrated
    if (ctx is None and rp.backend != "cpu"
            and os.environ.get("RIGMA_AUTO_CALIBRATE", "1") != "0"
            and (force_calibrate
                 or not is_calibrated(rp.model_slug, rp.gguf.quant, rp.backend))):
        _write_calib_marker(rp.model_slug, "starting")
        try:
            rp = auto_calibrate(rp, exe, model_path, port=port, extra_args=extra,
                                progress=lambda lbl:
                                _write_calib_marker(rp.model_slug, lbl))
        except Exception:
            pass   # tuning is best-effort; fall through to a normal launch
        finally:
            _clear_calib_marker()
        _await_port_free(port)   # last trial engine just released the port
    try:
        sp = runtime.launch_server(exe, rp, model_path,
                                   port=int(s["public_port"]) - 1,
                                   extra_args=extra)
    except Exception:
        # old engine is gone but the UI is still up — record an unloaded
        # state (not clear) so the UI stays manageable and can retry a load
        st.write_state(s["model"], s["quant"], int(s["public_port"]),
                       engine_pid=-1, ui_pid=int(s.get("ui_pid", os.getpid())),
                       backend=s.get("backend", "unknown"),
                       use_case=s.get("use_case", "general"),
                       ctx=int(s.get("ctx", 0)), unloaded=True,
                       gguf=s.get("gguf", ""))
        raise
    st.write_state(rp.model_slug, rp.gguf.quant, int(s["public_port"]),
                   engine_pid=sp.proc.pid,
                   ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=rp.backend, use_case=s.get("use_case", "general"),
                   ctx=rp.flags.ctx, kv_cache=rp.flags.cache_type_k or "",
                   no_vision=not vision, gguf=rp.gguf.file)
    return st.read_state() or {}


def _await_port_free(port: int, tries: int = 10, delay: float = 0.3) -> None:
    """After killing the old engine, its port lingers in TIME_WAIT briefly on
    Windows; wait for it to free before relaunching to avoid a bind crash."""
    import socket
    import time
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(delay)


def perform_unload() -> dict:
    """Stop the engine to free VRAM/RAM; the UI and state stay up so the
    model can be reloaded (or another one launched) with one click."""
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if s.get("unloaded"):
        raise RuntimeError("engine is already unloaded")
    st.kill_pid(int(s.get("engine_pid", -1)))
    st.write_state(s["model"], s["quant"], int(s["public_port"]),
                   engine_pid=-1, ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=s.get("backend", "unknown"),
                   use_case=s.get("use_case", "general"),
                   ctx=int(s.get("ctx", 0)), unloaded=True,
                   gguf=s.get("gguf", ""))
    return st.read_state()


def perform_load(registry=None, profile=None) -> dict:
    """Relaunch the model recorded in an unloaded state."""
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if not s.get("unloaded"):
        raise RuntimeError(f"{s['model']} is already loaded")
    return perform_switch(s["model"], registry, profile)


def perform_recalibrate(registry=None, profile=None) -> dict:
    """Forget the running model's tune and re-optimize it now (unload -> quick
    sweep -> launch the fresh winner). For when a noisy measurement crowned a
    config that's actually slower — the sweep always includes baseline, so a
    re-tune can only match or beat plain defaults."""
    from . import state as st
    from .bench import clear_calibration
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    clear_calibration(s["model"], s["quant"], s.get("backend", "unknown"))
    return perform_switch(s["model"], registry, profile, force_calibrate=True)
