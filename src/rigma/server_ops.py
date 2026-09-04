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


def _measured_placement(rp, ctx: int) -> dict:
    """Weight placement that was MEASURED at this context, if there is one.

    `fit_gguf` recomputes ngl / n_cpu_moe from a VRAM model that is deliberately
    conservative — the right answer for a context nobody has tried, and the
    wrong one for a context somebody has. Qwen3.8-27B was pinned at ngl 63 after
    measuring 33.24 t/s there and relaunched at ngl 59, because the calculator
    disagreed with the machine. The machine wins: a number that came from a
    stopwatch beats a number that came from arithmetic about the same thing.

    Only on an EXACT context match. Placement measured at 32K says nothing about
    256K — the KV cache is most of what moved — so a near miss falls back to the
    calculator rather than guessing.
    """
    try:
        from .bench import load_calibration
        entry = load_calibration().get(
            f"{rp.model_slug}:{rp.gguf.quant}:{rp.backend}") or {}
        if int(entry.get("ctx") or 0) != int(ctx):
            return {}
        flags = entry.get("flags") or {}
        return {k: v for k, v in flags.items() if k in ("ngl", "n_cpu_moe")}
    except Exception:
        return {}          # a missing or corrupt calibration is not an error


def launch_fit_spec(spec, flags, *, vision: bool, ctx: int = 0):
    """The spec to run the fit against, plus whether it differs from `spec`.

    AUDIT F18: docs/audit-2026-09-04-full.md — the plan and the launch have to
    agree about what sits on the card. `resolve.with_launch_overheads` folds
    both disagreements into the one mmproj slot `fit_gguf` already treats as
    non-offloadable memory: the projector comes OUT when vision is off (717-888
    MiB on the two shipped vision models, enough to cost two layers of GPU
    residency), and the speculative draft cache goes IN when speculation is on
    (~565 MiB at 32K, previously budgeted nowhere because spec_type was applied
    after the fit).

    The second return value says whether that changed the number. It is the
    guard on the extra fit: when nothing moved, the resolver's own arithmetic
    already stands and re-running it would only substitute the calculator's
    answer for a curated combo's.
    """
    from .resolve import with_launch_overheads
    ctx = int(ctx or flags.ctx)
    fit = with_launch_overheads(spec, vision=vision, ctx=ctx,
                                kv=flags.cache_type_k,
                                spec_type=flags.spec_type,
                                n_max=flags.spec_n_max)
    was = spec.mmproj.bytes if spec.mmproj else 0
    now = fit.mmproj.bytes if fit.mmproj else 0
    return fit, was != now


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
    from . import state as st
    reg = registry if registry is not None else Registry.load()
    prof = probe_hardware(reg.gpus)
    total = sum(g.vram_mb for g in prof.gpus)
    if not total:
        return None
    # Credit back the model we are already running. The adapter counter reports
    # everything held on the card, which INCLUDES our own engine — so with a
    # 15.7GB model loaded the panel said "other apps hold 15.7 GB of VRAM,
    # leaving 0.1 GB for the model" and advised closing a browser (owner,
    # 2026-08-25). The resolver has credited this back since 2026-08-21; this
    # read never did, so the number the user saw and the number rigma planned
    # against disagreed. "Other apps" has to mean other apps.
    s = st.read_state() or {}
    if s.get("model") and not s.get("unloaded"):
        prof = _free_current(prof, s, reg)
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
    # The model's own configuration fills in anything the caller did not ask
    # for. Precedence is explicit request > model default > resolver: a stored
    # default that could not be overridden would make changing context once
    # impossible on a model that pins it.
    launch = getattr(spec_full, "launch", None)
    if launch is not None:
        d = launch.as_overrides()
        if ctx is None and "ctx" in d:
            ctx = d["ctx"]
        if kv is None and "kv" in d:
            kv = d["kv"]
        if quant is None and "quant" in d:
            quant = d["quant"]
        if vision is None and "vision" in d:
            vision = d["vision"]
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
    # AUDIT F18: docs/audit-2026-09-04-full.md — speculation and vision are
    # settled BEFORE the fit. Both change what the card will actually hold, and
    # resolving them 30 lines after the arithmetic is what made this call charge
    # itself for a projector it then omitted from the command line.
    # Speculation only when the FILE carries the draft head: asking llama.cpp
    # for draft-mtp against a gguf without the tensors resets the Vulkan driver
    # rather than erroring (see ComboFlags.spec_type).
    if launch is not None and launch.is_set("spec_type"):
        from .hangar import file_has_mtp
        if launch.spec_type != "draft-mtp" or file_has_mtp(rp.gguf):
            rp.flags = rp.flags.model_copy(update={
                "spec_type": launch.spec_type,
                "spec_n_max": launch.spec_n_max or 1})
    # `vision` is remembered across relaunches: a ctx change must not silently
    # switch vision back on and eat the VRAM the user just freed.
    if vision is None:
        vision = not bool((st.read_state() or {}).get("no_vision"))
    if ctx is not None:
        # honest relaunch at a requested context: real fit math, not hope.
        # rp.flags.ctx is the calculator's grow-to-fit maximum for this quant.
        from .resolve import fit_gguf
        want = max(2048, min(int(ctx), spec_full.native_ctx))
        fit_spec, _ = launch_fit_spec(spec_full, rp.flags, vision=vision,
                                      ctx=want)
        flags = fit_gguf(fit_spec, rp.gguf, p, want, [])
        if flags is None:
            raise RuntimeError(
                f"ctx {want:,} doesn't fit — {model} ({rp.gguf.quant}) tops "
                f"out around {rp.flags.ctx:,} on this machine")
        update = {
            "ctx": flags.ctx, "n_cpu_moe": flags.n_cpu_moe, "ngl": flags.ngl,
            "cache_type_k": flags.cache_type_k,
            "cache_type_v": flags.cache_type_v}
        # ...except where the placement was MEASURED at this exact context.
        update.update(_measured_placement(rp, want))
        rp.flags = rp.flags.model_copy(update=update)
    else:
        # No ctx asked for, so the plan is the resolver's — which priced the
        # full spec. Re-place the weights only when the resident overhead has
        # actually moved; the freed projector memory is worth GPU layers, and
        # an unbudgeted draft cache is worth an offload nobody planned.
        fit_spec, differs = launch_fit_spec(spec_full, rp.flags, vision=vision)
        if differs:
            from .resolve import fit_gguf
            got = fit_gguf(fit_spec, rp.gguf, p, rp.flags.ctx, [])
            if got is not None:
                update = {"ngl": got.ngl, "n_cpu_moe": got.n_cpu_moe}
                # AUDIT F17/F18: docs/audit-2026-09-04-full.md — same rule as the
                # ctx branch above. `resolve` may already have written a measured
                # placement into these two keys; the calculator must not overwrite
                # a stopwatch. Reached on the ordinary path — a text-only vision
                # model that idle-unloads and reloads comes through here — and
                # that is the 37.59 -> 9.95 tok/s collapse the audit measured.
                update.update(_measured_placement(rp, rp.flags.ctx))
                rp.flags = rp.flags.model_copy(update=update)
    if kv is not None:
        rp.flags = rp.flags.model_copy(update={"cache_type_k": kv,
                                               "cache_type_v": kv})
    # vision projector: attach it if it's on disk, otherwise run text-only
    # rather than refusing — a vision model still works for text, and the user
    # can download the projector separately to turn vision on
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
        # AUDIT F22: docs/audit-2026-09-04-full.md — merge, never rebuild. The
        # whole-record form dropped kv_fp here, orphaning a prompt cache that
        # costs four minutes of prefill to rebuild, plus no_vision and kv_cache.
        st.update_state(engine_pid=-1,
                        ui_pid=int(s.get("ui_pid", os.getpid())),
                        unloaded=True)
        raise
    # Bring back the prompt cache if one was saved under EXACTLY this
    # configuration. A 120K window costs about four minutes of prefill to
    # rebuild and a couple of seconds to read off disk.
    from . import kvcache
    kv_fp = kvcache.fingerprint(kvcache.config_of(rp, str(exe)))
    try:
        kvcache.restore(int(s["public_port"]) - 1,
                        runtime.rigma_home() / "sessions", kv_fp)
    except Exception:
        pass          # a cache that will not load is a slow start, not a fault
    st.write_state(rp.model_slug, rp.gguf.quant, int(s["public_port"]),
                   engine_pid=sp.proc.pid,
                   ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=rp.backend, use_case=s.get("use_case", "general"),
                   ctx=rp.flags.ctx, kv_cache=rp.flags.cache_type_k or "",
                   no_vision=not vision, gguf=rp.gguf.file, kv_fp=kv_fp)
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
    # Write the conversation's KV cache out BEFORE the process dies. Everything
    # llama-server caches lives in that process; unloading to free the card is
    # otherwise paid for with a full re-prefill on the way back.
    if s.get("kv_fp"):
        from . import kvcache, runtime
        sessions = runtime.rigma_home() / "sessions"
        try:
            kvcache.save(int(s["public_port"]) - 1, sessions, s["kv_fp"],
                         meta={k: s.get(k) for k in ("model", "quant", "gguf",
                                                     "backend", "ctx")})
            kvcache.prune(sessions)
        except Exception:
            pass      # never let a cache write block freeing the GPU
    st.kill_pid(int(s.get("engine_pid", -1)))
    # AUDIT F22: docs/audit-2026-09-04-full.md — an unload changes exactly two
    # things: the engine is gone, and the record says so. Everything else is the
    # configuration the user chose and must survive. Rebuilding the record from
    # write_state's defaults reverted no_vision to False, so the next load
    # re-attached the projector and ate the VRAM the unload was for.
    return st.update_state(engine_pid=-1,
                           ui_pid=int(s.get("ui_pid", os.getpid())),
                           unloaded=True)


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
