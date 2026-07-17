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
    """Calibrated decode speed for the running combo, if bench ever ran."""
    try:
        cal = json.loads((rigma_home() / "calibration.json")
                         .read_text(encoding="utf-8"))
        return float(cal[f"{model}:{quant}:{backend}"]["tg_tps"])
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


def _free_current(profile, state: dict, reg):
    """A copy of `profile` with the CURRENTLY-running model's RAM footprint
    added back — perform_switch/ctx-change kill that engine before launching,
    so fitting against live free RAM is wrong. Without this, a ctx change on
    the running model probes against its own occupied RAM and reports an
    absurdly small ceiling (live repro 2026-07-18: 35B ctx change said 'tops
    out at 8,192' while it was running fine at 32K).

    Only RAM is credited: `_budgets` derives the VRAM budget from the card's
    static total capacity (not live-free), so VRAM is never starved — crediting
    it would over-report and risk an OOM launch."""
    if not state:
        return profile
    spec = reg.models.get(state.get("model", ""))
    if spec is None or not spec.ggufs:
        return profile
    # over-credit is safe (32GB RAM); fit_gguf then computes the real split
    freed_mb = max(g.bytes for g in spec.ggufs) / 2**20
    if spec.mmproj:
        freed_mb += spec.mmproj.bytes / 2**20
    return profile.model_copy(update={
        "ram_free_mb": profile.ram_free_mb + int(freed_mb)})


def _resolve_for(slug: str, state: dict, registry, profile):
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import resolve
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
    p = _free_current(p, state, reg)   # count the outgoing engine as freed
    return resolve(p, reg, use_case=state.get("use_case", "general"),
                   model_override=slug), reg, p


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


def perform_switch(model: str, registry=None, profile=None,
                   ctx: int | None = None) -> dict:
    """Stop the running engine and launch `model` in its place; with `ctx`,
    relaunch (same model allowed) at a requested context size.

    Raises RuntimeError with a user-facing message on any failure; a failure
    after the old engine died clears state (one requested plan, one honest
    result — the fallback ladder stays a CLI behavior)."""
    from . import runtime
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if model == s.get("model") and not s.get("unloaded") and ctx is None:
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
    trimmed = Registry(reg_full.gpus,
                       {**reg_full.models,
                        model: spec_full.model_copy(update={"ggufs": on_disk})},
                       reg_full.combos, reg_full.use_cases)
    rp, _, p = _resolve_for(model, s, trimmed, profile)
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
    # vision projector: attach it if it's on disk, otherwise run text-only
    # rather than refusing — a vision model still works for text, and the user
    # can download the projector separately to turn vision on
    mm = getattr(reg_full.models.get(model), "mmproj", None)
    extra = (["--mmproj", str(rigma_home() / "models" / mm.file)]
             if mm is not None and _model_on_disk(mm) else None)
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = rigma_home() / "models" / rp.gguf.file
    st.kill_pid(int(s.get("engine_pid", -1)))
    _await_port_free(int(s["public_port"]) - 1)   # Windows TIME_WAIT grace
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
                       ctx=int(s.get("ctx", 0)), unloaded=True)
        raise
    st.write_state(rp.model_slug, rp.gguf.quant, int(s["public_port"]),
                   engine_pid=sp.proc.pid,
                   ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=rp.backend, use_case=s.get("use_case", "general"),
                   ctx=rp.flags.ctx)
    return st.read_state()


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
                   ctx=int(s.get("ctx", 0)), unloaded=True)
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
