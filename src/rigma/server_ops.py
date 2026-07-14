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


def _resolve_for(slug: str, state: dict, registry, profile):
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import resolve
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
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


def perform_switch(model: str, registry=None, profile=None) -> dict:
    """Stop the running engine and launch `model` in its place.

    Raises RuntimeError with a user-facing message on any failure; a failure
    after the old engine died clears state (one requested plan, one honest
    result — the fallback ladder stays a CLI behavior)."""
    from . import runtime
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if model == s.get("model"):
        raise RuntimeError(f"{model} is already running")
    rp, _, _ = _resolve_for(model, s, registry, profile)
    if rp.model_slug != model:
        raise RuntimeError(f"{model} does not fit this machine")
    if not _model_on_disk(rp.gguf):
        raise RuntimeError(
            f"{model} is not downloaded — run: rigma up --model {model}")
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = rigma_home() / "models" / rp.gguf.file
    st.kill_pid(int(s.get("engine_pid", -1)))
    try:
        sp = runtime.launch_server(exe, rp, model_path,
                                   port=int(s["public_port"]) - 1)
    except Exception:
        st.clear_state()  # old engine is gone; don't advertise a dead server
        raise
    st.write_state(rp.model_slug, rp.gguf.quant, int(s["public_port"]),
                   engine_pid=sp.proc.pid,
                   ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=rp.backend, use_case=s.get("use_case", "general"),
                   ctx=rp.flags.ctx)
    return st.read_state()
