from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from .runtime import rigma_home


def state_path() -> Path:
    return rigma_home() / "state.json"


# Every field state.json carries, and the value a caller who does not name it
# gets. Named once so `write_state` (whole record) and `update_state` (merge)
# cannot drift apart about what a record even contains.
_FIELD_DEFAULTS = {
    "model": "", "quant": "", "public_port": 0, "engine_pid": -1, "ui_pid": -1,
    "backend": "unknown", "use_case": "general", "ctx": 0, "unloaded": False,
    "kv_cache": "", "no_vision": False, "gguf": "", "kv_fp": "",
}


def _write_record(rec: dict) -> dict:
    """Serialise one complete record. Key order is fixed here so a merged write
    and a full write produce the same file for the same values."""
    out = {
        "model": rec["model"], "quant": rec["quant"],
        "public_port": rec["public_port"], "engine_pid": rec["engine_pid"],
        "ui_pid": rec["ui_pid"], "backend": rec["backend"],
        "use_case": rec["use_case"], "ctx": rec["ctx"],
        "started_at": rec["started_at"], "unloaded": rec["unloaded"],
        "kv_cache": rec["kv_cache"],
        # the vision projector was deliberately left off this launch; sticky,
        # so a later ctx change doesn't silently reload it and eat the VRAM
        "no_vision": rec["no_vision"],
        # which FILE is loaded. `quant` is a derived label and moves when the
        # repo gains siblings; the filename does not, so it is what the Models
        # page matches the running row on.
        "gguf": rec["gguf"],
        # names the on-disk KV cache this launch may restore. Recorded at
        # launch rather than re-derived at unload: by then the plan is gone,
        # and a re-derivation that disagreed with what actually launched is
        # precisely the mismatch kvcache exists to prevent.
        "kv_fp": rec["kv_fp"],
    }
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def write_state(model_slug: str, quant: str, public_port: int,
                engine_pid: int, ui_pid: int, backend: str = "unknown",
                use_case: str = "general", ctx: int = 0,
                unloaded: bool = False, kv_cache: str = "",
                no_vision: bool = False, gguf: str = "",
                kv_fp: str = "") -> None:
    """Write a whole record. Every field not named reverts to its default —
    which is what a launch wants and what an edit must never do; see
    `update_state`."""
    _write_record({"model": model_slug, "quant": quant,
                   "public_port": public_port, "engine_pid": engine_pid,
                   "ui_pid": ui_pid, "backend": backend, "use_case": use_case,
                   "ctx": ctx, "started_at": time.time(), "unloaded": unloaded,
                   "kv_cache": kv_cache, "no_vision": no_vision, "gguf": gguf,
                   "kv_fp": kv_fp})


def update_state(**changed) -> dict:
    """Overwrite ONLY the named fields of the record on disk; return the result.

    AUDIT F22: docs/audit-2026-09-04-full.md — `write_state` rebuilds the whole
    record from keyword defaults, so a caller that forgets one silently reverts
    it. perform_unload omitted `no_vision` and `kv_cache`: unloading a vision
    model that was running text-only turned the projector back on and ate the
    ~888 MiB the user had deliberately freed, and orphaned the saved prompt
    cache. Merging means a field can only change when something names it.

    `started_at` is the exception, and stays a stamp of when the record was
    written — `rigma status` reports uptime from it, and carrying a launch time
    across an unload would report an uptime the engine never had. Pass it
    explicitly to keep a specific value.
    """
    cur = read_state()
    if cur is None:
        # AUDIT F22: there is nothing to merge into. Falling back to the
        # defaults here would RESURRECT state.json as an all-empty record
        # (model "", port 0, ui_pid -1) — which is what happens when a
        # `clear_state()` lands between a caller's own read and this one, or
        # when read_state swallows a torn read. "No engine is running" must
        # stay "no file", not "a file describing no engine".
        return {}
    rec = {**_FIELD_DEFAULTS, **cur, **changed}
    rec["started_at"] = changed.get("started_at", time.time())
    return _write_record(rec)


def read_state() -> dict | None:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_state() -> None:
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass


def pid_alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


def server_running() -> dict | None:
    s = read_state()
    if s is None:
        return None
    ui_alive = pid_alive(int(s.get("ui_pid", -1)))
    if s.get("unloaded"):
        # engine deliberately stopped to free VRAM/RAM; the UI process is
        # the thing that must still be alive
        if not ui_alive:
            clear_state()
            return None
        return s
    engine_alive = pid_alive(int(s.get("engine_pid", -1)))
    if not ui_alive and engine_alive:
        # terminal closed → UI died but llama-server lingers. Left alone this
        # locks the user out ("already running" with a dead UI). Reap it.
        kill_pid(int(s["engine_pid"]))
        clear_state()
        return None
    if not engine_alive:
        clear_state()
        return None
    return s


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=10)
        except psutil.TimeoutExpired:
            p.kill()
    except psutil.NoSuchProcess:
        pass
