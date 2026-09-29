from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from .atomicio import atomic_write_json
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
    # AUDIT F08-1: a PID alone does not identify a process. Windows reallocates
    # numbers, and state.json outlives the processes it names, so a stale record
    # can point at anything. These are the create times of engine_pid/ui_pid as
    # recorded at launch, and they are what makes "this is still our process"
    # checkable. 0.0 means "unknown" (a record written before this field existed,
    # or a pid that had already exited) and is treated as no evidence, not as a
    # mismatch — an old state.json keeps behaving exactly as it used to.
    "engine_started_at": 0.0, "ui_started_at": 0.0,
    # R3-STORE-2: the engine RUNTIME (llamacpp/vllm). Named here so a record
    # written before the field existed still merges to a usable value instead of
    # raising KeyError out of `update_state` — see `_write_record`.
    "engine": None,
    # R3-ENG-3: which BINARY served the launch, as a dict, or None for "not
    # recorded". Separate from `engine` above: that one is the runtime name and
    # is compared for equality by calibration, so it must stay a string.
    "engine_binary": None,
}

_PID_STAMP = {"engine_pid": "engine_started_at", "ui_pid": "ui_started_at"}


def _create_time(pid: int) -> float:
    """When this process started, or 0.0 if that cannot be established."""
    try:
        if pid <= 0:
            return 0.0
        return float(psutil.Process(pid).create_time())
    except Exception:
        return 0.0


def _is_recorded_process(pid: int, started_at) -> bool:
    """Is `pid` still the process the record was written about?

    True when the record carries no identity to check (started_at 0.0), so a
    pre-upgrade state.json is never made unkillable. Otherwise the create time
    must match: a recycled pid belongs to some other process and must be left
    alone. Unreadable identity counts as "not ours" — the safe direction, since
    the only action gated on this is terminating a process.
    """
    if pid <= 0:
        return False
    try:
        want = float(started_at)
    except (TypeError, ValueError):
        return False
    if want <= 0.0:
        return True
    try:
        return abs(psutil.Process(pid).create_time() - want) < 1.0
    except Exception:
        return False


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
        # AUDIT F08-1: the identity of the two pids above, so a later
        # `rigma stop` can tell "our engine" from "whatever now owns that
        # number". See _is_recorded_process.
        "engine_started_at": rec["engine_started_at"],
        "ui_started_at": rec["ui_started_at"],
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
        # R3-VLLM-4: the ENGINE RUNTIME (llamacpp/vllm). This dict is a fixed key
        # list, so a field not named here is SILENTLY dropped from disk — the
        # first version of this fix added it to write_state's signature only and
        # `read_state()["engine"]` raised KeyError. Its own test caught it.
        #
        # R3-STORE-2: `.get`, not `[...]`. The same KeyError was still reachable
        # from the other direction: a state.json written by a build BEFORE this
        # field existed has no `engine`, and `update_state` merges the defaults
        # into the stored record before rewriting it — so `perform_unload()` on
        # such a file raised KeyError AFTER the engine had already been killed,
        # leaving a record claiming a live engine. `_FIELD_DEFAULTS` names it too,
        # so a caller who does not set it gets the documented default rather than
        # a crash. A record that predates a field must degrade, not explode.
        "engine": rec.get("engine"),
        # R3-ENG-3: WHICH BINARY served this launch — {"kind", "name", "path", "source"}.
        # Distinct from `engine` above, which is the runtime (llamacpp/vllm). Needed
        # because a registered third-party build can now be chosen over the pin, and
        # "which engine is actually running" was previously unanswerable from state: the
        # owner found out only by reading the process table. `.get` for the same
        # back-compat reason as `engine` — a record written before this field exists must
        # degrade rather than raise.
        "engine_binary": rec.get("engine_binary"),
    }
    atomic_write_json(state_path(), out)
    return out


def write_state(model_slug: str, quant: str, public_port: int,
                engine_pid: int, ui_pid: int, backend: str = "unknown",
                use_case: str = "general", ctx: int = 0,
                unloaded: bool = False, kv_cache: str = "",
                no_vision: bool = False, gguf: str = "",
                kv_fp: str = "", engine: str | None = None,
                engine_binary: dict | None = None) -> None:
    """Write a whole record. Every field not named reverts to its default —
    which is what a launch wants and what an edit must never do; see
    `update_state`.

    `engine` is the ENGINE RUNTIME (llamacpp/vllm), a different axis from
    `backend`, which is the llama.cpp COMPUTE backend (vulkan/rocm/cuda/cpu).
    Recording it matters because `kv_fp`, the GGUF fit arithmetic and the slot
    cache are all llama.cpp's: a status that showed a vLLM launch under a
    llama.cpp compute backend would be describing an engine that is not running.

    It defaults to **None, not "llamacpp"**, and that is deliberate. This function
    is called from places that did not launch an engine at all (the UI-only
    `rigma up`, `perform_unload`), and a default of "llamacpp" would make every
    state.json ever written — including one from a vLLM launch that named it —
    assert llama.cpp. Absent means "not recorded"; a launcher that knows says so.
    """
    _write_record({"model": model_slug, "quant": quant,
                   "public_port": public_port, "engine_pid": engine_pid,
                   "ui_pid": ui_pid, "backend": backend, "use_case": use_case,
                   "ctx": ctx, "started_at": time.time(), "unloaded": unloaded,
                   "kv_cache": kv_cache, "no_vision": no_vision, "gguf": gguf,
                   "kv_fp": kv_fp, "engine": engine,
                   "engine_binary": engine_binary,
                   "engine_started_at": _create_time(engine_pid),
                   "ui_started_at": _create_time(ui_pid)})


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
    # AUDIT F08-1: a record that changes hands (a new engine pid after a model
    # switch, or -1 after an unload) must not carry the previous process's
    # identity forward: a stale stamp would make the NEW pid look like a
    # mismatch and so unkillable.
    for pid_key, stamp_key in _PID_STAMP.items():
        if pid_key in changed and stamp_key not in changed:
            rec[stamp_key] = _create_time(int(rec.get(pid_key) or -1))
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
    ui_pid = int(s.get("ui_pid", -1))
    ui_alive = pid_alive(ui_pid) and _is_recorded_process(ui_pid, s.get("ui_started_at"))
    if s.get("unloaded"):
        # engine deliberately stopped to free VRAM/RAM; the UI process is
        # the thing that must still be alive
        if not ui_alive:
            clear_state()
            return None
        return s
    engine_pid = int(s.get("engine_pid", -1))
    engine_alive = (pid_alive(engine_pid)
                    and _is_recorded_process(engine_pid, s.get("engine_started_at")))
    if not ui_alive and engine_alive:
        # terminal closed → UI died but llama-server lingers. Left alone this
        # locks the user out ("already running" with a dead UI). Reap it.
        kill_recorded(s, "engine_pid")
        clear_state()
        return None
    if not engine_alive:
        clear_state()
        return None
    return s


def kill_recorded(s: dict, pid_key: str) -> bool:
    """Terminate the process a state record names — but only if it is still
    that process. Returns True when a process was actually signalled.

    AUDIT F08-1: the old call sites asked only `pid_alive(pid)` and then killed
    whatever owned the number. state.json survives the processes it names (it is
    cleared only by `rigma stop`, `server_running()` or `up`'s finally block),
    and Windows reallocates pids, so a record left behind by a crash could name
    the owner's editor, browser or build. `_is_recorded_process` requires the
    create time recorded at launch to still match; on a mismatch nothing is
    killed and the caller clears the record.
    """
    pid = int(s.get(pid_key, -1) or -1)
    if not _is_recorded_process(pid, s.get(_PID_STAMP[pid_key])):
        return False
    kill_pid(pid)
    return True


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
