"""R3-ORPHAN: an engine with no state file, and the only way back.

`rigma up`'s `finally` is what stops the engine it started. A hard kill —
closing the terminal, a power event, a crash — skips it, and the llama-server
keeps the inference port. `state.json` is written by that same process, so the
record can be gone while the engine is not: the next `rigma up` sees a foreign
listener on the port, refuses, and `rigma stop` answers "not running" because
there is no record to read. The user is locked out of their own engine, with a
model loaded and VRAM committed, and the only way back is Task Manager.

The distinction that matters is not "is the port free" but **whose** the listener
is. A listener running Rigma's own engine binary out of Rigma's own engines
directory, answering `/health`, is Rigma's engine. This module answers that
question and reconstructs a state record from what the engine itself reports, so
`rigma up --reattach` can bring the UI back to the model that is already loaded.

Why not reattach automatically: adopting a process we did not start is a
different kind of act from starting one, and `rigma up` refusing loudly is what
tells the user their previous session ended abnormally. The refusal now names the
remedy instead of implying the port is held by a stranger.
"""
from __future__ import annotations

import json
from pathlib import Path

# How long to wait for the engine to answer. A loaded engine answers /health
# immediately — this is a liveness question about a process that is already
# listening, not a load in progress.
PROBE_TIMEOUT = 5.0


def is_rigma_engine(pid: int) -> bool:
    """Is `pid` running an engine binary out of Rigma's own engines directory?

    Deliberately narrow: the executable must be named `llama-server*` (or
    `vllm*`) AND sit under `<rigma home>/engines/`. A llama-server the user
    built themselves is not Rigma's to adopt, and a name check alone would claim
    it.
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        from .runtime import rigma_home

        exe = psutil.Process(pid).exe()
    except Exception:
        return False
    if not exe:
        return False
    try:
        name = Path(exe).name.lower()
        under = Path(exe).resolve().is_relative_to(
            (rigma_home() / "engines").resolve())
    except Exception:
        return False
    if not under:
        return False
    return name.startswith("llama-server") or name.startswith("vllm")


def engine_props(port: int, timeout: float = PROBE_TIMEOUT) -> dict | None:
    """`/props` from a llama.cpp server on `port`, or None if it is not one.

    Returns None for anything that is not a healthy llama.cpp server: a foreign
    service, a closed port, a server still loading. Absence is not an error here,
    it is the answer to "can we adopt this".
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{int(port)}/health",
                timeout=timeout) as r:
            if r.status != 200:
                return None
            if b"ok" not in r.read(200).lower():
                return None
        with urllib.request.urlopen(
                f"http://127.0.0.1:{int(port)}/props",
                timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    except Exception:
        return None


def _backend_from_exe(exe: str) -> str:
    """The compute backend, from `<engines>/<build>/<backend>/llama-server.exe`.

    Read from the PATH rather than guessed, because the path is the only place
    the running build recorded which backend it is. Empty when the layout is not
    the one Rigma creates — a state record saying "unknown" is honest, and a
    wrong backend name is not.
    """
    try:
        parts = Path(exe).resolve().parts
        i = len(parts) - 1 - list(reversed(parts)).index("engines")
        # engines/<build>/<backend>/<binary>
        return parts[i + 2] if len(parts) > i + 2 else ""
    except Exception:
        return ""


def record_from_props(props: dict, pid: int, exe: str,
                      public_port: int) -> dict:
    """A `write_state` keyword set describing the engine that is already up.

    Everything here is read from the engine, not remembered: the model path comes
    from `/props.model_path`, the window from `/props.default_generation_settings.n_ctx`,
    the backend from the binary's path, and the start time from the process. A
    field the engine does not report is left at its honest default rather than
    filled in with a plausible guess — `quant` in particular is a derived label
    that moves as a repo gains siblings, so an empty one is better than a wrong
    one.

    `gguf` is the loaded FILE, which is what the Models page matches the running
    row on, so it is taken from the engine and never re-derived.
    """
    props = props or {}
    dgs = props.get("default_generation_settings") or {}
    try:
        ctx = int(dgs.get("n_ctx") or 0)
    except (TypeError, ValueError):
        ctx = 0
    model_path = str(props.get("model_path") or "")
    gguf = Path(model_path).name if model_path else ""
    # The slug is the file's stem: Rigma's own slug for a custom install is not
    # recoverable from the engine, and a wrong slug is worse than a plain one.
    slug = Path(gguf).stem if gguf else ""
    return {
        "model_slug": slug, "quant": "", "public_port": int(public_port),
        "engine_pid": int(pid), "ui_pid": -1, "backend": _backend_from_exe(exe),
        "ctx": ctx, "gguf": gguf, "engine": "llamacpp",
    }


def find_engines(port: int) -> list[tuple[int, dict]]:
    """Every listener on `port` that is Rigma's own healthy engine.

    A list, not an optional: "none", "one" and "more than one" are three
    different situations and collapsing them loses the one that needs a human.
    Two healthy Rigma engines on one port is not something to adopt, it is
    something to report.
    """
    found: list[tuple[int, dict]] = []
    try:
        import psutil
    except Exception:
        return found
    try:
        conns = psutil.net_connections(kind="tcp")
    except Exception:
        return found
    seen: set[int] = set()
    for c in conns:
        if not (c.laddr and c.laddr.port == int(port) and c.status == "LISTEN"
                and c.pid):
            continue
        if c.pid in seen:
            continue
        seen.add(c.pid)
        if not is_rigma_engine(c.pid):
            continue
        props = engine_props(port)
        if props:
            found.append((int(c.pid), props))
    return found


def describe(port: int, pid: int, props: dict) -> str:
    """One sentence naming the engine and the model it has loaded."""
    dgs = (props or {}).get("default_generation_settings") or {}
    try:
        ctx = int(dgs.get("n_ctx") or 0)
    except (TypeError, ValueError):
        ctx = 0
    name = Path(str((props or {}).get("model_path") or "")).name or "a model"
    return (f"Rigma's own engine (pid {pid}) is still serving {name} on :{port}"
            + (f" with a {ctx}-token window" if ctx else ""))
