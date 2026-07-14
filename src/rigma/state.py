from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from .runtime import rigma_home


def state_path() -> Path:
    return rigma_home() / "state.json"


def write_state(model_slug: str, quant: str, public_port: int,
                engine_pid: int, ui_pid: int, backend: str = "unknown",
                use_case: str = "general", ctx: int = 0) -> None:
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "model": model_slug, "quant": quant, "public_port": public_port,
        "engine_pid": engine_pid, "ui_pid": ui_pid, "backend": backend,
        "use_case": use_case, "ctx": ctx, "started_at": time.time(),
    }, indent=2), encoding="utf-8")


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
    if not pid_alive(int(s.get("engine_pid", -1))):
        clear_state()
        return None
    return s


def kill_pid(pid: int) -> None:
    try:
        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=10)
        except psutil.TimeoutExpired:
            p.kill()
    except psutil.NoSuchProcess:
        pass
