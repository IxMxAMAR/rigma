"""Autonomous-run store: one long-running mission at a time.

A Run drives the agentic loop headlessly toward a fixed mission. This module owns
the durable state — run.json, plan.json (the model's todo/working-memory),
progress.md (semantic log the user watches), actions.jsonl (deterministic audit)
— plus the small helpers the executor and tools need. No asyncio, no engine: pure
state so it's trivially testable.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

# terminal statuses release the active-run pointer
TERMINAL = {"done", "stalled", "frozen", "budget_exhausted", "stopped", "error"}
PROFILES = {"all", "no-network", "no-delete", "confined"}
MAX_ITERS = 2000
BUDGET_HOURS_DEFAULT = 8.0
BUDGET_HOURS_MAX = 48.0


def _runs_dir() -> Path:
    d = rigma_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    d = _runs_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _active_path() -> Path:
    return _runs_dir() / "active.json"


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --- lifecycle ---------------------------------------------------------------

def create(mission: str, session_id: str, workspace: str = "",
           profile: str = "all", budget_hours: float = BUDGET_HOURS_DEFAULT,
           token_cap: int = 0) -> dict:
    rid = new_id()
    now = time.time()
    hours = max(0.1, min(float(budget_hours or BUDGET_HOURS_DEFAULT),
                         BUDGET_HOURS_MAX))
    run = {
        "id": rid, "mission": str(mission), "session_id": session_id,
        "workspace": workspace or "",
        "profile": profile if profile in PROFILES else "all",
        "status": "running", "iteration": 0, "tokens_used": 0,
        "token_cap": int(token_cap or 0),
        "started_at": now, "deadline": now + hours * 3600,
        "last_progress_at": now, "error_streak": 0, "lazy_streak": 0,
        "verified_once": False, "external_calls": 0, "paused": False,
        "steer_queue": [], "summary": "", "halt_reason": "",
    }
    d = run_dir(rid)
    (d / "outputs").mkdir(exist_ok=True)
    write_plan(rid, [])
    _atomic_write(d / "progress.md",
                  f"# Autonomous run {rid}\nMission: {mission}\n\n")
    save(run)
    _atomic_write(_active_path(), json.dumps({"id": rid}))
    return run


def save(run: dict) -> None:
    _atomic_write(run_dir(run["id"]) / "run.json",
                  json.dumps(run, indent=2))


def load(run_id: str) -> dict | None:
    try:
        return json.loads((run_dir(run_id) / "run.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return None


def active() -> dict | None:
    try:
        rid = json.loads(_active_path().read_text(encoding="utf-8"))["id"]
    except Exception:
        return None
    return load(rid)


def clear_active() -> None:
    try:
        _active_path().unlink()
    except FileNotFoundError:
        pass


def set_status(run: dict, status: str, halt_reason: str = "") -> None:
    run["status"] = status
    if halt_reason:
        run["halt_reason"] = halt_reason
    save(run)
    if status in TERMINAL:
        a = active()
        if a and a["id"] == run["id"]:
            clear_active()


# --- plan (todo / working memory) --------------------------------------------

def read_plan(run_id: str) -> list:
    try:
        return json.loads((run_dir(run_id) / "plan.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return []


def write_plan(run_id: str, plan: list) -> None:
    _atomic_write(run_dir(run_id) / "plan.json", json.dumps(plan, indent=2))


def plan_add(run_id: str, text: str) -> int:
    plan = read_plan(run_id)
    tid = max((t.get("id", 0) for t in plan), default=0) + 1
    plan.append({"id": tid, "text": str(text)[:300], "status": "pending"})
    write_plan(run_id, plan)
    return tid


def plan_complete(run_id: str, task_id) -> bool:
    plan = read_plan(run_id)
    hit = False
    for t in plan:
        if str(t.get("id")) == str(task_id):
            t["status"] = "done"
            hit = True
    write_plan(run_id, plan)
    return hit


def plan_update(run_id: str, task_id, text: str) -> bool:
    """Reword a step in place. Models reach for this naturally; without it they
    burn tool calls on errors (and can stall the run on repeated failures)."""
    plan = read_plan(run_id)
    hit = False
    for t in plan:
        if str(t.get("id")) == str(task_id):
            t["text"] = str(text)[:300]
            hit = True
    write_plan(run_id, plan)
    return hit


def pending_tasks(run_id: str) -> list:
    return [t for t in read_plan(run_id) if t.get("status") == "pending"]


def done_summary(run_id: str, limit: int = 6) -> str:
    """Completed steps — the anti-restart signal. Telling a small model what is
    ALREADY DONE stops it redoing finished phases far better than restating the
    mission (which it reads as a fresh instruction)."""
    done = [t for t in read_plan(run_id) if t.get("status") == "done"]
    if not done:
        return ""
    tail = done[-limit:]
    more = f" (+{len(done) - len(tail)} earlier)" if len(done) > len(tail) else ""
    return "; ".join(f"#{t['id']} {t['text']}" for t in tail) + more


def plan_counts(run_id: str) -> tuple[int, int]:
    """(done, total) — a one-glance position marker for the driving line, so the
    model never has to go hunting on disk to work out where it is."""
    plan = read_plan(run_id)
    return sum(1 for t in plan if t.get("status") == "done"), len(plan)


def next_pending(run_id: str) -> str:
    """The single next step. One target beats a list — a list invites a small
    model to jump around or start from the top."""
    pend = pending_tasks(run_id)
    return f"#{pend[0]['id']} {pend[0]['text']}" if pend else ""


def plan_summary(run_id: str, limit: int = 8) -> str:
    pend = pending_tasks(run_id)
    if not pend:
        return "(no pending plan items)"
    return "; ".join(f"#{t['id']} {t['text']}" for t in pend[:limit])


# --- progress log (semantic, user-facing) ------------------------------------

def append_progress(run_id: str, done: str, next_step: str,
                    workspace: str = "") -> None:
    line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done: "
            f"{str(done)[:600]}  ->  next: {str(next_step)[:600]}\n")
    with open(run_dir(run_id) / "progress.md", "a", encoding="utf-8") as f:
        f.write(line)
    if workspace:
        try:
            with open(Path(workspace) / "rigma-progress.md", "a",
                      encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def log_tool_action(run_id: str, name: str, args, result: str,
                    workspace: str = "") -> None:
    """Server-authored progress line. The model will not reliably narrate its
    own work (progress.md was empty in practice) and the server already knows
    exactly what ran — so the server writes the log."""
    a = args if isinstance(args, dict) else {}
    shown = ", ".join(f"{k}={str(v)[:60]}" for k, v in list(a.items())[:3])
    res = " ".join(str(result).split())[:200]
    append_progress(run_id, f"{name}({shown})", res or "(no output)", workspace)


def get_log_tail(run_id: str, n: int = 5) -> str:
    try:
        lines = (run_dir(run_id) / "progress.md").read_text(
            encoding="utf-8").splitlines()
    except Exception:
        return ""
    prog = [ln for ln in lines if "->  next:" in ln]
    return "\n".join(prog[-n:])


# --- action audit (deterministic) --------------------------------------------

def append_action(run_id: str, tool: str, args, ok: bool) -> None:
    try:
        a = json.dumps(args, default=str)[:300]
    except Exception:
        a = str(args)[:300]
    rec = {"ts": time.time(), "tool": tool, "args": a, "ok": bool(ok)}
    with open(run_dir(run_id) / "actions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_actions(run_id: str) -> list:
    """The action trace, for post-mortem mining. A missing or partly-written
    file yields what it can — this feeds memory, which is never load-bearing."""
    out = []
    try:
        text = (run_dir(run_id) / "actions.jsonl").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except (ValueError, TypeError):
            continue
    return out


# --- budget ------------------------------------------------------------------

def budget_exceeded(run: dict) -> str:
    now = time.time()
    if now >= run.get("deadline", now + 1):
        return "time budget reached"
    if run.get("iteration", 0) >= MAX_ITERS:
        return "iteration cap reached"
    cap = run.get("token_cap") or 0
    if cap and run.get("tokens_used", 0) >= cap:
        return "token budget reached"
    return ""


def set_last_sample(run_id: str, paths: list) -> None:
    """Remember the files sample_files just handed out, so the model can act on
    them by REFERENCE. It cannot reliably retype a name like
    comfyui-airport-editorial_00013205_(2).webp — three attempts produced three
    different digit strings — so don't ask it to."""
    run = load(run_id)
    if run is not None:
        run["last_sample"] = [str(p) for p in paths][:50]
        save(run)


def get_last_sample(run_id: str) -> list:
    return (load(run_id) or {}).get("last_sample") or []


# --- live state (hot path) ----------------------------------------------------
# The activity feed + heartbeat are written many times per turn. Keeping them in
# run.json meant re-serialising ~100KB of history on every tool event and every
# tick — synchronous disk I/O on the event loop, and real write amplification
# over a long run. They live in their own small file instead.

def save_live(run_id: str, live: dict) -> None:
    try:
        _atomic_write(run_dir(run_id) / "live.json", json.dumps(live))
    except Exception:
        pass


def load_live(run_id: str) -> dict:
    try:
        return json.loads((run_dir(run_id) / "live.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return {}
