"""Autonomous-run store: one long-running mission at a time.

A Run drives the agentic loop headlessly toward a fixed mission. This module owns
the durable state — run.json, plan.json (the model's todo/working-memory),
progress.md (semantic log the user watches), actions.jsonl (deterministic audit)
— plus the small helpers the executor and tools need. No asyncio, no engine: pure
state so it's trivially testable.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

# terminal statuses release the active-run pointer
TERMINAL = {"done", "stalled", "frozen", "budget_exhausted", "stopped",
            "error", "interrupted"}
# terminal states a run can be RESTARTED from (everything needed to continue
# is on disk; "done" stays done)
RESTARTABLE = {"interrupted", "stopped", "stalled", "frozen",
               "budget_exhausted", "error"}
PROFILES = {"all", "no-network", "no-delete", "confined"}
MAX_ITERS = 2000
# What a RESTART grants on top of a used-up step budget, the counterpart of the
# grace hour `restart_run` gives the clock: a resume exists to finish work, not
# to instantly re-die on the old cap (AUDIT F45).
RESTART_ITER_GRACE = 500
BUDGET_HOURS_DEFAULT = 8.0
BUDGET_HOURS_MAX = 48.0


def _runs_dir() -> Path:
    d = rigma_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# A run id becomes a DIRECTORY name and arrives from a URL path parameter.
# Starlette's {param} converter is [^/]+, so it stops %2f but not %5c, and
# uvicorn decodes before routing -- so GET /api/runs/..%5C..%5Cfoo%5Cbar built
# arbitrary directory trees, on the READ path, from a CORS-simple request that
# needed no rebinding. Same guard as skills._path_for.
# AUDIT F40: docs/audit-2026-09-04-full.md
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# CON, PRN, AUX, NUL, COM1-9, LPT1-9 are unopenable as files on Windows
_WIN_DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


class RunIdError(ValueError):
    """The requested id can't name a run directory."""


def run_dir(run_id: str, *, create: bool = False) -> Path:
    """The folder for a run, or raise. `create` is opt-in and belongs to the
    WRITE sites only: this used to mkdir unconditionally, so every reader --
    load(), read_plan(), the log endpoint -- conjured a directory for a run
    that does not exist, and a hostile id conjured it outside ~/.rigma."""
    rid = str(run_id or "")
    if not _SAFE_ID.match(rid) or _WIN_DEVICE.match(rid):
        raise RunIdError(f"'{rid[:40]}' is not a valid run id")
    d = (_runs_dir() / rid).resolve()
    if d.parent != _runs_dir().resolve():
        raise RunIdError("that id would write outside the runs folder")
    if create:
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
        "stop_reason": "",
    }
    d = run_dir(rid, create=True)
    (d / "outputs").mkdir(exist_ok=True)
    write_plan(rid, [])
    _atomic_write(d / "progress.md",
                  f"# Autonomous run {rid}\nMission: {mission}\n\n")
    save(run)
    _atomic_write(_active_path(), json.dumps({"id": rid}))
    return run


def save(run: dict, revive: bool = False) -> None:
    """Persist a run. Terminal is STICKY.

    The loop loads the run once per iteration, then awaits the engine for a
    long time; it holds a snapshot that says "running" the whole while. If a
    halt lands during that await — /stop, a budget trip, a crash — the loop's
    next save used to write that snapshot back and un-halt the run. The task
    was already dying, so the result was a zombie: "running" on disk with
    nothing driving it, refusing both resume and restart ("run is running"),
    clearable only by restarting the server (which is why boot has an
    orphan reaper). Diagnosed from an intermittent restart failure, 2026-07-21.

    A writer that does not know the run halted does not get to un-halt it: the
    on-disk status wins and the rest of the snapshot (iteration counters, token
    accounting) still persists. `revive=True` is the single deliberate
    exception — restart_run, which reattaches a loop on purpose.
    """
    if not revive and run.get("status") not in TERMINAL:
        cur = load(run["id"])
        if cur is not None and cur.get("status") in TERMINAL:
            # tell the caller the truth too, so a loop still holding this dict
            # sees the halt on its next check instead of running on stale state
            run["status"] = cur["status"]
            run["halt_reason"] = cur.get("halt_reason", "")
    _atomic_write(run_dir(run["id"], create=True) / "run.json",
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
    # IMP-8: persist a stable CODE alongside the prose. The run view has to
    # tell "finished" from "gave up" and name which cap was hit; matching on a
    # human sentence would break the first time a sentence is reworded.
    run["stop_reason"] = (stop_code(status, run.get("halt_reason", ""))
                          if status in TERMINAL else "")
    save(run)
    if status in TERMINAL:
        a = active()
        if a and a["id"] == run["id"]:
            clear_active()


# IMP-8: the closed set of ways a run can end. `unknown` is deliberate — a
# stop nobody classified is reported as unclassified, never guessed at.
STOP_REASONS = ("completed", "step_cap", "token_cap", "time_budget",
                "frozen_streak", "tool_errors", "no_progress",
                "paused_timeout", "user_stop", "session_deleted",
                "engine_error", "interrupted", "unknown")

# (status, halt_reason prefix) -> code, checked in order.
_STOP_MATCHES = (
    ("done", "", "completed"),
    ("budget_exhausted", "time budget", "time_budget"),
    ("budget_exhausted", "iteration cap", "step_cap"),
    ("budget_exhausted", "token budget", "token_cap"),
    ("frozen", "", "frozen_streak"),
    ("stalled", "too many tool errors", "tool_errors"),
    ("stalled", "no progress", "no_progress"),
    ("stalled", "paused waiting", "paused_timeout"),
    ("stopped", "", "user_stop"),
    ("error", "session was deleted", "session_deleted"),
    ("interrupted", "", "interrupted"),
    ("error", "", "engine_error"),
)


def stop_code(status: str, halt_reason: str = "") -> str:
    """Classify one terminal (status, halt_reason) pair into a STOP_REASONS
    code. Anything unrecognised is `unknown`, which is honest about the gap."""
    why = str(halt_reason or "")
    for want_status, prefix, code in _STOP_MATCHES:
        if status == want_status and why.startswith(prefix):
            return code
    return "unknown"


def stop_reason(run: dict) -> str:
    """Why this run is not running, as a code the UI can switch on.

    A run written by an older build has no `stop_reason` field, so it is
    derived from (status, halt_reason) instead of reported as unknown.
    """
    if str(run.get("status") or "") not in TERMINAL:
        return ""
    return (str(run.get("stop_reason") or "")
            or stop_code(str(run.get("status") or ""),
                         str(run.get("halt_reason") or "")))


def budget_snapshot(run: dict) -> dict:
    """What is LEFT of the run's step, token and wall-clock budgets.

    The run view shows only "stopped" today, so a run that spent its clock and
    a run that finished look identical. These are the same numbers the loop
    judges against (`iter_ceiling`, `deadline`, `token_cap`), read for display.
    """
    def _int(v) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    ceiling = iter_ceiling(run)
    used = _int(run.get("iteration"))
    started = float(run.get("started_at") or 0.0)
    deadline = float(run.get("deadline") or 0.0)
    left = max(0.0, deadline - time.time()) if deadline else 0.0
    cap = _int(run.get("token_cap"))
    tokens = _int(run.get("tokens_used"))
    return {
        "steps_used": used, "steps_total": ceiling,
        "steps_remaining": max(0, ceiling - used),
        "seconds_remaining": int(left),
        "time_budget_seconds": int(max(0.0, deadline - started))
        if deadline and started else 0,
        "tokens_used": tokens, "token_cap": cap,
        # None (not 0) when no token cap was set: "unlimited" and "exhausted"
        # are different answers and the UI must not render them the same.
        "tokens_remaining": max(0, cap - tokens) if cap else None,
    }


# --- plan (todo / working memory) --------------------------------------------

def read_plan(run_id: str) -> list:
    try:
        return json.loads((run_dir(run_id) / "plan.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return []


def write_plan(run_id: str, plan: list) -> None:
    _atomic_write(run_dir(run_id, create=True) / "plan.json",
                  json.dumps(plan, indent=2))


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


def plan_block(run_id: str, task_id, reason: str = "") -> bool:
    """Mark a step blocked so the run can route AROUND it. Before this, one
    impossible step held the whole remaining plan hostage until the global
    error streak killed the run — the classic naive-loop failure: nothing
    between 'keep hammering' and 'give up entirely'."""
    plan = read_plan(run_id)
    hit = False
    for t in plan:
        if str(t.get("id")) == str(task_id):
            t["status"] = "blocked"
            t["blocked_reason"] = str(reason)[:200]
            hit = True
    write_plan(run_id, plan)
    return hit


def blocked_tasks(run_id: str) -> list:
    return [t for t in read_plan(run_id) if t.get("status") == "blocked"]


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
    with open(run_dir(run_id, create=True) / "progress.md", "a",
              encoding="utf-8") as f:
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


# The active-run poll reads progress.md every 2 s. Reading and splitting the
# WHOLE file to produce 40 lines made total work O(n^2) over a run; only the
# tail can hold the last 40 lines, so read that.
_LOG_TAIL_BYTES = 64 * 1024


def get_log_tail(run_id: str, n: int = 5) -> str:
    p = run_dir(run_id) / "progress.md"
    try:
        size = p.stat().st_size
    except OSError:
        return ""
    try:
        with open(p, "rb") as f:
            if size > _LOG_TAIL_BYTES:
                f.seek(size - _LOG_TAIL_BYTES)
                data = f.read()
                nl = data.find(b"\n")       # the first line may be cut in half
                if nl >= 0:
                    data = data[nl + 1:]
            else:
                data = f.read()
    except Exception:
        return ""
    lines = data.decode("utf-8", "replace").splitlines()
    prog = [ln for ln in lines if "->  next:" in ln]
    if not prog and size > _LOG_TAIL_BYTES:
        # the last 64 KB held no progress lines (a huge tool dump, say) —
        # fall back to the whole file rather than report nothing
        try:
            lines = p.read_text(encoding="utf-8",
                                errors="replace").splitlines()
        except Exception:
            return ""
        prog = [ln for ln in lines if "->  next:" in ln]
    return "\n".join(prog[-n:])


# --- action audit (deterministic) --------------------------------------------

def append_action(run_id: str, tool: str, args, ok: bool) -> None:
    try:
        full = json.dumps(args, default=str)
    except Exception:
        full = str(args)
    # args are truncated for display, but identity must come from the FULL
    # args: write_file/run_python calls routinely exceed 300 chars, and two
    # different calls agreeing in their first 300 (same path, evolving
    # content) would otherwise collide into a false "loop" event and feed the
    # memory distiller a wrong premise.
    import hashlib
    rec = {"ts": time.time(), "tool": tool, "args": full[:300],
           "args_sha": hashlib.sha1(full.encode("utf-8", "replace"))
           .hexdigest()[:16],
           "ok": bool(ok)}
    with open(run_dir(run_id, create=True) / "actions.jsonl", "a",
              encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_actions(run_id: str) -> list:
    """The action trace, for post-mortem mining. A missing or partly-written
    file yields what it can — this feeds memory, which is never load-bearing."""
    out = []
    try:
        text = (run_dir(run_id) / "actions.jsonl").read_text(encoding="utf-8")
    except (RunIdError, FileNotFoundError, OSError):
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

def iter_ceiling(run: dict) -> int:
    """The step budget this run is judged against.

    Normally MAX_ITERS; a restart that had already spent it raises it by
    RESTART_ITER_GRACE (see `serve.restart_run`). The ceiling is a separate
    field rather than a smaller `iteration` because the counter is also the
    honest record of how much work the run has done (AUDIT F45)."""
    try:
        got = int(run.get("iter_ceiling") or MAX_ITERS)
    except (TypeError, ValueError):
        return MAX_ITERS
    return got if got > 0 else MAX_ITERS


def budget_exceeded(run: dict) -> str:
    now = time.time()
    if now >= run.get("deadline", now + 1):
        return "time budget reached"
    if run.get("iteration", 0) >= iter_ceiling(run):
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
        _atomic_write(run_dir(run_id, create=True) / "live.json",
                      json.dumps(live))
    except Exception:
        pass


def load_live(run_id: str) -> dict:
    try:
        return json.loads((run_dir(run_id) / "live.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return {}
