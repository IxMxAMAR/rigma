from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import typer

from .probe import probe_hardware
from .registry import Registry
from .resolve import ResolveError, resolve

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context,
          version: bool = typer.Option(False, "--version",
                                       help="Print rigma version and exit")):
    if version:
        from . import __version__
        typer.echo(f"rigma {__version__}")
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


def _listening_pid(port: int):
    """The pid LISTENing on `port` on ANY address, or None.

    AUDIT F16-1: the bind probe in `_port_holder` is address-specific. On
    Windows a bind to 127.0.0.1:port SUCCEEDS while another process already
    listens on 0.0.0.0:port, because the two addresses do not overlap unless
    SO_EXCLUSIVEADDRUSE is set — so the psutil fallback, which only ran after a
    bind FAILURE, never ran in exactly the case it was written for.
    """
    try:
        import psutil
        for c in psutil.net_connections(kind="tcp"):
            if (c.laddr and c.laddr.port == port and c.status == "LISTEN"
                    and c.pid):
                return c.pid
    except Exception:
        pass
    return None


def _port_holder(port: int) -> str:
    import socket
    free = False
    with socket.socket() as s:
        # Windows-only: makes the probe bind conflict with a wildcard listener
        # on the same port instead of silently overlapping it (see above).
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        try:
            s.bind(("127.0.0.1", port))
            free = True
        except OSError:
            pass
    # Consult the OS regardless of how the bind went: a wildcard listener on
    # this port is a conflict even when our specific-address bind succeeded.
    pid = _listening_pid(port)
    if pid is not None:
        try:
            import psutil
            name = psutil.Process(pid).name()
        except Exception:
            name = "unknown"
        return f" (held by pid {pid}: {name})"
    return "" if free else " (holder unknown)"


rag_app = typer.Typer(no_args_is_help=True)
app.add_typer(rag_app, name="rag",
              help="Chat with your documents (Raggity sidecar).")

run_app = typer.Typer(no_args_is_help=True)
app.add_typer(run_app, name="run",
              help="Autonomous long-running jobs (give it a mission, walk away).")

session_app = typer.Typer(no_args_is_help=True)
app.add_typer(session_app, name="session",
              help="Inspect a chat and change what it may do.")


@session_app.command("list")
def session_list(limit: int = typer.Option(20, "--limit",
                                           help="how many chats to show")):
    """Recent chats, with whether each one may run shell commands and code."""
    from . import sessions
    for s in sessions.list_sessions()[:max(1, int(limit))]:
        full = sessions.load(s["id"]) or {}
        flag = "exec" if full.get("confirm_exec") else "    "
        typer.echo(f"{s['id']}  {flag}  {str(s.get('title') or '')[:60]}")


@session_app.command("exec")
def session_exec(
    session_id: str = typer.Argument(..., help="chat id (see: rigma session list)"),
    on: bool = typer.Option(False, "--on", help="allow shell/code in this chat"),
    off: bool = typer.Option(False, "--off",
                             help="revoke it (the default, and the safe one)"),
):
    """Show or change whether one chat may run shell commands and code.

    AUDIT 13-3 made spawning a process its OWN grant: `allow_code` alone no
    longer runs anything, because the destructive-command regex is only an
    advisory text check and a wrapper can decode past it. The product default
    is OFF, so this command is the way to grant it from the CLI. The UI toggle
    is a separate frontend change.
    """
    from . import sessions
    if on and off:
        typer.echo("pick one of --on / --off")
        raise typer.Exit(1)
    s = sessions.load(session_id)
    if s is None:
        typer.echo(f"no such session: {session_id}")
        raise typer.Exit(1)
    if not on and not off:
        typer.echo(f"{session_id}: execution is "
                   f"{'granted' if s.get('confirm_exec') else 'off'}")
        return
    want = bool(on)
    if bool(s.get("confirm_exec")) == want:
        typer.echo(f"{session_id}: execution is already "
                   f"{'granted' if want else 'off'}")
        return
    s["confirm_exec"] = want
    try:
        sessions.save(s, base_rev=s.get(sessions.REV_KEY))
    except sessions.StaleWriteError:
        # Another writer (the browser, or a turn in flight) moved the row on.
        # Writing the stale snapshot anyway would drop their change.
        typer.echo("the session changed while this ran — nothing was written; "
                   "try again")
        raise typer.Exit(1) from None
    typer.echo(f"{session_id}: execution {'granted' if want else 'revoked'}")


memory_app = typer.Typer(no_args_is_help=True)
app.add_typer(memory_app, name="memory",
              help="See, correct and forget what Rigma learned.")


@memory_app.command("list")
def memory_list(kind: str = typer.Option("", "--kind",
                                         help="only this kind of memory"),
                limit: int = typer.Option(50, "--limit")):
    """Every learned rule, most proven first (the order it is injected in)."""
    from .memory import MemoryStore
    from .runtime import rigma_home
    store = MemoryStore(rigma_home() / "memory" / "memories.jsonl")
    rows = store.all()
    if kind:
        rows = [m for m in rows if m.get("kind") == kind]
    rows.sort(key=lambda m: (m.get("outcome_score", 0),
                             m.get("seen_count", 0)), reverse=True)
    if not rows:
        typer.echo("no memories yet"
                   + (f" of kind {kind}" if kind else ""))
        return
    for m in rows[:max(1, int(limit))]:
        typer.echo(f"{str(m.get('id', '?')):<12} {str(m.get('kind', '')):<9} "
                   f"score {int(m.get('outcome_score', 0)):>3} "
                   f"seen {int(m.get('seen_count', 0)):>3}  "
                   f"{str(m.get('text', ''))[:80]}")


@memory_app.command("forget")
def memory_forget(
    memory_id: str = typer.Argument(..., help="id (see: rigma memory list)"),
):
    """Delete one learned rule. Takes the store's lock, like every writer."""
    from .memory import MemoryStore
    from .runtime import rigma_home
    store = MemoryStore(rigma_home() / "memory" / "memories.jsonl")
    if not store.delete(memory_id):
        typer.echo(f"no such memory: {memory_id}")
        raise typer.Exit(1)
    typer.echo(f"forgot {memory_id} ({len(store.all())} left)")


def _run_server_base() -> str:
    from . import state as st
    s = st.read_state()
    if s is None:
        typer.echo("Rigma isn't running — start it first with: rigma up")
        raise typer.Exit(1)
    return f"http://127.0.0.1:{s['public_port']}"


def _active_run_id() -> str:
    import httpx
    r = httpx.get(_run_server_base() + "/api/runs/active", timeout=10).json()
    if not r:
        typer.echo("no active run")
        raise typer.Exit(1)
    return r["id"]


def _response_error(r) -> str:
    """The error sentence out of a run-control response, never a traceback."""
    try:
        body = r.json()
    except Exception:
        return str(r.status_code)
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return str(r.status_code)


def _run_action(rid: str, path: str, ok_message: str,
                payload: dict | None = None) -> None:
    """POST one run-control action and claim success only when it succeeded.

    AUDIT F08-2: stop/pause/resume/steer discarded the response and always
    echoed success, so a 409/500 (the run had already finished, or the id went
    stale between the lookup and the POST) told the user the autonomous run was
    off while it kept iterating and spending budget. In a script, exit 0 was
    taken as proof the run stopped.
    """
    import httpx
    r = httpx.post(_run_server_base() + f"/api/runs/{rid}/{path}",
                   json=payload, timeout=15)
    if r.status_code != 200:
        typer.echo("error: " + _response_error(r))
        raise typer.Exit(1)
    typer.echo(ok_message)


@run_app.command("start")
def run_start(mission: str = typer.Argument(..., help="the job to accomplish"),
              hours: float = typer.Option(8.0, "--hours",
                                          help="time budget (safety cap)"),
              workspace: str = typer.Option("", "--workspace",
                                            help="folder the model works in"),
              profile: str = typer.Option("all", "--profile",
                                          help="all|no-network|no-delete|confined"),
              effort: str = typer.Option("on", "--effort",
                                         help="reasoning: on|auto|off "
                                              "(on = plan each step, best quality)")):
    """Hand the model a mission; it plans, works, logs, and compacts on its own
    until it finishes or a safety cap trips. Survives closing the browser."""
    import httpx
    r = httpx.post(_run_server_base() + "/api/runs",
                   json={"mission": mission, "budget_hours": hours,
                         "workspace": workspace, "profile": profile,
                         "effort": effort}, timeout=30)
    if r.status_code != 200:
        typer.echo("error: " + r.json().get("error", str(r.status_code)))
        raise typer.Exit(1)
    run = r.json()
    typer.echo(f"started run {run['id']}  (budget {hours}h, profile {profile})")
    typer.echo("watch:  rigma run log -f     status: rigma run status")
    typer.echo("steer:  rigma run steer \"…\"   stop:   rigma run stop")


@run_app.command("status")
def run_status():
    """Show the active run's status, progress, and pending plan."""
    import httpx
    r = httpx.get(_run_server_base() + "/api/runs/active", timeout=10).json()
    if not r:
        typer.echo("no active run")
        return
    typer.echo(f"{r['status'].upper()}  iter {r.get('iteration', 0)}  "
               f"errors {r.get('error_streak', 0)}  "
               f"mission: {r.get('mission', '')[:60]}")
    pend = [t for t in r.get("plan", []) if t.get("status") == "pending"]
    if pend:
        typer.echo("pending: " + "; ".join(f"#{t['id']} {t['text']}"
                                           for t in pend[:6]))
    if r.get("log_tail"):
        typer.echo(r["log_tail"])


@run_app.command("stop")
def run_stop():
    """Stop the active run."""
    rid = _active_run_id()
    _run_action(rid, "stop", f"stopped run {rid}")


@run_app.command("pause")
def run_pause():
    """Pause the active run (frees the GPU without losing progress)."""
    rid = _active_run_id()
    _run_action(rid, "pause", "paused — resume with: rigma run resume")


@run_app.command("resume")
def run_resume():
    """Resume a paused run."""
    rid = _active_run_id()
    _run_action(rid, "resume", "resumed")


@run_app.command("steer")
def run_steer(message: str = typer.Argument(..., help="guidance for the model")):
    """Inject guidance used on the next step — course-correct without stopping."""
    rid = _active_run_id()
    _run_action(rid, "inject", "guidance queued for the next step",
                payload={"message": message})


@run_app.command("log")
def run_log(follow: bool = typer.Option(False, "-f", "--follow",
                                        help="stream new entries live")):
    """Print the run's progress log (the model's journal)."""
    import time as _t

    import httpx
    rid = _active_run_id()
    base = _run_server_base()
    seen = 0
    while True:
        log = httpx.get(base + f"/api/runs/{rid}/log", timeout=15).json().get(
            "log", "")
        if len(log) > seen:
            typer.echo(log[seen:], nl=False)
            seen = len(log)
        if not follow:
            break
        try:
            r = httpx.get(base + f"/api/runs/{rid}", timeout=10).json()
            if r.get("status") not in ("running", "paused", "waiting_approval"):
                typer.echo(f"\n[run {r.get('status')}]")
                break
        except Exception:
            pass
        _t.sleep(2)


@rag_app.command("add")
def rag_add(path: str = typer.Argument(..., help="Folder or file to index")):
    """Add a folder to the knowledge base and index it."""
    from pathlib import Path as _P

    from . import rag as _rag

    if not _P(path).exists():
        typer.echo(f"path does not exist: {path}")
        raise typer.Exit(1)
    srcs = _rag.add_source(path)
    typer.echo(f"sources: {len(srcs)}")
    if _rag.raggity_cmd() is None:
        typer.echo("raggity not installed — pip install raggity[server]")
        raise typer.Exit(1)
    typer.echo(_rag.ingest().strip())


@rag_app.command("ask")
def rag_ask(question: str = typer.Argument(...)):
    """Ask a question grounded in your indexed documents."""
    from . import rag as _rag
    from . import state as st

    if st.server_running() is None:
        typer.echo("model not running — start it first: rigma up")
        raise typer.Exit(1)
    _rag.ensure_sidecar()
    a = _rag.ask(question)
    prefix = ("(abstained — not enough evidence in your documents)\n"
              if a.get("abstained") else "")
    typer.echo(prefix + a.get("answer", ""))
    cites = a.get("citations") or []
    if cites:
        typer.echo(f"[{len(cites)} citation(s)]")


@rag_app.command("status")
def rag_status():
    """Sidecar health and indexed sources."""
    from . import rag as _rag

    port = _rag.recorded_sidecar_port()
    h = _rag.sidecar_health(port) if port else None
    if h is None:
        typer.echo("rag sidecar: not running")
    else:
        typer.echo(f"rag sidecar: ok (raggity {h.get('version')}, "
                   f"{h.get('documents')} chunks)")
    for s in _rag.load_sources():
        typer.echo(f"  source: {s}")


@rag_app.command("stop")
def rag_stop():
    """Stop the RAG sidecar."""
    from . import rag as _rag

    typer.echo("stopped" if _rag.stop_sidecar() else "not running")


def _profile(reg: Registry):
    return probe_hardware(reg.gpus)


@app.command()
def update():
    """Fetch the latest community combo registry (and engine pin) from
    GitHub."""
    from .registry import update_registry
    from .runtime import _engines_manifest, update_engines_manifest

    before = Registry.load()
    dest = update_registry()
    after = Registry.load()
    typer.echo(f"registry updated -> {dest}")
    typer.echo(f"models {len(before.models)} -> {len(after.models)}; "
               f"combos {len(before.combos)} -> {len(after.combos)}")
    # engine pin rides the same update: llama.cpp ships weekly, pip doesn't
    old_v = _engines_manifest().get("version", "?")
    if update_engines_manifest():
        new_v = _engines_manifest().get("version", "?")
        typer.echo(f"engine pin: {old_v} -> {new_v}"
                   + ("" if new_v != old_v else " (unchanged)"))
    else:
        typer.echo(f"engine pin: {old_v} (no newer pin published)")


def _port_status(port: int) -> str:
    """Who is LISTENing on `port`, or "" — READ-ONLY.

    Deliberately not `_port_holder`: that one binds the port to probe it, and
    `rigma doctor` must not touch a running server's port at all. This only
    asks the OS who is already there.
    """
    pid = _listening_pid(port)
    if pid is None:
        return ""
    try:
        import psutil
        return f"pid {pid}: {psutil.Process(pid).name()}"
    except Exception:
        return f"pid {pid}"


def _engine_row(backend: str, os_name: str) -> dict:
    """Is the pinned engine for this backend on disk? Never downloads."""
    from . import runtime
    try:
        man = runtime._engines_manifest()
    except Exception as e:
        return {"id": "engine", "state": "warn",
                "detail": f"cannot read the engine pin: {e}",
                "remedy": "run `rigma update` to fetch a fresh pin"}
    key = f"{os_name}/{backend}"
    if key not in (man.get("assets") or {}):
        return {"id": "engine", "state": "warn",
                "detail": f"no pinned engine build for {key}",
                "remedy": "run `rigma plan --explain` and pick a supported "
                          "backend, then `rigma update`"}
    root = runtime.rigma_home() / "engines" / man["version"] / backend
    exe = root / ("llama-server.exe" if os_name == "windows"
                  else "llama-server")
    if exe.exists():
        return {"id": "engine", "state": "ok",
                "detail": f"{backend} engine {man['version']} present",
                "remedy": ""}
    return {"id": "engine", "state": "warn",
            "detail": f"{backend} engine {man['version']} is not downloaded",
            "remedy": "run `rigma up` — it downloads and verifies the engine "
                      "on first run (`rigma up --dry-run` previews it)"}


def _model_present(gguf) -> bool:
    """Is this gguf on disk, in either the managed or the custom models dir?"""
    from . import hangar, runtime
    if (runtime.rigma_home() / "models" / gguf.file).exists():
        return True
    try:
        return (hangar.custom_dir() / gguf.file).exists()
    except Exception:
        return False


def _doctor_rows() -> list[dict]:
    """Every cheap pre-flight check, as {id, state, detail, remedy} rows.

    Never downloads, never binds a port, never starts or stops anything: a
    doctor with side effects is worse than none. `state` is ok / warn / fail,
    and only `fail` is fatal — a missing download is a warn, because `rigma up`
    fetches it.
    """
    from . import state as st
    rows: list[dict] = []
    reg = Registry.load()
    try:
        p = _profile(reg)
    except Exception as e:
        return [{"id": "hardware", "state": "fail",
                 "detail": f"the hardware probe failed: {e}",
                 "remedy": "repair the GPU driver, then run `rigma reprobe`"}]

    # --- GPU backend ---
    if not p.gpus:
        rows.append({"id": "gpu", "state": "fail",
                     "detail": "no GPU detected",
                     "remedy": "install or repair the GPU driver; Rigma needs "
                               "a Vulkan, ROCm or CUDA device"})
        backend = "vulkan"
    else:
        g = p.gpus[0]
        backs = ", ".join(g.backends or []) or "no compute backend"
        rows.append({"id": "gpu",
                     "state": "ok" if g.backends else "fail",
                     "detail": f"{g.name} ({g.slug}) — {backs}",
                     "remedy": "" if g.backends
                     else "update the GPU driver so a Vulkan/ROCm/CUDA "
                          "backend is exposed"})
        backend = (g.backends or ["vulkan"])[0]

    # --- engine binary ---
    rows.append(_engine_row(backend, getattr(p, "os", "windows")))

    # --- model on disk ---
    gguf = None
    size_gb = 0.0
    try:
        rp = resolve(p, reg, use_case="general")
    except ResolveError as e:
        rows.append({"id": "model", "state": "fail",
                     "detail": f"no model fits this machine: {e}",
                     "remedy": "run `rigma plan --explain`; free VRAM/RAM or "
                               "install a smaller model from the Models page"})
    except Exception as e:
        rows.append({"id": "model", "state": "warn",
                     "detail": f"could not resolve a model: {e}",
                     "remedy": "run `rigma plan --explain` for the reason"})
    else:
        gguf, size_gb = rp.gguf, rp.gguf.bytes / 2**30
        if _model_present(rp.gguf):
            rows.append({"id": "model", "state": "ok",
                         "detail": f"{rp.model_slug} ({rp.gguf.quant}, "
                                   f"{size_gb:.1f} GB) is on disk",
                         "remedy": ""})
        else:
            rows.append({"id": "model", "state": "warn",
                         "detail": f"{rp.model_slug} ({size_gb:.1f} GB) is "
                                   "not downloaded yet",
                         "remedy": "run `rigma up` — it downloads and verifies "
                                   "the model on first run"})

    # --- ports (read-only: doctor must not disturb a running server) ---
    port = 11500
    holder = _port_status(port)
    s = st.read_state()
    if holder and s and int(s.get("public_port", 0) or 0) == port:
        rows.append({"id": "port", "state": "ok",
                     "detail": f"Rigma is already running on :{port}",
                     "remedy": ""})
    elif holder:
        rows.append({"id": "port", "state": "warn",
                     "detail": f"port {port} is in use ({holder})",
                     "remedy": f"stop that process, or run `rigma up --port "
                               f"{port + 1}`"})
    else:
        rows.append({"id": "port", "state": "ok",
                     "detail": f"port {port} is free", "remedy": ""})

    # --- disk space ---
    free = float(getattr(p, "disk_free_gb", 0) or 0)
    need = 10.0 + (0.0 if gguf is None or _model_present(gguf) else size_gb)
    if free < 5:
        rows.append({"id": "disk", "state": "fail",
                     "detail": f"only {free:.1f} GB free",
                     "remedy": "free at least 5 GB; the engine plus a model "
                               "will not fit otherwise"})
    elif free < need:
        rows.append({"id": "disk", "state": "warn",
                     "detail": f"{free:.1f} GB free, about {need:.0f} GB "
                               "wanted for the first download",
                     "remedy": "free some space, or install a smaller model "
                               "from the Models page"})
    else:
        rows.append({"id": "disk", "state": "ok",
                     "detail": f"{free:.1f} GB free", "remedy": ""})

    # --- mcp.json parse ---
    from . import mcp_client
    mp = mcp_client.config_path()
    if not mp.exists():
        rows.append({"id": "mcp", "state": "ok",
                     "detail": "no mcp.json (no MCP servers configured)",
                     "remedy": ""})
    else:
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
        except ValueError as e:
            rows.append({"id": "mcp", "state": "warn",
                         "detail": f"mcp.json does not parse: {e}",
                         "remedy": f"fix the JSON at {mp} — Rigma ignores a "
                                   "bad file, so every MCP tool silently "
                                   "disappears"})
        except OSError as e:
            rows.append({"id": "mcp", "state": "warn",
                         "detail": f"cannot read mcp.json: {e}",
                         "remedy": f"check permissions on {mp}"})
        else:
            if not isinstance(raw, dict):
                rows.append({"id": "mcp", "state": "warn",
                             "detail": "mcp.json is not a JSON object",
                             "remedy": f"make {mp} an object with an "
                                       "`mcpServers` key"})
            else:
                servers = raw.get("mcpServers") or {}
                rows.append({"id": "mcp", "state": "ok",
                             "detail": f"{len(servers)} MCP server(s) "
                                       "configured",
                             "remedy": ""})
    return rows


@app.command()
def doctor(as_json: bool = typer.Option(False, "--json",
                                        help="machine-readable rows")):
    """Run every cheap pre-flight check, with the exact remedy for each.

    One line per check. Exits non-zero only on a hard failure: a missing
    download is a warning, not a failure, because `rigma up` fetches it.
    """
    rows = _doctor_rows()
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
    else:
        for r in rows:
            typer.echo(f"{r['state']:<4} {r['id']:<7} {r['detail']}")
            if r.get("remedy"):
                typer.echo(f"     fix: {r['remedy']}")
    if any(r["state"] == "fail" for r in rows):
        raise typer.Exit(1)


@app.command("engine-runtimes")
def engine_runtimes(as_json: bool = typer.Option(False, "--json",
                                                 help="machine-readable rows")):
    """Which ENGINE RUNTIME can serve here: llama.cpp, or vLLM.

    "Backend" means two other things in this tool — the llama.cpp COMPUTE
    backend (vulkan/cuda/rocm/cpu, see `rigma doctor`) and the agent HARNESS
    backend (native/dsh/mcode, see `rigma harness`). This is the third axis:
    WHICH PROGRAM serves the API. llama.cpp is always the default and nothing
    here changes that.

    vLLM cannot run on this machine today (Windows; its stated requirement is
    OS: Linux) and saying so precisely is the point — "not installed" and
    "installed but unsupported here" have different remedies, and on an AMD
    host a wrong Python version installs the CUDA wheel silently. Never
    downloads, never installs, never launches anything.
    """
    from . import server_ops

    rows = server_ops.available_engine_runtimes()
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        state = "available" if r["available"] else f"unavailable ({r['state']})"
        typer.echo(f"{r['engine']:<9} {state}")
        typer.echo(f"          {r['reason']}")


@app.command()
def harness(backend: str = typer.Option(None, "--backend", "-b",
                                        help="check only this one"),
            as_json: bool = typer.Option(False, "--json")):
    """Check each agent backend against the build Rigma was verified with.

    The failure this exists for: an external agent updates underneath Rigma.
    Rigma absorbs an ADDITIVE change to its event schema on purpose, so every
    turn still looks fine — and a RENAMED one degrades quietly, with tool calls
    simply ceasing to appear while the reply still arrives. Nothing in a turn
    would say so. This does.
    """
    # Imported HERE, not at module level: this function is named `harness`, so a
    # module-level `from . import harness` would be shadowed by it.
    from . import harness as seam

    rows = seam.conformance(backend)
    if not rows:
        typer.echo(f"no such backend: {backend}")
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        if not r["installed"]:
            state = "not installed"
        elif r["drift"] is True:
            state = (f"DRIFTED - verified against {r['verified']}, "
                     f"found {r['version']}")
        elif r["drift"] is False:
            state = f"ok ({r['version']})"
        elif r.get("built_in"):
            # Rigma's own loop: shipped with Rigma, so there is no separate build
            # it could have drifted from. "unverified" here made the built-in read
            # as the least trustworthy option on the list, which is backwards.
            state = "built in - this build, nothing to drift from"
        else:
            state = "unverified - no build of this has been checked"
        typer.echo(f"{r['name']:<8} {state}")
    # Drift is the one outcome this command exists to surface, and it used to
    # exit 0 either way — so `rigma harness && ...` was a green light on a
    # backend whose event schema nobody has checked. `unverified` (drift is
    # None) stays a success on purpose: nobody can say anything changed.
    if any(r.get("drift") is True for r in rows):
        raise typer.Exit(1)


@app.command()
def plan(use_case: str = typer.Option("general", "--use-case"),
         model: str = typer.Option(None, "--model"),
         explain: bool = typer.Option(False, "--explain"),
         verify: bool = typer.Option(
             False, "--verify",
             help="Ask the engine itself what the plan will use, instead of "
                  "trusting Rigma's arithmetic")):
    """Show what `rigma up` would run, and why."""
    reg = Registry.load()
    try:
        rp = resolve(_profile(reg), reg, use_case=use_case, model_override=model)
    except ResolveError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from None
    typer.echo(f"model:   {rp.model_slug} ({rp.gguf.quant}, "
               f"{rp.gguf.bytes / 2**30:.1f} GB)")
    typer.echo(f"backend: {rp.backend}   origin: {rp.origin}")
    typer.echo(f"flags:   {rp.flags.model_dump()}")
    if explain:
        for line in rp.explain:
            typer.echo(f"  {line}")
    if verify:
        _verify_plan_or_explain(rp)


def _verify_plan_or_explain(rp, *, refuse: bool = False) -> None:
    """R3-MEM-1: check the plan against the engine's own measurement.

    Rigma's VRAM figure is a formula over GGUF metadata that nothing has ever
    validated. The pinned llama.cpp ships `llama-fit-params`, which does a
    no-alloc dummy load and reports exact per-device accounting against real free
    memory. This runs it and prints both numbers.

    It never blocks a launch: a disagreement is worth telling the user about, but
    the two numbers are not the same quantity, and refusing to start on a
    modelling difference would be worse than starting.
    """
    from . import hangar, memtruth
    from pathlib import Path
    server_exe = _engine_server_exe(rp)
    if server_exe is None:
        typer.echo("verify: the pinned engine is not downloaded "
                   "(`rigma up` fetches it first)")
        return
    # Resolved from the PLAN, not from state: `rigma plan --verify` has to work
    # before anything is running, which is the whole point of a pre-flight check.
    model_path = hangar.models_dir() / rp.gguf.file
    if not Path(model_path).exists():
        typer.echo(f"verify: {rp.gguf.file} is not downloaded yet "
                   "(`rigma up` fetches it first)")
        return
    res, disagree = memtruth.verify_plan(rp, str(model_path), server_exe)
    if res.primary is None:
        typer.echo(f"verify: {res.reason}")
        return
    d = res.primary
    typer.echo(f"verify:  engine measures {d.self_mb} MiB "
               f"(model {d.model} + context {d.context} + compute {d.compute}) "
               f"against {d.free} MiB free of {d.total} MiB")
    typer.echo(f"         engine's own fit verdict: "
               f"{'fits' if res.ok else 'DOES NOT FIT'}"
               + (f" (target margin {res.target_mb} MiB)"
                  if res.target_mb else ""))
    if disagree:
        typer.echo(f"         {disagree}")
    if not res.ok and res.reason:
        typer.echo(f"         {res.reason}")
    if refuse and not res.ok:
        raise typer.Exit(1)


def _engine_server_exe(rp) -> Path | None:
    """The pinned `llama-server` for a plan's backend, or None if not present."""
    from . import runtime
    try:
        man = runtime._engines_manifest()
    except Exception:
        return None
    key = f"{_os_key()}/{rp.backend}"
    if key not in (man.get("assets") or {}):
        return None
    root = runtime.rigma_home() / "engines" / man["version"] / rp.backend
    exe = root / ("llama-server.exe" if _os_key() == "windows"
                  else "llama-server")
    return exe if exe.exists() else None


def _os_key() -> str:
    return {"Windows": "windows", "Linux": "linux",
            "Darwin": "darwin"}.get(platform.system(), "linux")


@app.command()
def models():
    """List registry models and whether they fit this machine."""
    reg = Registry.load()
    p = _profile(reg)
    for slug, spec in sorted(reg.models.items()):
        try:
            rp = resolve(p, reg, model_override=slug)
            fit = (f"fits as {rp.gguf.quant} (n_cpu_moe={rp.flags.n_cpu_moe})"
                   if rp.model_slug == slug else "does not fit")
        except Exception:
            fit = "does not fit"
        typer.echo(f"{slug:24} {spec.kind:5} {fit}")


def _detached_argv() -> list[str]:
    """The command a detached child re-runs: this one without --detach, plus
    --no-browser/--yes so it can never prompt into a closed stdin."""
    import sys
    argv = [a for a in sys.argv[1:] if a not in ("--detach", "-d")]
    if "--no-browser" not in argv:
        argv.append("--no-browser")
    if "-y" not in argv and "--yes" not in argv:
        argv.append("--yes")
    exe = [sys.executable, "-m", "rigma"] if not getattr(sys, "frozen", False) \
        else [sys.executable]
    return exe + argv


def _detached_log_path(port: int):
    """Where a detached child's stdout/stderr goes.

    AUDIT F08-5: it used to be DEVNULL, so the child's "port already in use",
    resolve or download error was written nowhere while the parent had already
    claimed success.
    """
    from .runtime import rigma_home
    return rigma_home() / "logs" / f"detached-{port}.log"


def _spawn_detached(port: int, spawn=None) -> None:
    """Re-launch `rigma up` as a background process and return the terminal.
    The child re-runs the same resolution (fast — engine/model already on
    disk) but this time stays foreground inside its own detached session."""
    import subprocess
    argv = _detached_argv()
    log_path = _detached_log_path(port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if spawn is None:
        spawn = subprocess.Popen
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_f:
        kwargs = {"stdin": subprocess.DEVNULL, "stdout": log_f,
                  "stderr": subprocess.STDOUT}
        if platform.system() == "Windows":
            # DETACHED_PROCESS alone is not enough when the launching shell runs
            # inside a Windows Job Object that kills children on close (many
            # terminals/tools do) — the server dies with the shell. BREAKAWAY_FROM_
            # JOB frees it. Some jobs forbid breakaway, so fall back without it.
            base = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED
            try:
                spawn(argv, creationflags=base | 0x01000000,  # BREAKAWAY_FROM_JOB
                      **kwargs)
            except OSError:
                spawn(argv, creationflags=base, **kwargs)
        else:
            spawn(argv, start_new_session=True, **kwargs)
    typer.echo(f"Rigma is starting in the background on port {port}.")
    typer.echo(f"  UI:    http://127.0.0.1:{port}")
    typer.echo(f"  log:   {log_path}")
    typer.echo("  stop:  rigma stop   ·   status: rigma status")


@app.command(name="list")
def list_local():
    """List models on disk with their size (ollama list parity)."""
    from . import hangar
    out = hangar.list_models()
    any_disk = False
    for m in out["models"]:
        on = [q for q in m["quants"] if q["on_disk"]]
        if m.get("mmproj") and m["mmproj"].get("on_disk"):
            on.append({"quant": "mmproj", "bytes": m["mmproj"]["bytes"]})
        if not on:
            continue
        any_disk = True
        gb = sum(q["bytes"] for q in on) / 2**30
        run = "  ← running" if m["running"] else ""
        tag = "  [custom]" if m["custom"] else ""
        typer.echo(f"{m['slug']:28} {gb:6.1f} GB  "
                   f"{', '.join(q['quant'] for q in on)}{tag}{run}")
    if not any_disk:
        typer.echo("no models downloaded — get one with: rigma up  (or the "
                   "Models tab in the UI)")
    typer.echo(f"\ndisk: {out['disk']['models_gb']} GB models, "
               f"{out['disk']['free_gb']} GB free")


@app.command()
def reprobe(model: str = typer.Argument(
        None, help="Model slug (see `rigma list`); omit for every custom model"),
        offline: bool = typer.Option(
            False, "--offline",
            help="only read files already on disk, never the repo")):
    """Re-read a model's gguf and correct what an older probe got wrong.

    Registry loads heal from files already on disk automatically. This is for
    the case they cannot reach: a model added but never downloaded, still
    carrying the geometry an older probe wrote."""
    from . import hangar
    from .registry import Registry
    slugs = ([model] if model
             else [s for s, m in Registry.load().models.items() if m.custom])
    if not slugs:
        typer.echo("no custom models to re-probe")
        raise typer.Exit(0)
    bad = 0
    for slug in slugs:
        try:
            spec = hangar.reprobe(slug, allow_remote=not offline)
        except hangar.HangarError as e:
            typer.echo(f"{slug:28} {e}")
            bad += 1
            continue
        caps = " ".join(spec.capabilities) or "none"
        typer.echo(f"{slug:28} {spec.full_attn_layers}/{spec.n_layers} layers"
                   + (f" +{spec.mtp_layers} mtp" if spec.mtp_layers else "")
                   + (f", {spec.params / 1e9:.1f}B params" if spec.params else "")
                   + f", caps: {caps}"
                   + ("" if spec.has_template else "  [no chat template]"))
    if bad and len(slugs) == 1:
        raise typer.Exit(1)


@app.command()
def rename(model: str = typer.Argument(..., help="Model slug (see `rigma list`)"),
           new: str = typer.Argument(..., help="what to call it instead")):
    """Rename a custom model, carrying its template and calibration with it.

    A model is keyed by the `general.name` inside its gguf, which some
    quantisers never set — so a real model can land in the library as
    `base-model` with nothing tying it to the repo it came from."""
    from . import hangar
    try:
        spec = hangar.rename_model(model, new)
    except hangar.HangarError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"{model} -> {spec.slug}")


@app.command()
def rm(model: str = typer.Argument(..., help="Model slug (see `rigma list`)"),
       yes: bool = typer.Option(False, "--yes", "-y")):
    """Delete a model's files from disk (ollama rm parity)."""
    from . import hangar
    out = hangar.list_models()
    m = next((x for x in out["models"] if x["slug"] == model), None)
    if m is None:
        typer.echo(f"no such model: {model}  (see `rigma list`)")
        raise typer.Exit(1)
    if m["running"]:
        typer.echo(f"{model} is running — stop or switch first")
        raise typer.Exit(1)
    on = [q for q in m["quants"] if q["on_disk"]]
    if not on:
        typer.echo(f"{model} has no files on disk")
        raise typer.Exit(0)
    gb = sum(q["bytes"] for q in on) / 2**30
    if not yes:
        typer.confirm(f"delete {len(on)} file(s), {gb:.1f} GB, for {model}?",
                      abort=True)
    try:
        if m["custom"]:
            hangar.delete_model(model)
        else:
            for q in on:
                hangar.delete_file(model, q["file"])
            if m.get("mmproj") and m["mmproj"].get("on_disk"):
                hangar.delete_file(model, m["mmproj"]["file"])
    except hangar.HangarError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"deleted {model} ({gb:.1f} GB freed)")


def _engine_error(err) -> str:
    """The human sentence out of an engine error payload.

    llama.cpp sends `{"error": {"message": ...}}` (OpenAI style) and, in older
    builds, a bare string — print whichever it is, never a Python repr.
    """
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)


def _stream_chat(port: int, history: list[dict], params: dict | None = None) -> str:
    import json as _json

    import httpx

    text = ""
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"messages": history, "stream": True, **(params or {})},
                      timeout=600) as r:
        # AUDIT F08-2: the status was never checked and every non-chunk line was
        # skipped, so an error body ("context overflow") and a mid-stream
        # `data: {"error": ...}` chunk ("slot released") both looked exactly like
        # an empty reply — and `chat` saved that empty turn as the answer.
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = _json.loads(payload)
            except ValueError:
                continue          # not a chunk we can read; keep the stream
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                raise RuntimeError(_engine_error(obj["error"]))
            choices = obj.get("choices")
            if not isinstance(choices, list) or not choices:
                # a 200 that is not a chat chunk is still a failed turn; the old
                # code swallowed it and returned "" as a successful reply.
                raise RuntimeError(f"engine returned no choices: {payload[:200]}")
            first = choices[0]
            delta = first.get("delta") if isinstance(first, dict) else None
            content = delta.get("content") if isinstance(delta, dict) else None
            if content:
                text += content
                typer.echo(content, nl=False)
    typer.echo("")
    return text


def _save_chat_turn(sid: str, turn: list, stored: int, first_line: str,
                    tries: int = 3):
    """Graft this turn onto whatever is stored NOW and write it under the guard.

    AUDIT F2: docs/audit-2026-09-04-full.md — `sessions.save` replaces the whole
    row, so writing back a list this terminal read before the generation started
    destroys everything that landed during it. `reload_and_extend` re-reads and
    appends only this turn's own messages; `base_rev` makes a write that lost
    the race fail loudly instead of winning it.

    Returns the saved session, or None (having said why) when the turn could not
    be written — the session was deleted, or another writer kept winning.
    """
    from . import sessions
    for _ in range(tries):
        merged = sessions.reload_and_extend(sid, turn, since=stored)
        if merged is None:
            typer.echo(f"session {sid} was deleted elsewhere — ending this chat")
            return None
        if merged.get("title") == "New chat":
            merged["title"] = first_line[:40]
        try:
            sessions.save(merged, base_rev=merged[sessions.REV_KEY])
            return merged
        except sessions.StaleWriteError:
            continue      # something landed between the reload and the write
    typer.echo("could not save this turn — the session is being written from "
               "somewhere else. Nothing was overwritten; the reply is above.")
    return None


@app.command()
def chat(session: str = typer.Option(None, "--session",
                                     help="Resume a session by id (ids shown in the UI)")):
    """Chat with the running model in this terminal."""
    from . import sessions
    from . import state as st
    s = st.server_running()
    if s is None:
        typer.echo("not running — start with: rigma up")
        raise typer.Exit(1)
    # AUDIT F15-2: the UI-only record (`rigma up` with no model) has model=""
    # and unloaded=True; without this it looked like a running model, so chat
    # opened a session and the first send died with "model unreachable".
    if s.get("unloaded") and not s.get("model"):
        typer.echo(f"Rigma is up, no model loaded — pick one in the Models page "
                   f"(http://127.0.0.1:{s['public_port']})")
        raise typer.Exit(1)
    created = False
    if session:
        sess = sessions.load(session)
        if sess is None:
            typer.echo(f"no such session: {session}")
            raise typer.Exit(1)
    else:
        sess = sessions.create()
        created = True
    if sess.get("use_rag"):
        typer.echo("note: this session has 'use my documents' on — terminal "
                   "chat replies are ungrounded (use `rigma rag ask`)")
    try:
        default = sessions.default_prompt()
    except Exception:
        default = ""
    preset = None
    try:
        from . import presets as _presets
        preset = _presets.resolve(sess.get("preset_id", ""))
    except Exception:
        pass
    model_defaults = {}
    try:
        from .registry import Registry
        model_defaults = Registry.load().models[s["model"]].default_params
    except Exception:
        pass
    sid = sess["id"]
    typer.echo(f"{s['model']} ({s['quant']}) — session {sid} — "
               f"exit with 'exit' or Ctrl+C")
    while True:
        try:
            q = typer.prompt("you")
        except (typer.Abort, EOFError):
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        # AUDIT F2: docs/audit-2026-09-04-full.md — re-read at the top of every
        # turn. This prompt can sit open for hours while the browser writes into
        # the same session, and the loaded-once snapshot took title, system
        # prompt, notes, params, digest and archive down with the messages. The
        # legacy per-chat JSON files froze at the SQLite migration, so there was
        # nothing to recover from. serve.py has reloaded before saving since the
        # same bug cost it minutes; here the window is an evening.
        fresh = sessions.load(sid)
        if fresh is None:
            typer.echo(f"session {sid} was deleted elsewhere — ending this chat")
            break
        sess = fresh
        stored = len(sess["messages"])
        turn = list(sess["messages"]) + [{"role": "user", "content": q}]
        try:
            sending = {**sess, "messages": turn}
            reply = _stream_chat(s["public_port"],
                                 sessions.build_messages(sending, default,
                                                         preset),
                                 sessions.effective_params(sending, preset,
                                                           model_defaults))
        except Exception as e:
            # Nothing has been written yet. The save used to happen BEFORE this
            # call, so an unreachable engine landed the stale snapshot anyway,
            # and the handler then popped the user turn and landed it a second
            # time. A turn that never got an answer is a turn that never was.
            typer.echo(f"\nmodel unreachable: {e} — check `rigma status`")
            continue
        turn.append({"role": "assistant", "content": reply})
        saved = _save_chat_turn(sid, turn, stored, q)
        if saved is None:
            break
        sess = saved
    # AUDIT F2: docs/audit-2026-09-04-full.md — tidying away an unused new chat
    # has to ask the STORE, not `sess`. On the lost-race break above, `sess` is
    # the snapshot this loop read before the turn; deleting on it destroys what
    # the writer that won had just put there — the prose F2 exists to protect.
    if created:
        final = sessions.load(sid)
        if final is not None and not final.get("messages"):
            sessions.delete(sid)


@app.command()
def status():
    """Is Rigma running, and what is it serving?"""
    from . import state as st
    s = st.server_running()
    if s is None:
        typer.echo("not running  (start with: rigma up)")
        raise typer.Exit(0)
    # AUDIT F15-2: "a state record exists" is not "a model is loaded". The
    # UI-only record has model="" and unloaded=True, and printing
    # "running:  ()" for it read as a broken install.
    if s.get("unloaded") and not s.get("model"):
        typer.echo(f"Rigma is up, no model loaded — pick one in the Models page "
                   f"(http://127.0.0.1:{s['public_port']})")
        typer.echo("stop with: rigma stop")
        raise typer.Exit(0)
    up_min = (time.time() - s["started_at"]) / 60
    # R3-VLLM-4: a vLLM launch has NO quant — it serves HF safetensors, not a
    # quantised GGUF — so the old line printed `running: Qwen/Qwen3-8B ()  up 3
    # min`, an empty pair of brackets where the one fact that explains why they
    # are empty belongs. Name the engine instead.
    _eng = str(s.get("engine") or "")
    if _eng == "vllm":
        typer.echo(f"running: {s['model']} (vLLM)  up {up_min:.0f} min")
    elif s.get("quant"):
        typer.echo(f"running: {s['model']} ({s['quant']})  up {up_min:.0f} min")
    else:
        # no engine recorded and no quant: do not assert an engine nobody wrote
        typer.echo(f"running: {s['model']}  up {up_min:.0f} min")
    typer.echo(f"chat UI:  http://127.0.0.1:{s['public_port']}")
    typer.echo(f"OpenAI:   http://127.0.0.1:{s['public_port']}/v1")
    typer.echo("stop with: rigma stop")


@app.command()
def bench(prompt_tokens: int = typer.Option(2048, "--prompt-tokens"),
          gen_tokens: int = typer.Option(128, "--gen-tokens"),
          evidence: str = typer.Option(None, "--evidence",
                                       help="Write registry-format evidence JSON here")):
    """Measure real prefill/generation speed of the running server."""
    import datetime
    import json as _json
    from pathlib import Path

    import httpx

    from . import state as st
    from .bench import calibration_path, run_bench, save_calibration, verdict

    s = st.server_running()
    if s is None:
        typer.echo("not running — start with: rigma up")
        raise typer.Exit(1)
    typer.echo(f"benchmarking {s['model']} ({s['quant']}) ...")
    # AUDIT F08-8: run_bench raises HTTPStatusError/ConnectError, and
    # server_running() returns the UI-only record (unloaded=True) for a Rigma
    # with no engine at all — so this reached the user as a multi-frame httpx
    # traceback instead of "the engine is not answering". RuntimeError covers
    # the no-timings case 08-7 raises.
    try:
        r = run_bench(s["public_port"], prompt_tokens, gen_tokens)
    except (httpx.HTTPError, RuntimeError) as e:
        typer.echo(f"benchmark failed: {e}")
        raise typer.Exit(1) from e
    typer.echo(f"prefill: {r.pp_tps:.0f} t/s   gen: {r.tg_tps:.1f} t/s "
               f"({r.prompt_tokens}-token prompt)")
    reg = Registry.load()
    combo_expected = None
    for c in reg.combos.values():
        if c.model == s["model"] and c.quant == s["quant"] and c.expected:
            combo_expected = c.expected
            break
    typer.echo(verdict(r, combo_expected))
    key = f"{s['model']}:{s['quant']}:{s.get('backend', 'unknown')}"
    save_calibration(key, r.model_dump())
    # AUDIT F15-7: `~` is POSIX shorthand — Explorer and cmd do not expand it.
    # Print the path the file was actually written to.
    typer.echo(f"recorded to {calibration_path()}")
    if evidence:
        from .runtime import _engines_manifest
        payload = {"combo": f"{s['model']} {s['quant']}",
                   "date": datetime.date.today().isoformat(),
                   "llamacpp": _engines_manifest()["version"],
                   "os": platform.system().lower(),
                   "measured": r.model_dump()}
        Path(evidence).parent.mkdir(parents=True, exist_ok=True)
        Path(evidence).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"evidence written -> {evidence}")


@app.command()
def sweep(use_case: str = typer.Option("general", "--use-case"),
          model: str = typer.Option(None, "--model"),
          port: int = typer.Option(11601, "--port",
                                   help="Scratch port for the trial engine — "
                                        "NOT your live server"),
          prompt_tokens: int = typer.Option(2048, "--prompt-tokens"),
          gen_tokens: int = typer.Option(96, "--gen-tokens")):
    """A/B every speedup on THIS machine and save the winner to calibration.

    Launches a throwaway engine on a scratch port (default 11601) and measures
    baseline vs FA-off, quantized KV, big prefill batch, coopmat-off, and (for
    MoE) graphics-queue / lighter offload. The fastest config is written to
    Rigma's calibration store (see `rigma bench`), which `rigma up` then applies
    automatically. Your running server is never touched."""
    from . import runtime
    from .bench import run_sweep

    reg = Registry.load()
    p = _profile(reg)
    try:
        rp = resolve(p, reg, use_case=use_case, model_override=model)
    except ResolveError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from None
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    typer.echo(f"sweeping {rp.model_slug} {rp.gguf.quant} on {rp.backend} "
               f"(scratch engine on :{port}, live server untouched)")
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = runtime.ensure_model(rp.gguf)
    rows = run_sweep(rp, exe, model_path, port=port,
                     prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                     progress=lambda label: typer.echo(f"  trying {label} ..."))
    typer.echo("\n  config             gen t/s   prefill t/s")
    typer.echo("  " + "-" * 40)
    for r in rows:
        mark = "" if r["ok"] else "  (failed)"
        typer.echo(f"  {r['label']:<18} {r['tg_tps']:>6.1f}   "
                   f"{r['pp_tps']:>8.0f}{mark}")
    # ask bench what it crowned rather than recomputing. The old line here used
    # its own rule (fastest row WITH flags, no speculation margin gate), so this
    # could announce a winner that calibration.json does not contain.
    from .bench import crowned_row
    best = crowned_row(rows)
    if best and best["flags"]:
        typer.echo(f"\nwinner: {best['label']} {best['flags']} "
                   f"-> saved to calibration; `rigma up` will use it")
    else:
        typer.echo("\nbaseline stays best — no override saved")


@app.command()
def unload():
    """Stop the engine to free VRAM/RAM. The UI stays up for reload."""
    from . import server_ops
    try:
        s = server_ops.perform_unload()
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"unloaded {s['model']} — VRAM/RAM freed. Reload with "
               f"`rigma load` or from the UI (⚙ → Server).")


@app.command()
def load():
    """Relaunch the model that was unloaded."""
    from . import server_ops
    try:
        s = server_ops.perform_load()
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"loaded {s['model']} ({s['quant']}) at ctx {s.get('ctx', 0)}")


@app.command()
def recalibrate(model: str = typer.Option(None, "--model",
                                          help="Forget one model's tune so it "
                                               "re-optimizes on next load"),
                all: bool = typer.Option(False, "--all",
                                         help="Wipe every stored tune")):
    """Reset or redo hardware tuning. No args re-optimizes the RUNNING model now
    (unload -> quick sweep -> relaunch the fresh winner). The sweep always
    includes baseline, so a re-tune can only match or beat plain defaults."""
    import json as _json

    from .bench import calibration_path, load_calibration, reset_all_calibration
    if all:
        n = reset_all_calibration()
        typer.echo(f"cleared {n} tune(s) — each model re-optimizes on next load")
        return
    if model:
        cal = load_calibration()
        gone = [k for k in list(cal) if k.startswith(model + ":")]
        for k in gone:
            del cal[k]
        calibration_path().write_text(_json.dumps(cal, indent=2), encoding="utf-8")
        typer.echo(f"cleared {len(gone)} tune(s) for {model} — "
                   f"re-optimizes on next load")
        return
    from . import server_ops
    from . import state as st
    if st.read_state() is None:
        typer.echo("not running — load a model first, or pass --model X / --all")
        raise typer.Exit(1)
    typer.echo("re-optimizing the running model (unloads, tunes, relaunches; "
               "takes a few minutes)...")
    try:
        s = server_ops.perform_recalibrate()
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"done — {s['model']} ({s['quant']}) relaunched with the fresh "
               f"config")


@app.command()
def stop():
    """Stop the running model server and UI."""
    from . import state as st
    s = st.read_state()
    if s is None:
        typer.echo("not running")
        raise typer.Exit(0)
    killed = [key for key in ("engine_pid", "ui_pid") if st.kill_recorded(s, key)]
    from . import rag as _rag
    _rag.stop_sidecar()
    st.clear_state()
    # AUDIT F08-1: this used to print "stopped" unconditionally, so a record
    # whose pid had been recycled read as a successful stop while the process
    # it now named was left running — or, before the identity check, killed.
    if killed:
        typer.echo("stopped")
    else:
        # The message was already honest; the EXIT CODE was not. A caller that
        # only reads the status could not tell a stop that happened from a stop
        # that found nothing to stop — `rigma stop && rigma up` walked past it.
        typer.echo("stale state — nothing was killed")
        raise typer.Exit(1)


def _serve_or_exit(port: int) -> None:
    """Serve the UI, turning a lost bind race into the pre-check's own message.

    AUDIT F16-3: `_port_holder` is a check-then-bind (TOCTOU). If another
    process takes the port between that check and uvicorn's bind, `uvicorn.run`
    raises OSError ("address already in use") or SystemExit, and neither the
    command nor the `finally` blocks caught it — typer's default handler printed
    a raw traceback where the pre-check already knew how to say
    "port N is already in use — free it or pass a different --port".
    """
    from . import serve
    try:
        serve.run_ui(port, port - 1)
    except (OSError, SystemExit) as e:
        typer.echo(f"port {port} is already in use — free it or pass a "
                   f"different --port")
        raise typer.Exit(1) from e


def _open_when_listening(port: int, url: str, timeout: float = 180.0):
    """Open `url` in a browser once 127.0.0.1:port accepts a connection.

    AUDIT F16-2: `webbrowser.open` ran BEFORE `serve.run_ui`, so on a cold start
    (uvicorn imports FastAPI, builds the app and binds after the browser already
    had the URL) the first navigation could race the bind and show "can't reach
    this site" — on the very first impression. The wait runs in a daemon thread
    so the caller can start the server immediately; `--no-browser` still skips
    this entirely.

    R3 09-7: the deadline was 15 s and reaching it was SILENT — no browser, no
    message, no retry. Measured: it returned after 0.62 s for a 0.5 s deadline
    with nothing printed, while `up` had already told the user "chat UI: http://…".
    15 s is also shorter than the case this helper exists for: a cold start is
    uvicorn importing FastAPI, building the app, and spawning a 13 GB model load,
    so the automatic open failed exactly when the user was most likely to be
    waiting for it. The deadline now outlasts a real cold start, and reaching it
    says so.

    The thread is a daemon, so Ctrl+C at the terminal ends it; a message written
    from it during interpreter shutdown is a torn line at worst, which is a better
    trade than the silence it replaces.
    """
    import threading

    def _wait() -> None:
        import socket
        import sys
        import time
        import webbrowser
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.1)
        # stderr, so a scripted caller reading stdout is not confused by this
        print(f"the UI did not come up within {timeout:g}s — open {url} yourself",
              file=sys.stderr)

    t = threading.Thread(target=_wait, name="rigma-open-browser", daemon=True)
    t.start()
    return t


def _wants_vision(spec) -> bool:
    """This model's own opinion about its vision projector.

    True unless the model pinned `vision: false` — LaunchDefaults keeps None
    (no opinion) distinct from False (deliberately text-only) precisely so a
    relaunch cannot reload a projector the user turned off.
    """
    return getattr(getattr(spec, "launch", None), "vision", None) is not False


def _launch_extra(cand, reg: Registry, mmproj_path=None) -> list[str]:
    """The flags a launch adds on top of the plan's own args.

    AUDIT F15-3: the vision projector and a repaired chat template used to be
    built inline in the launch loop, so `--dry-run` (which printed only
    `plan.server_args`) omitted them. `mmproj_path` is the downloaded projector
    when the caller already has it; the dry-run preview passes nothing and gets
    the deterministic path `ensure_model` returns.
    """
    from . import runtime
    spec = reg.models.get(cand.model_slug)
    extra: list[str] = []
    if spec is not None and spec.mmproj is not None and _wants_vision(spec):
        extra += ["--mmproj", str(mmproj_path
                                  or runtime.rigma_home() / "models"
                                  / spec.mmproj.file)]
    tmpl = runtime.rigma_home() / "templates" / f"{cand.model_slug}.jinja"
    if tmpl.is_file():
        # repaired-template override, same rule as server_ops.switch_model.
        # Live-verify 2026-07-20 caught the asymmetry: a model whose fixed
        # template sat in ~/.rigma/templates booted through `up` with its
        # BROKEN embedded template, because only the switch path looked.
        extra += ["--chat-template-file", str(tmpl)]
    return extra


def _launch_argv(cand, reg: Registry, port: int, model_label="<model>") -> list[str]:
    """The exact argv `rigma up` will launch — the single preview/launch source."""
    from . import runtime
    return runtime.server_argv("llama-server", cand, model_label, port,
                               _launch_extra(cand, reg))


@app.command()
def up(use_case: str = typer.Option("general", "--use-case"),
       model: str = typer.Option(None, "--model"),
       yes: bool = typer.Option(False, "--yes", "-y"),
       dry_run: bool = typer.Option(False, "--dry-run"),
       verify: bool = typer.Option(
           False, "--verify",
           help="Before launching, ask the engine what the plan will actually "
                "use and compare it with Rigma's own estimate"),
       port: int = typer.Option(11500, "--port"),
       no_browser: bool = typer.Option(False, "--no-browser"),
       ctx: int = typer.Option(None, "--ctx",
                               help="Context size override (clamped to the "
                                    "model's native window)"),
       reasoning: str = typer.Option(None, "--reasoning",
                                     help="Reasoning/thinking: on|off|auto"),
       reasoning_budget: int = typer.Option(
           None, "--reasoning-budget",
           help="Max thinking tokens per turn (0 = end thinking "
                "immediately, -1 = unlimited)"),
       fa: str = typer.Option(None, "--fa",
                              help="FlashAttention: on|off|auto"),
       spec: str = typer.Option(None, "--spec",
                                help="Speculative decoding: none|draft-mtp|"
                                     "ngram-simple|... (engine-supported)"),
       detach: bool = typer.Option(False, "--detach", "-d",
                                   help="Run in the background; the terminal "
                                        "returns and Rigma keeps serving"),
       no_calibrate: bool = typer.Option(False, "--no-calibrate",
                                         help="Skip the one-time first-load "
                                              "hardware auto-tune"),
       engine: str = typer.Option(
           None, "--engine",
           help="Engine runtime: llamacpp (default) or vllm. NOTE this is not "
                "--backend: --backend picks llama.cpp's COMPUTE backend "
                "(vulkan/rocm/cuda/cpu), --engine picks the engine itself. "
                "vLLM is refused with a reason on a machine that cannot run it."),
       ):
    """Start Rigma: probe -> resolve -> download -> serve chat UI."""
    import os

    from . import engines as _engines
    from . import runtime
    from . import state as st

    # R3-VLLM-4: THE ENGINE RUNTIME IS CHOSEN HERE, and refused rather than
    # fallen back from.
    #
    # `detect_engine_runtime` already knew the answer but nothing called it, so
    # `--engine vllm` could not be expressed at all: vLLM was specified,
    # documented and diagnosable, but unreachable. Worse, its designed behaviour
    # is to FALL BACK to llama.cpp with the reason recorded — correct for a stored
    # preference, wrong for a command line. Someone who types `--engine vllm` and
    # gets a working UI has been told vLLM works. So: an explicit request that
    # cannot be honoured is an ERROR, and the engine verdict says what to do
    # instead. A stored preference keeps the fallback behaviour.
    _want = (engine or "").strip().lower()
    if _want and _want not in _engines.ENGINE_RUNTIMES:
        typer.echo(f"unknown engine runtime {engine!r}: must be one of "
                   f"{', '.join(_engines.ENGINE_RUNTIMES)} "
                   f"(this is --engine, not --backend)")
        raise typer.Exit(2)
    _decision = _engines.detect_engine_runtime(_want or None)
    if _want == _engines.VLLM and _decision.runtime != _engines.VLLM:
        typer.echo("vLLM was requested but cannot run on this machine, so "
                   "Rigma will not start llama.cpp in its place:")
        # The verdict's own sentence, not `_decision.reason`: that one is written
        # for the FALLBACK case and reads "…so llama.cpp is used instead", which
        # contradicts a message whose whole point is that it will not be. The
        # availability reason says the same thing without the contradiction.
        _why = (_decision.availability.reason if _decision.availability
                else _decision.reason)
        typer.echo(f"  {_why}")
        typer.echo("  llama.cpp is what works here: `rigma up --engine "
                   "llamacpp`, or drop --engine entirely.")
        raise typer.Exit(1)

    if st.server_running():
        typer.echo("already running — see: rigma status   (or: rigma stop)")
        raise typer.Exit(1)

    reg = Registry.load()

    # UI-only: `rigma up` with no --model just starts Rigma. You pick a model
    # later in the UI (Models tab / Bazaar), which downloads, tunes, and loads
    # it on demand. Pass --model <slug> to load one straight away instead.
    if model is None:
        if dry_run:                              # dry-run never touches ports
            typer.echo(f"would start Rigma (no model) on :{port}")
            raise typer.Exit(0)
        for needed in (port, port - 1):
            holder = _port_holder(needed)
            if holder:
                typer.echo(f"port {needed} is already in use{holder} — "
                           f"free it or pass a different --port")
                raise typer.Exit(1)
        # AUDIT F08-5: detaching used to happen ABOVE the port check, so the
        # parent claimed success (and exited 0) before the child could discover
        # the port was taken — and the child's error went to DEVNULL.
        if detach:
            _spawn_detached(port)
            raise typer.Exit(0)
        st.write_state("", "", port, engine_pid=-1, ui_pid=os.getpid(),
                       backend="", use_case=use_case, ctx=0, unloaded=True)
        typer.echo(f"Rigma:  http://127.0.0.1:{port}")
        typer.echo("        pick a model in the UI (Models tab) — it downloads, "
                   "tunes, and loads on demand")
        typer.echo("stop:   Ctrl+C here, or `rigma stop` from any terminal")
        if not no_browser:
            _open_when_listening(port, f"http://127.0.0.1:{port}")
        try:
            _serve_or_exit(port)
        finally:
            s_end = st.read_state()
            if s_end:
                st.kill_recorded(s_end, "engine_pid")   # AUDIT F08-1: identity-checked
            st.clear_state()
        return

    # R3-VLLM-4: vLLM takes a different launch path, and it has to be entered
    # BEFORE `resolve()`: that resolves a GGUF against the hardware plan, and a
    # vLLM model is a HuggingFace repo id or a safetensors directory with no GGUF
    # and no llama.cpp fit arithmetic at all. `--model` keeps its meaning — for
    # vLLM it is the model vLLM is asked to serve.
    if _decision.runtime == _engines.VLLM:
        _argv = _engines.vllm_argv(
            model, port=port - 1, served_model_name=model,
            max_model_len=ctx, executable="vllm")
        typer.echo(f"starting vllm serve: {model} (first load can take "
                   f"minutes — vLLM profiles memory and captures graphs)...")
        if dry_run:
            typer.echo("would run: " + " ".join(_argv))
            raise typer.Exit(0)
        for needed in (port, port - 1):
            holder = _port_holder(needed)
            if holder:
                typer.echo(f"port {needed} is already in use{holder} — "
                           f"free it or pass a different --port")
                raise typer.Exit(1)
        try:
            sp = _engines.launch_vllm_server(_argv, port - 1)
        except RuntimeError as e:
            typer.echo(str(e))
            raise typer.Exit(1) from None
        # The ENGINE RUNTIME goes in `engine`; `backend` is llama.cpp's compute
        # backend and vLLM does not have one, so it is named rather than guessed.
        # kv_fp stays EMPTY on purpose: it keys llama.cpp's slot cache, and a
        # snapshot must never cross engine runtimes (see the spec).
        st.write_state(model, "", port, engine_pid=sp.proc.pid,
                       ui_pid=os.getpid(), backend="vllm", use_case=use_case,
                       ctx=ctx or 0, gguf="", kv_cache="", kv_fp="",
                       engine=_engines.VLLM)
        typer.echo(f"chat UI:  http://127.0.0.1:{port}")
        typer.echo(f"OpenAI:   http://127.0.0.1:{port}/v1")
        typer.echo("stop:     Ctrl+C here, or `rigma stop` from any terminal")
        if not no_browser:
            _open_when_listening(port, f"http://127.0.0.1:{port}")
        try:
            _serve_or_exit(port)
        finally:
            s_end = st.read_state()
            if s_end:
                st.kill_recorded(s_end, "engine_pid")
            try:
                sp.stop()
            except Exception:
                pass
            st.clear_state()
        return

    # AUDIT F15-1: the hardware probe used to run ABOVE the UI-only branch and
    # be discarded, so the README's "no probing" first command spent up to 20s
    # in a PowerShell GPU query (probe.py) before the UI appeared. It is only
    # needed for the resolve path below.
    p = _profile(reg)
    try:
        rp = resolve(p, reg, use_case=use_case, model_override=model)
    except ResolveError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from None
    # The model's stored configuration, under anything asked for on the command
    # line. `rigma up` resolves directly rather than going through
    # perform_switch, so without this a pinned default would apply to the UI's
    # load button and silently not to the CLI.
    _launch = getattr(reg.models.get(rp.model_slug), "launch", None)
    _vision = _wants_vision(reg.models.get(rp.model_slug))
    if _launch is not None:
        _d = _launch.as_overrides()
        if "quant" in _d:
            from .server_ops import _model_on_disk
            want = _d["quant"].strip().lower()
            pick = next((g for g in reg.models[rp.model_slug].ggufs
                         if g.quant.lower() == want), None)
            if pick is not None and _model_on_disk(pick):
                rp.gguf = pick
                rp.origin += "+default-quant"
        if "backend" in _d and _d["backend"] != rp.backend:
            _avail = p.primary_gpu.backends if p.primary_gpu else []
            if _d["backend"] not in _avail:
                typer.echo(f"{rp.model_slug} pins the {_d['backend']} backend, "
                           f"which this GPU does not offer "
                           f"({', '.join(_avail) or 'none'})")
                raise typer.Exit(1)
            rp.backend = _d["backend"]
            rp.origin += "+default-backend"
        _upd = {}
        if ctx is None and "ctx" in _d:
            _upd["ctx"] = _d["ctx"]
        if "kv" in _d:
            _upd["cache_type_k"] = _upd["cache_type_v"] = _d["kv"]
        if spec is None and "spec_type" in _d:
            # only when THIS file carries the head — see the --spec branch below
            from .hangar import file_has_mtp
            if _d["spec_type"] != "draft-mtp" or file_has_mtp(rp.gguf) is True:
                _upd["spec_type"] = _d["spec_type"]
                _upd["spec_n_max"] = _d.get("spec_n_max") or 1
        if _upd:
            rp.flags = rp.flags.model_copy(update=_upd)
            rp.origin += "+model-default"
    if ctx is not None:
        native = reg.models[rp.model_slug].native_ctx
        rp.flags = rp.flags.model_copy(update={"ctx": max(1024, min(ctx, native))})
        rp.origin += "+ctx-override"
    if reasoning is not None:
        if reasoning not in ("on", "off", "auto"):
            typer.echo("--reasoning must be on, off, or auto")
            raise typer.Exit(2)
        rp.flags = rp.flags.model_copy(update={"reasoning": reasoning})
        rp.origin += "+reasoning-override"
    if reasoning_budget is not None:
        rp.flags = rp.flags.model_copy(
            update={"reasoning_budget": reasoning_budget})
        rp.origin += "+rbudget-override"
    if fa is not None:
        if fa not in ("on", "off", "auto"):
            typer.echo("--fa must be on, off, or auto")
            raise typer.Exit(2)
        rp.flags = rp.flags.model_copy(update={"flash_attn": fa})
        rp.origin += "+fa-override"
    if spec is not None:
        allowed = ("none", "draft-simple", "draft-eagle3", "draft-mtp",
                   "draft-dflash", "ngram-simple", "ngram-map-k",
                   "ngram-map-k4v", "ngram-mod", "ngram-cache")
        if spec not in allowed:
            typer.echo(f"--spec must be one of: {', '.join(allowed)}")
            raise typer.Exit(2)
        if spec == "draft-mtp":
            # spec-decode without the MTP tensors is a documented Vulkan
            # driver-reset loop — refuse unless THIS gguf actually carries them.
            # The model's capability list is the wrong thing to ask: MTP
            # survives or is dropped per artefact, so a repo can advertise it
            # and still hand you a quant without the tensors. The file is on
            # disk by the time we launch, so the real answer is always available.
            from .hangar import file_has_mtp
            has = file_has_mtp(rp.gguf)
            if has is not True:
                why = ("carries no MTP tensors" if has is False
                       else "has not been downloaded, so its MTP tensors "
                            "cannot be verified")
                typer.echo(f"{rp.gguf.file} {why} — draft-mtp would reset the "
                           "GPU driver rather than fail cleanly. Refusing.")
                raise typer.Exit(2)
        rp.flags = rp.flags.model_copy(update={"spec_type": spec})
        rp.origin += "+spec-override"
    # Re-fit against what will ACTUALLY be resident: no projector when vision is
    # off, plus the draft cache when speculation is on. Otherwise the plan
    # reserves 600MB for a projector it will not load and nothing for a draft
    # cache it will, and lands on a needless offload.
    # AUDIT F18: docs/audit-2026-09-04-full.md — this sits BELOW the overrides
    # now. --ctx and --spec both move the answer, and running it above them
    # meant `rigma up --spec ...` budgeted a draft cache of the wrong size, or
    # none at all.
    _spec_r = reg.models.get(rp.model_slug)
    if _spec_r is not None:
        from .server_ops import launch_fit_spec as _fit_spec
        _spec2, _differs = _fit_spec(_spec_r, rp.flags, vision=_vision)
        if _launch is not None or _differs:
            from .resolve import fit_gguf as _fit
            _fl = _fit(_spec2, rp.gguf, p, rp.flags.ctx, [])
            if _fl is not None:
                rp.flags = rp.flags.model_copy(update={
                    "ngl": _fl.ngl, "n_cpu_moe": _fl.n_cpu_moe})
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    typer.echo(f"plan: {rp.model_slug} {rp.gguf.quant} on {rp.backend} "
               f"({rp.origin})")
    # AUDIT F15-3: this prints the SAME argv the launch below passes to
    # launch_server, including --mmproj/--chat-template-file/--slot-save-path.
    typer.echo("argv: " + " ".join(_launch_argv(rp, reg, port - 1)))
    if dry_run:
        raise typer.Exit(0)
    for needed in (port, port - 1):
        holder = _port_holder(needed)
        if holder:
            typer.echo(f"port {needed} is already in use{holder} — "
                       f"free it or pass a different --port")
            raise typer.Exit(1)
    # AUDIT F08-5: below the port check, so the parent only claims success once
    # the port it is about to use has been validated.
    if detach:
        _spawn_detached(port)
        raise typer.Exit(0)
    if not yes:
        typer.confirm(
            f"download engine + model ({rp.gguf.bytes / 2**30:.1f} GB)?", abort=True)
    from .resolve import fallback_plans
    candidates = [rp, *fallback_plans(rp, reg, p)]
    sp = None
    for i, cand in enumerate(candidates):
        try:
            exe = runtime.ensure_engine(cand.backend, os_name)
            model_path = runtime.ensure_model(cand.gguf)
            extra = []
            spec_c = reg.models.get(cand.model_slug)
            # AUDIT F18: docs/audit-2026-09-04-full.md — the plan frees the
            # projector's memory and hands it to more GPU layers, so attaching
            # it here anyway overcommits the card (and `ensure_model` DOWNLOADS
            # it first). Asked per candidate: a fallback is a different model
            # with its own opinion. AUDIT F15-3: the flag list itself is built by
            # the shared `_launch_extra`, the same helper the preview uses.
            mm_path = None
            if (spec_c is not None and spec_c.mmproj is not None
                    and _wants_vision(spec_c)):
                mm_path = runtime.ensure_model(spec_c.mmproj)
            extra = _launch_extra(cand, reg, mm_path)
            from .bench import auto_calibrate, is_calibrated
            if (ctx is None and not no_calibrate and cand.backend != "cpu"
                    and os.environ.get("RIGMA_AUTO_CALIBRATE", "1") != "0"
                    and not is_calibrated(cand.model_slug, cand.gguf.quant,
                                          cand.backend)):
                typer.echo(f"tuning {cand.model_slug} for your hardware "
                           f"(one-time, a few minutes)...")
                cand = auto_calibrate(
                    cand, exe, model_path, port=port - 1, extra_args=extra or None,
                    progress=lambda lbl: typer.echo(f"  trying {lbl} ..."))
            typer.echo(f"starting llama-server: {cand.model_slug} "
                       f"{cand.gguf.quant} (first load can take minutes)...")
            # R3-MEM-1: the pre-flight check, HERE because this is the first point
            # where both the engine binary and the model file exist — the oracle
            # needs the real GGUF and cannot run before `ensure_model` fetched it.
            # Deliberately placed BEFORE launch_server and deliberately never
            # blocking: it reports, and only a plan that provably does not fit is
            # worth refusing, which the caller opts into with --verify.
            if verify:
                _verify_plan_or_explain(cand)
            sp = runtime.launch_server(exe, cand, model_path, port=port - 1,
                                       extra_args=extra or None)
            rp = cand
            break
        except RuntimeError as e:
            typer.echo(str(e).splitlines()[0])
            if i + 1 < len(candidates):
                nxt = candidates[i + 1]
                typer.echo(f"falling back -> {nxt.model_slug} {nxt.gguf.quant} "
                           f"({nxt.origin})")
    if sp is None:
        # AUDIT F15-7: `~` is POSIX shorthand — Explorer and cmd do not expand
        # it, so the one message that says where the failure was logged named a
        # path the user could not open. Print the directory actually written.
        typer.echo(f"all fallbacks failed — see logs in "
                   f"{runtime.rigma_home() / 'logs'}")
        raise typer.Exit(1)
    # Both caches are keyed by `kv_fp`, and an EMPTY one disables them silently
    # rather than loudly: `serve._prefix_ctx` returns None on a falsy fp, so
    # `_prefix_warm`/`_prefix_snapshot` become no-ops, and `perform_unload` skips
    # its KV save for the same reason. `rigma up` launches through here, so
    # omitting this turned off prefix reuse AND restore-on-unload for every
    # CLI-started run — the whole reason a long conversation re-prefilled from
    # zero. `write_state` reverts unnamed fields to their defaults by design, so
    # the launch path has to name it.
    from . import kvcache as _kvcache
    st.write_state(rp.model_slug, rp.gguf.quant, port,
                   engine_pid=sp.proc.pid, ui_pid=os.getpid(),
                   backend=rp.backend, use_case=use_case, ctx=rp.flags.ctx,
                   gguf=rp.gguf.file,
                   kv_cache=rp.flags.cache_type_k or "",
                   kv_fp=_kvcache.launch_fingerprint(rp, exe),
                   engine=_engines.LLAMACPP,
                   # a projector this launch left off must stay off: perform_switch
                   # reads no_vision back when the caller has no opinion, and a
                   # ctx change from the UI would otherwise reload it
                   no_vision=not _wants_vision(reg.models.get(rp.model_slug)))
    typer.echo(f"chat UI:  http://127.0.0.1:{port}")
    typer.echo(f"OpenAI:   http://127.0.0.1:{port}/v1")
    typer.echo("stop:     Ctrl+C here, or `rigma stop` from any terminal")
    if not no_browser:
        _open_when_listening(port, f"http://127.0.0.1:{port}")
    try:
        _serve_or_exit(port)
    finally:
        s_end = st.read_state()
        if s_end:
            # AUDIT F08-1: identity-checked — the engine may have been switched
            st.kill_recorded(s_end, "engine_pid")
        try:
            sp.stop()
        except Exception:
            pass
        st.clear_state()
