from __future__ import annotations

import platform
import time

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


def _port_holder(port: int) -> str:
    import socket
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return ""
        except OSError:
            pass
    try:
        import psutil
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == "LISTEN" and c.pid:
                return f" (held by pid {c.pid}: {psutil.Process(c.pid).name()})"
    except Exception:
        pass
    return " (holder unknown)"
rag_app = typer.Typer(no_args_is_help=True)
app.add_typer(rag_app, name="rag",
              help="Chat with your documents (Raggity sidecar).")


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
    """Fetch the latest community combo registry from GitHub."""
    from .registry import update_registry

    before = Registry.load()
    dest = update_registry()
    after = Registry.load()
    typer.echo(f"registry updated -> {dest}")
    typer.echo(f"models {len(before.models)} -> {len(after.models)}; "
               f"combos {len(before.combos)} -> {len(after.combos)}")


@app.command()
def doctor():
    """Print detected hardware and registry status."""
    reg = Registry.load()
    p = _profile(reg)
    typer.echo(p.model_dump_json(indent=2))
    typer.echo(f"fingerprint: {p.fingerprint}")
    typer.echo(f"registry: {len(reg.models)} models, {len(reg.combos)} combos")


@app.command()
def plan(use_case: str = typer.Option("general", "--use-case"),
         model: str = typer.Option(None, "--model"),
         explain: bool = typer.Option(False, "--explain")):
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


def _spawn_detached(port: int) -> None:
    """Re-launch `rigma up` as a background process and return the terminal.
    The child re-runs the same resolution (fast — engine/model already on
    disk) but this time stays foreground inside its own detached session."""
    import subprocess
    import sys
    argv = [a for a in sys.argv[1:] if a not in ("--detach", "-d")]
    if "--no-browser" not in argv:
        argv.append("--no-browser")
    if "-y" not in argv and "--yes" not in argv:
        argv.append("--yes")
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
              "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | 0x00000008)   # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    exe = [sys.executable, "-m", "rigma"] if not getattr(sys, "frozen", False) \
        else [sys.executable]
    subprocess.Popen(exe + argv, **kwargs)
    typer.echo(f"Rigma is starting in the background on port {port}.")
    typer.echo(f"  UI:    http://127.0.0.1:{port}")
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


def _stream_chat(port: int, history: list[dict], params: dict | None = None) -> str:
    import json as _json

    import httpx

    text = ""
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"messages": history, "stream": True, **(params or {})},
                      timeout=600) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                delta = _json.loads(payload)["choices"][0]["delta"].get("content")
            except Exception:
                continue
            if delta:
                text += delta
                typer.echo(delta, nl=False)
    typer.echo("")
    return text


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
    typer.echo(f"{s['model']} ({s['quant']}) — session {sess['id']} — "
               f"exit with 'exit' or Ctrl+C")
    while True:
        try:
            q = typer.prompt("you")
        except (typer.Abort, EOFError):
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        sess["messages"].append({"role": "user", "content": q})
        if sess.get("title") == "New chat":
            sess["title"] = q[:40]
        sessions.save(sess)
        try:
            reply = _stream_chat(s["public_port"],
                                 sessions.build_messages(sess, default, preset),
                                 sessions.effective_params(sess, preset,
                                                           model_defaults))
        except Exception as e:
            typer.echo(f"\nmodel unreachable: {e} — check `rigma status`")
            sess["messages"].pop()   # drop the unanswered user turn: a
            sessions.save(sess)      # dangling user msg breaks strict templates
            continue
        sess["messages"].append({"role": "assistant", "content": reply})
        sessions.save(sess)
    if created and not sess["messages"]:
        sessions.delete(sess["id"])


@app.command()
def status():
    """Is Rigma running, and what is it serving?"""
    from . import state as st
    s = st.server_running()
    if s is None:
        typer.echo("not running  (start with: rigma up)")
        raise typer.Exit(0)
    up_min = (time.time() - s["started_at"]) / 60
    typer.echo(f"running: {s['model']} ({s['quant']})  up {up_min:.0f} min")
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

    from . import state as st
    from .bench import run_bench, save_calibration, verdict

    s = st.server_running()
    if s is None:
        typer.echo("not running — start with: rigma up")
        raise typer.Exit(1)
    typer.echo(f"benchmarking {s['model']} ({s['quant']}) ...")
    r = run_bench(s["public_port"], prompt_tokens, gen_tokens)
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
    typer.echo("recorded to ~/.rigma/calibration.json")
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
def stop():
    """Stop the running model server and UI."""
    from . import state as st
    s = st.read_state()
    if s is None:
        typer.echo("not running")
        raise typer.Exit(0)
    for key in ("engine_pid", "ui_pid"):
        if st.pid_alive(int(s.get(key, -1))):
            st.kill_pid(int(s[key]))
    from . import rag as _rag
    _rag.stop_sidecar()
    st.clear_state()
    typer.echo("stopped")


@app.command()
def up(use_case: str = typer.Option("general", "--use-case"),
       model: str = typer.Option(None, "--model"),
       yes: bool = typer.Option(False, "--yes", "-y"),
       dry_run: bool = typer.Option(False, "--dry-run"),
       port: int = typer.Option(11500, "--port"),
       no_browser: bool = typer.Option(False, "--no-browser"),
       ctx: int = typer.Option(None, "--ctx",
                               help="Context size override (clamped to the "
                                    "model's native window)"),
       reasoning: str = typer.Option(None, "--reasoning",
                                     help="Reasoning/thinking: on|off|auto"),
       fa: str = typer.Option(None, "--fa",
                              help="FlashAttention: on|off|auto"),
       spec: str = typer.Option(None, "--spec",
                                help="Speculative decoding: none|draft-mtp|"
                                     "ngram-simple|... (engine-supported)"),
       detach: bool = typer.Option(False, "--detach", "-d",
                                   help="Run in the background; the terminal "
                                        "returns and Rigma keeps serving"),
       ):
    """Start Rigma: probe -> resolve -> download -> serve chat UI."""
    import os
    import webbrowser

    from . import runtime, serve
    from . import state as st

    if st.server_running():
        typer.echo("already running — see: rigma status   (or: rigma stop)")
        raise typer.Exit(1)

    reg = Registry.load()
    p = _profile(reg)
    try:
        rp = resolve(p, reg, use_case=use_case, model_override=model)
    except ResolveError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from None
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
        rp.flags = rp.flags.model_copy(update={"spec_type": spec})
        rp.origin += "+spec-override"
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    typer.echo(f"plan: {rp.model_slug} {rp.gguf.quant} on {rp.backend} "
               f"({rp.origin})")
    typer.echo("argv: llama-server " + " ".join(rp.server_args("<model>", port - 1)))
    if dry_run:
        raise typer.Exit(0)
    if detach:
        _spawn_detached(port)
        raise typer.Exit(0)
    for needed in (port, port - 1):
        holder = _port_holder(needed)
        if holder:
            typer.echo(f"port {needed} is already in use{holder} — "
                       f"free it or pass a different --port")
            raise typer.Exit(1)
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
            if spec_c is not None and spec_c.mmproj is not None:
                mm_path = runtime.ensure_model(spec_c.mmproj)
                extra = ["--mmproj", str(mm_path)]
            typer.echo(f"starting llama-server: {cand.model_slug} "
                       f"{cand.gguf.quant} (first load can take minutes)...")
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
        typer.echo("all fallbacks failed — see logs in ~/.rigma/logs/")
        raise typer.Exit(1)
    st.write_state(rp.model_slug, rp.gguf.quant, port,
                   engine_pid=sp.proc.pid, ui_pid=os.getpid(),
                   backend=rp.backend, use_case=use_case, ctx=rp.flags.ctx)
    typer.echo(f"chat UI:  http://127.0.0.1:{port}")
    typer.echo(f"OpenAI:   http://127.0.0.1:{port}/v1")
    typer.echo("stop:     Ctrl+C here, or `rigma stop` from any terminal")
    if not no_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        serve.run_ui(port, port - 1)
    finally:
        s_end = st.read_state()
        if s_end and st.pid_alive(int(s_end.get("engine_pid", -1))):
            st.kill_pid(int(s_end["engine_pid"]))   # engine may have been switched
        try:
            sp.stop()
        except Exception:
            pass
        st.clear_state()
