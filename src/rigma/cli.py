from __future__ import annotations

import platform
import time

import typer

from .probe import probe_hardware
from .registry import Registry
from .resolve import resolve

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
    rp = resolve(_profile(reg), reg, use_case=use_case, model_override=model)
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


def _stream_chat(port: int, history: list[dict]) -> str:
    import json as _json

    import httpx

    text = ""
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"messages": history, "stream": True},
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
def chat():
    """Chat with the running model in this terminal."""
    from . import state as st
    s = st.server_running()
    if s is None:
        typer.echo("not running — start with: rigma up")
        raise typer.Exit(1)
    typer.echo(f"{s['model']} ({s['quant']}) — exit with 'exit' or Ctrl+C")
    history: list[dict] = []
    while True:
        try:
            q = typer.prompt("you")
        except (typer.Abort, EOFError):
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": q})
        reply = _stream_chat(s["public_port"], history)
        history.append({"role": "assistant", "content": reply})


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
    st.clear_state()
    typer.echo("stopped")


@app.command()
def up(use_case: str = typer.Option("general", "--use-case"),
       model: str = typer.Option(None, "--model"),
       yes: bool = typer.Option(False, "--yes", "-y"),
       dry_run: bool = typer.Option(False, "--dry-run"),
       port: int = typer.Option(11500, "--port"),
       no_browser: bool = typer.Option(False, "--no-browser"),
       turbo: bool = typer.Option(False, "--turbo",
                                  help="Max-speed download (may saturate your connection)")):
    """Start Rigma: probe -> resolve -> download -> serve chat UI."""
    import os
    import webbrowser

    from . import runtime, serve
    from . import state as st

    if st.server_running():
        typer.echo("already running — see: rigma status   (or: rigma stop)")
        raise typer.Exit(1)
    if turbo:
        os.environ["HF_HUB_DISABLE_XET"] = "0"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "16"

    reg = Registry.load()
    p = _profile(reg)
    rp = resolve(p, reg, use_case=use_case, model_override=model)
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    typer.echo(f"plan: {rp.model_slug} {rp.gguf.quant} on {rp.backend} "
               f"({rp.origin})")
    typer.echo("argv: llama-server " + " ".join(rp.server_args("<model>", port - 1)))
    if dry_run:
        raise typer.Exit(0)
    if not yes:
        typer.confirm(
            f"download engine + model ({rp.gguf.bytes / 2**30:.1f} GB)?", abort=True)
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = runtime.ensure_model(rp.gguf)
    typer.echo("starting llama-server (first load can take minutes)...")
    sp = runtime.launch_server(exe, rp, model_path, port=port - 1)
    st.write_state(rp.model_slug, rp.gguf.quant, port,
                   engine_pid=sp.proc.pid, ui_pid=os.getpid())
    typer.echo(f"chat UI:  http://127.0.0.1:{port}")
    typer.echo(f"OpenAI:   http://127.0.0.1:{port}/v1")
    typer.echo("stop:     Ctrl+C here, or `rigma stop` from any terminal")
    if not no_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        serve.run_ui(port, port - 1)
    finally:
        sp.stop()
        st.clear_state()
