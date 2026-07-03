from __future__ import annotations

import platform

import typer

from .probe import probe_hardware
from .registry import Registry
from .resolve import resolve

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _profile(reg: Registry):
    return probe_hardware(reg.gpus)


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


@app.command()
def up(use_case: str = typer.Option("general", "--use-case"),
       model: str = typer.Option(None, "--model"),
       yes: bool = typer.Option(False, "--yes", "-y"),
       dry_run: bool = typer.Option(False, "--dry-run"),
       port: int = typer.Option(11500, "--port")):
    """Probe -> resolve -> download -> serve."""
    from . import runtime  # local import: keeps --dry-run path light

    reg = Registry.load()
    p = _profile(reg)
    rp = resolve(p, reg, use_case=use_case, model_override=model)
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    argv_preview = rp.server_args("<model>", port)
    typer.echo(f"plan: {rp.model_slug} {rp.gguf.quant} on {rp.backend} "
               f"({rp.origin})")
    typer.echo("argv: llama-server " + " ".join(argv_preview))
    if dry_run:
        raise typer.Exit(0)
    if not yes:
        typer.confirm(
            f"download engine + model ({rp.gguf.bytes / 2**30:.1f} GB)?", abort=True)
    exe = runtime.ensure_engine(rp.backend, os_name)
    model_path = runtime.ensure_model(rp.gguf)
    typer.echo("starting llama-server (first load can take minutes)...")
    sp = runtime.launch_server(exe, rp, model_path, port=port)
    typer.echo(f"ready: OpenAI-compatible endpoint at {sp.url}/v1")
    typer.echo("Ctrl+C to stop.")
    try:
        sp.proc.wait()
    except KeyboardInterrupt:
        sp.stop()
