"""A5c: the `--refuse` lever had no production caller.

`_verify_plan_or_explain(..., refuse=True)` was reachable only from tests —
no tree declared a `--refuse` option — so the gate that turns "explain the
divergence" into "refuse to launch" could never be pulled by a user. The
LOGIC was right (an independent re-verification confirmed both directions);
only the lever was missing.

These tests pin the lever on BOTH verify paths (`plan` and `up`) and pin that
its ABSENCE keeps today's behaviour: without `--refuse`,
`_verify_plan_or_explain` is called with `refuse=False` — its default — so the
verify path only explains, exactly as before.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()


def _fake_plan():
    return types.SimpleNamespace(
        model_slug="m", backend="rocm", origin="test",
        gguf=types.SimpleNamespace(quant="Q4_K_M", bytes=2 ** 30, file="m.gguf"),
        flags=types.SimpleNamespace(model_dump=lambda: {}),
        explain=["because"])


def _fake_probe(gpu_table, raw_gpus=None):
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows", disk_free_gb=400.0)


@pytest.fixture
def plan_env(monkeypatch):
    """`plan` without touching a real registry or resolver."""
    monkeypatch.setattr(cli.Registry, "load", classmethod(lambda cls: object()))
    monkeypatch.setattr(cli, "_profile", lambda reg: "general")
    monkeypatch.setattr(cli, "resolve", lambda *a, **k: _fake_plan())


@pytest.fixture
def up_env(tmp_path, monkeypatch):
    """`up --model ...` driven to the verify call without loading anything."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    # first-load auto-tune would run the engine; the verify path must not.
    monkeypatch.setenv("RIGMA_AUTO_CALIBRATE", "0")
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    from rigma import runtime
    monkeypatch.setattr(runtime, "ensure_engine",
                        lambda backend, os_name: Path("llama-server.exe"))
    monkeypatch.setattr(runtime, "ensure_model",
                        lambda gguf, **kw: Path(gguf.file))
    return tmp_path


# --- the lever exists and is discoverable ------------------------------------

def test_refuse_is_a_documented_option_on_both_verify_paths():
    """A lever a user cannot find is the whole finding. Both `--verify` paths
    must declare it."""
    for cmd in ("plan", "up"):
        res = runner.invoke(cli.app, [cmd, "--help"])
        assert res.exit_code == 0, res.output
        assert "--refuse" in res.output, f"{cmd} has no --refuse"


# --- the absence of the flag preserves today's behaviour ---------------------

def test_plan_verify_without_refuse_still_only_explains(plan_env, monkeypatch):
    """Byte-for-byte: `--verify` alone calls `_verify_plan_or_explain` with
    `refuse=False`, its default, so the engine's refusal explains and does not
    block."""
    seen = []

    def spy(rp, *, refuse=False):
        seen.append(refuse)

    monkeypatch.setattr(cli, "_verify_plan_or_explain", spy)
    res = runner.invoke(cli.app, ["plan", "--verify"])
    assert res.exit_code == 0, res.output
    assert seen == [False]


def test_up_verify_without_refuse_still_only_explains(up_env, monkeypatch):
    """The `up` half: the flag's absence leaves the launch path exactly as it
    was — verify reports, launch proceeds."""
    seen = []

    def spy(rp, *, refuse=False):
        seen.append(refuse)

    monkeypatch.setattr(cli, "_verify_plan_or_explain", spy)
    from rigma import runtime

    def boom(*a, **k):
        raise RuntimeError("boom: failed to become healthy")

    monkeypatch.setattr(runtime, "launch_server", boom)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--yes", "--verify"])
    assert res.exit_code == 1, res.output
    assert seen and all(v is False for v in seen), seen


# --- the lever is wired through ----------------------------------------------

def test_plan_verify_refuse_passes_the_lever_through(plan_env, monkeypatch):
    seen = []

    def spy(rp, *, refuse=False):
        seen.append(refuse)
        if refuse:
            raise typer.Exit(1)

    monkeypatch.setattr(cli, "_verify_plan_or_explain", spy)
    res = runner.invoke(cli.app, ["plan", "--verify", "--refuse"])
    assert res.exit_code == 1, res.output
    assert seen == [True]


def test_up_verify_refuse_refuses_instead_of_launching(up_env, monkeypatch):
    """`up`'s fallback ladder catches `RuntimeError`, and `typer.Exit` IS one,
    so the refusal must not be swallowed as a failed launch and answered with
    the next model. It must stop the ladder on the FIRST candidate."""
    seen = []

    def spy(rp, *, refuse=False):
        seen.append(refuse)
        if refuse:
            raise cli._VerifyRefused(1)

    monkeypatch.setattr(cli, "_verify_plan_or_explain", spy)
    from rigma import runtime

    def never(*a, **k):
        raise AssertionError("launched despite --refuse")

    monkeypatch.setattr(runtime, "launch_server", never)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--yes",
                                  "--verify", "--refuse"])
    assert res.exit_code == 1, res.output
    assert seen == [True], seen


def test_the_refusal_is_a_typer_exit_but_a_distinct_type(
        monkeypatch, tmp_path):
    """The CLI contract is unchanged — it is still a `typer.Exit` (exit 1) —
    but the type is distinct so the fallback ladder can tell a decision from a
    launch failure."""
    import types

    from rigma import hangar, memtruth

    monkeypatch.setattr(cli, "_engine_server_exe",
                        lambda rp: tmp_path / "llama-server.exe")
    monkeypatch.setattr(hangar, "models_dir", lambda: tmp_path)
    (tmp_path / "m.gguf").write_bytes(b"x")
    primary = types.SimpleNamespace(self_mb=100, model=80, context=20,
                                    compute=0, free=50, total=200)
    stub = types.SimpleNamespace(primary=primary, reason="does not fit",
                                 ok=False, target_mb=None)
    monkeypatch.setattr(memtruth, "verify_plan", lambda *a, **k: (stub, None))
    rp = types.SimpleNamespace(
        backend="rocm", gguf=types.SimpleNamespace(file="m.gguf"))
    with pytest.raises(cli._VerifyRefused) as ei:
        cli._verify_plan_or_explain(rp, refuse=True)
    assert isinstance(ei.value, typer.Exit)


# --- a refuse gate with nothing to gate is the same class of bug -------------

def test_refuse_without_verify_is_refused_not_silently_inert():
    """`--refuse` alone would be a lever that pulls nothing — the exact failure
    A5c is about — so it is a usage error naming the missing flag."""
    for cmd in ("plan", "up"):
        res = runner.invoke(cli.app, [cmd, "--refuse"])
        assert res.exit_code == 2, (cmd, res.output)
        assert "--verify" in res.output, (cmd, res.output)
