"""`rigma up --detach` must validate the port before it claims success, and the
detached child's output must land in a log instead of DEVNULL.

AUDIT F08-5: the parent detached ABOVE the port check, echoed "Rigma is starting
in the background", and exited 0; the child re-ran the same command with
stdout/stderr=DEVNULL, so its "port already in use" / resolve / download error
was written nowhere. No detached process is ever spawned here: the spawn helper
is injected (or the whole helper is stubbed) and only the argv/log construction
and the ordering are asserted.
"""
import os
import socket

from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()


def _fake_probe(gpu_table, raw_gpus=None):
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _free_port() -> int:
    """An ephemeral loopback port the OS just told us was free."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _block_port(port: int = 0):
    """Occupy a loopback port and return (socket, port).

    REC-1: this used to be `_block_port(11594)` / `_block_port(11595)` — literal
    binds shared by every concurrent run, so two runs collided here with an
    OSError. Binding 0 and reading the kernel's choice keeps the assertion (the
    port is in use) with no shared resource. The blocker stays open for the
    caller to close.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s, int(s.getsockname()[1])


def test_detached_argv_drops_detach_and_avoids_prompts(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv",
                        ["rigma", "up", "--detach", "--model", "m"])
    argv = cli._detached_argv()
    assert argv[:3] == [sys.executable, "-m", "rigma"]
    assert argv[3:] == ["up", "--model", "m", "--no-browser", "--yes"]


def test_spawn_detached_sends_the_child_output_to_a_log(tmp_path, monkeypatch):
    import subprocess
    import sys
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rigma", "up", "--detach"])
    calls = []

    def fake_spawn(argv, **kw):
        calls.append((argv, kw))
        kw["stdout"].write("child failed\n")
        return object()

    cli._spawn_detached(11500, spawn=fake_spawn)

    assert len(calls) == 1
    _argv, kw = calls[0]
    assert kw["stderr"] == subprocess.STDOUT      # not DEVNULL
    assert kw["stdout"].name == str(cli._detached_log_path(11500))
    log = cli._detached_log_path(11500)
    assert log.exists() and "child failed" in log.read_text(encoding="utf-8")


def test_detach_checks_the_port_first_ui_only(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda port: spawned.append(port))
    blocker, port = _block_port()
    try:
        res = runner.invoke(cli.app, ["up", "--detach", "--port", str(port)])
    finally:
        blocker.close()
    assert res.exit_code == 1 and "in use" in res.output.lower()
    assert spawned == []


def test_detach_checks_the_port_first_with_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda port: spawned.append(port))
    blocker, port = _block_port()
    try:
        res = runner.invoke(cli.app, ["up", "--detach", "--model",
                                      "qwen3.6-35b-a3b", "--use-case", "coding",
                                      "--port", str(port)])
    finally:
        blocker.close()
    assert res.exit_code == 1 and "in use" in res.output.lower()
    assert spawned == []


def test_detach_spawns_once_the_port_is_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda port: spawned.append(port))
    port = _free_port()  # ephemeral, not the literal 11592 a concurrent run shares
    res = runner.invoke(cli.app, ["up", "--detach", "--port", str(port)])
    assert res.exit_code == 0 and spawned == [port]


def test_detached_log_path_is_under_rigma_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    p = cli._detached_log_path(11500)
    assert p == tmp_path / "logs" / "detached-11500.log"
    assert os.path.dirname(str(p)).endswith("logs")
