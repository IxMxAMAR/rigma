"""A lost bind race reports the port, not a traceback.

AUDIT F16-3: the pre-check (`_port_holder`) and uvicorn's bind are separate, so
another process can take the port in between. `uvicorn.run` then raises OSError
("address already in use") or SystemExit, and neither `up` nor its `finally`
caught it.
"""
import socket

import pytest
import typer
from typer.testing import CliRunner

import rigma.cli as cli
from rigma import serve

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_hang_on_run_ui():
    """Override the conftest guard for THIS module only.

    The guard replaces `serve.run_ui` with a function that raises, so no test can
    accidentally serve. The last two tests below deliberately execute the REAL
    `run_ui` — with `build_app` and `uvicorn.run` patched, so nothing binds —
    because the question they answer is "does uvicorn get the host?", and that is
    precisely the line the guard hides. Every other test here patches `run_ui`
    itself, so nothing in this module can block.
    """
    yield


def _free_port():
    """A port whose predecessor is free too: `up` pre-checks port-1 for the
    engine, and an ephemeral port's predecessor may be occupied."""
    for _ in range(50):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        if port > 1024 and cli._port_holder(port) == "" and \
                cli._port_holder(port - 1) == "":
            return port
    raise AssertionError("no free port pair found")


@pytest.mark.parametrize("exc", [OSError("address already in use"),
                                 SystemExit(1)])
def test_up_reports_a_lost_bind_race(tmp_path, monkeypatch, exc):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    def boom(*a):
        raise exc

    monkeypatch.setattr(serve, "run_ui", boom)
    port = _free_port()
    res = runner.invoke(cli.app, ["up", "--port", str(port), "--no-browser"])
    assert res.exit_code == 1
    assert f"port {port} is already in use" in res.output
    assert "different --port" in res.output


def test_serve_or_exit_returns_normally_when_serving_succeeds(monkeypatch):
    monkeypatch.setattr(serve, "run_ui", lambda *a: None)
    cli._serve_or_exit(11500)


def test_serve_or_exit_exits_one_on_a_bind_failure(monkeypatch):
    def boom(*a):
        raise OSError("address already in use")

    monkeypatch.setattr(serve, "run_ui", boom)
    with pytest.raises(typer.Exit) as ei:
        cli._serve_or_exit(11500)
    assert ei.value.exit_code == 1


# --- `--host`: what a container needs, and what a desktop must not get -------
#
# Runpod's proxy cannot reach 127.0.0.1, so a pod needs the UI on 0.0.0.0, and
# the platform's own pod workflow is "bind 0.0.0.0 and declare the port"
# (runpod-usage/reference/pod-workflows.md). Without the flag the pod could only
# be reached through a raw-TCP socat bridge on a second port, which is what
# deploy/runpod/ did before this existed.


def test_serve_or_exit_forwards_the_host(monkeypatch):
    seen = []
    monkeypatch.setattr(serve, "run_ui", lambda *a: seen.append(a))
    cli._serve_or_exit(11500, "0.0.0.0")
    assert seen == [(11500, 11499, "0.0.0.0")]


def test_up_forwards_host_to_the_ui(tmp_path, monkeypatch):
    """The end-to-end wiring, on the path a FRESH pod takes. A host that stops
    at the CLI is a pod nobody can open."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    seen = []

    def record_then_fail(*a):
        seen.append(a)
        raise OSError("address already in use")

    monkeypatch.setattr(serve, "run_ui", record_then_fail)
    port = _free_port()
    res = runner.invoke(cli.app, ["up", "--port", str(port), "--no-browser",
                                  "--host", "0.0.0.0"])
    assert res.exit_code == 1, res.output
    assert seen and seen[0][2] == "0.0.0.0"


def test_run_ui_binds_the_host_it_is_given(monkeypatch):
    """The last hop: uvicorn must receive it, not a hardcoded loopback."""
    import uvicorn

    got = {}
    monkeypatch.setattr(serve, "build_app", lambda upstream: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: got.update(kw))
    serve.run_ui(11500, 11499, "0.0.0.0")
    assert got == {"host": "0.0.0.0", "port": 11500, "log_level": "warning"}


def test_run_ui_still_defaults_to_loopback(monkeypatch):
    """A desktop must not start listening on the LAN because a container
    needed to. The default is the promise; the flag is the exception."""
    import uvicorn

    got = {}
    monkeypatch.setattr(serve, "build_app", lambda upstream: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: got.update(kw))
    serve.run_ui(11500, 11499)
    assert got["host"] == "127.0.0.1"
