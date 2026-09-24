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
