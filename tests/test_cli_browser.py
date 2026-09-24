"""The browser opens only once the UI port accepts a connection.

AUDIT F16-2: `webbrowser.open` ran before `serve.run_ui`, so on a cold start the
browser could reach the URL before uvicorn had bound it and show "can't reach
this site". The poll helper is tested against an in-process listener; the `up`
ordering is asserted with a stubbed helper so no browser is ever launched.
"""
import socket

from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()


def _free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_open_when_listening_opens_once_the_port_accepts(monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        t = cli._open_when_listening(port, "http://127.0.0.1:1", timeout=5.0)
        t.join(timeout=5.0)
    finally:
        srv.close()
    assert opened == ["http://127.0.0.1:1"]


def test_open_when_listening_does_not_open_a_dead_port(monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    t = cli._open_when_listening(_free_port(), "http://127.0.0.1:1", timeout=0.4)
    t.join(timeout=5.0)
    assert opened == []


def test_up_defers_the_browser_to_the_listening_poll(tmp_path, monkeypatch):
    """`up` must not call webbrowser.open itself: the helper is started just
    before run_ui and waits for the port."""
    import webbrowser
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    calls = []
    monkeypatch.setattr(cli, "_open_when_listening",
                        lambda port, url, timeout=15.0:
                        calls.append(("poll", port, url)))
    from rigma import serve
    monkeypatch.setattr(serve, "run_ui",
                        lambda *a: calls.append(("run_ui",)))
    port = _free_port()
    res = runner.invoke(cli.app, ["up", "--port", str(port)])
    assert res.exit_code == 0
    assert calls == [("poll", port, f"http://127.0.0.1:{port}"), ("run_ui",)]
    assert opened == []


def test_no_browser_still_skips_the_poll(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli, "_open_when_listening",
                        lambda *a, **k: calls.append(1))
    from rigma import serve
    monkeypatch.setattr(serve, "run_ui", lambda *a: None)
    res = runner.invoke(cli.app, ["up", "--port", str(_free_port()),
                                  "--no-browser"])
    assert res.exit_code == 0 and calls == []
