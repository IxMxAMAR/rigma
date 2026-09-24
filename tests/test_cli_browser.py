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


# R3 09-7: the poll gave up silently. `rigma up` printed "chat UI: http://…" and
# then no browser ever appeared and nothing said why. The measurement was a 0.62 s
# return for a 0.5 s deadline with `opened == []` and no output.
#
# The reported fix was one print in the thread, deferred because "it writes to
# stderr from a daemon thread that can outlive the terminal's Ctrl+C". That risk is
# real but it is not the interesting half: 15 s is simply too short for the case
# this helper exists for. A cold start is uvicorn importing FastAPI, building the
# app, and spawning a 13 GB model load — the exact slow first run F16-2 was written
# about. Giving up at 15 s means the automatic open fails precisely when the user is
# most likely to be waiting, and it is not retried. So the deadline is now long
# enough to outlast a real cold start, and reaching it says so.


def test_open_when_listening_says_so_when_it_gives_up(monkeypatch, capsys):
    """Silence is the bug. If the poll does give up, it must say why."""
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    t = cli._open_when_listening(_free_port(), "http://127.0.0.1:1", timeout=0.3)
    t.join(timeout=5.0)
    err = capsys.readouterr().err
    assert opened == []
    assert "did not come up" in err and "http://127.0.0.1:1" in err


def test_open_when_listening_keeps_waiting_past_the_old_15s_deadline(monkeypatch):
    """The regression this fixes: 15 s is shorter than a real cold start.

    A cold start is uvicorn importing FastAPI, building the app, and spawning a
    13 GB model load — the exact slow first run F16-2 was written about. So the
    DEFAULT deadline must outlast it, and both call sites must take the default
    rather than passing a short one. Pinned as a contract, because the failure mode
    is a number nobody looks at again.

    (An earlier version of this test drove a fake clock past 15 s and passed
    against the UNFIXED code, because it passed `timeout=120.0` itself — it was
    asserting its own argument. The thing worth pinning is the default.)
    """
    import inspect
    sig = inspect.signature(cli._open_when_listening)
    assert sig.parameters["timeout"].default >= 120.0, (
        "the browser poll's default deadline is shorter than a cold start")

    src = inspect.getsource(cli)
    callers = [ln.strip() for ln in src.splitlines()
               if "_open_when_listening(" in ln and "def " not in ln
               and "monkeypatch" not in ln]
    assert callers, "no call site found — did the helper get renamed?"
    for ln in callers:
        assert "timeout" not in ln, (
            f"call site overrides the deadline, reintroducing 09-7: {ln}")


