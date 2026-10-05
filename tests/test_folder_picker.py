"""The native folder picker: return-code mapping, and the endpoint's contract.

No test here shows a dialog. `subprocess.run` is stubbed for the mapping and
`folder_picker.pick_folder` is stubbed for the route, so a CI machine with no
desktop (and no Windows) still covers every branch the caller can see.
"""
import subprocess

import pytest

from rigma import folder_picker


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _fake_run(result, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append((cmd, kw))
        if isinstance(result, Exception):
            raise result
        return result
    return run


def test_a_chosen_folder_comes_back_as_a_path(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Windows")
    seen = []
    monkeypatch.setattr(folder_picker.subprocess, "run",
                        _fake_run(_Proc(stdout="OK:D:\\Writing\r\n"), seen))
    got = folder_picker.pick_folder("C:\\start")
    assert got == folder_picker.Picked(path="D:\\Writing")
    # -STA is what the COM dialog needs; the start folder rides in the env, not
    # in the command line, so a path with spaces or quotes cannot break it.
    cmd, kw = seen[0]
    assert "-STA" in cmd
    assert kw["env"]["RIGMA_PICK_INITIAL"] == "C:\\start"
    assert "C:\\start" not in " ".join(cmd)


def test_closing_the_dialog_is_a_cancel_not_a_failure(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Windows")
    monkeypatch.setattr(folder_picker.subprocess, "run",
                        _fake_run(_Proc(returncode=3)))
    got = folder_picker.pick_folder()
    assert got.cancelled is True
    assert got.path == "" and got.reason == ""


def test_a_powershell_failure_is_reported_with_its_own_words(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Windows")
    monkeypatch.setattr(folder_picker.subprocess, "run",
                        _fake_run(_Proc(stderr="Add-Type : cannot find type\n",
                                        returncode=1)))
    got = folder_picker.pick_folder()
    assert got.cancelled is False and got.path == ""
    assert "Add-Type" in got.reason


def test_an_unanswered_dialog_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Windows")
    monkeypatch.setattr(folder_picker.subprocess, "run",
                        _fake_run(subprocess.TimeoutExpired("powershell", 5)))
    got = folder_picker.pick_folder(timeout=5)
    assert got.cancelled is False
    assert "not answered" in got.reason


def test_a_platform_without_explorer_is_told_so(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Linux")
    called = []
    monkeypatch.setattr(folder_picker.subprocess, "run",
                        _fake_run(_Proc(), called))
    got = folder_picker.pick_folder()
    assert got.reason and called == []          # never even tries to spawn


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    import rigma.serve as srv
    from fastapi.testclient import TestClient
    return TestClient(srv.build_app(upstream_port=1)), srv


def _picked(monkeypatch, srv, result, seen=None):
    def fake(initial="", **kw):
        if seen is not None:
            seen["initial"] = initial
        return result
    monkeypatch.setattr(srv.folder_picker, "pick_folder", fake)


def test_the_pick_route_returns_the_chosen_path_without_saving_it(client,
                                                                 monkeypatch):
    c, srv = client
    sid = c.post("/api/sessions", json={}).json()["id"]
    seen = {}
    _picked(monkeypatch, srv, folder_picker.Picked(path="D:\\Writing"), seen)
    r = c.post(f"/api/sessions/{sid}/workspace/pick", json={"initial": "C:\\go"})
    assert r.status_code == 200 and r.json() == {"path": "D:\\Writing"}
    assert seen["initial"] == "C:\\go"
    # the route only REPORTS: the session still has no workspace until the
    # caller saves it through POST /api/sessions/{sid}
    assert c.get(f"/api/sessions/{sid}").json()["workspace"] == ""


def test_the_pick_route_distinguishes_cancel_from_no_dialog(client, monkeypatch):
    c, srv = client
    sid = c.post("/api/sessions", json={}).json()["id"]
    _picked(monkeypatch, srv, folder_picker.Picked(cancelled=True))
    assert c.post(f"/api/sessions/{sid}/workspace/pick", json={}).status_code \
        == 204
    _picked(monkeypatch, srv, folder_picker.Picked(
        reason="the server has no interactive desktop"))
    r = c.post(f"/api/sessions/{sid}/workspace/pick", json={})
    assert r.status_code == 503
    assert r.json()["error"] == "the server has no interactive desktop"


def test_the_pick_route_404s_an_unknown_session(client):
    c, _ = client
    assert c.post("/api/sessions/nope/workspace/pick",
                  json={}).status_code == 404


def test_the_pick_route_defaults_the_dialog_to_the_recorded_workspace(
        client, tmp_path, monkeypatch):
    c, srv = client
    sid = c.post("/api/sessions", json={}).json()["id"]
    ws = tmp_path / "ws"
    ws.mkdir()
    c.post(f"/api/sessions/{sid}", json={"workspace": str(ws)})
    seen = {}
    _picked(monkeypatch, srv, folder_picker.Picked(cancelled=True), seen)
    c.post(f"/api/sessions/{sid}/workspace/pick", json={})
    assert seen["initial"] == str(ws)


def test_the_route_runs_the_dialog_off_the_event_loop(tmp_path, monkeypatch):
    """A modal dialog blocks for as long as a human takes, and this server runs
    one loop for every live stream, so the handler must hand it to a thread.
    `srv.asyncio` is shimmed rather than patched, so only serve's reference is
    replaced and the real module keeps working for everything else."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    import asyncio
    import rigma.serve as srv
    from fastapi.testclient import TestClient

    class _Shim:
        def __init__(self, real):
            self._real, self.calls = real, []

        async def to_thread(self, fn, *a, **k):
            self.calls.append(fn)
            return await self._real.to_thread(fn, *a, **k)

    shim = _Shim(asyncio)
    monkeypatch.setattr(srv, "asyncio", shim)
    monkeypatch.setattr(srv.folder_picker, "pick_folder",
                        lambda *a, **k: folder_picker.Picked(cancelled=True))
    c = TestClient(srv.build_app(upstream_port=1))
    sid = c.post("/api/sessions", json={}).json()["id"]
    assert c.post(f"/api/sessions/{sid}/workspace/pick",
                  json={}).status_code == 204
    # serve also threads `sessions.create`, so assert membership, not equality
    assert srv.folder_picker.pick_folder in shim.calls
    assert shim.calls[-1] is srv.folder_picker.pick_folder
