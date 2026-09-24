"""AUDIT 03-3: an autonomous run's Stop must not kill a chat's background job.

`stop_run` called `tools.kill_all_jobs()`, which walks the module-global
`_JOBS` and kills every live process tree — including a download the owner had
started in a normal chat, reported afterwards as a nonzero exit with nothing
saying who killed it. Jobs now carry the `run_id` that started them and
`stop_run` kills only its own.

Fake job records and a stubbed `_kill_tree`: no real process is spawned or
killed here (the sandbox denies taskkill and a real kill would hang the test).
"""
import io

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions, tools
from rigma import state as st


class _FakeProc:
    def __init__(self, pid, rc=None):
        self.pid = pid
        self._rc = rc
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc if self._rc is not None else 0


def _job(pid, run_id="", rc=None):
    return {"proc": _FakeProc(pid, rc), "chunks": [], "buflen": 0,
            "lock": None, "cmd": "echo hi", "started": 0.0,
            "run_id": run_id}


@pytest.fixture
def fake_kill(monkeypatch):
    """A fresh `_JOBS` table and a `_kill_tree` that only records pids."""
    killed = []
    monkeypatch.setattr(tools, "_JOBS", {})
    monkeypatch.setattr(tools, "_kill_tree",
                        lambda pid, proc=None: killed.append(pid) or True)
    yield killed
    tools._JOBS.clear()


def test_kill_jobs_for_run_only_kills_that_runs_live_jobs(fake_kill):
    tools._JOBS[1] = _job(101, run_id="run-a")
    tools._JOBS[2] = _job(102, run_id="run-b")
    tools._JOBS[3] = _job(103, run_id="run-a", rc=0)     # already exited
    tools._JOBS[4] = _job(104, run_id="")                # a chat's job

    assert tools.kill_jobs_for_run("run-a") == 1
    assert fake_kill == [101]
    # entries stay so job_output can still report the exit code
    assert set(tools._JOBS) == {1, 2, 3, 4}


def test_kill_jobs_for_run_with_no_run_id_matches_nothing(fake_kill):
    tools._JOBS[1] = _job(101, run_id="")
    assert tools.kill_jobs_for_run("") == 0
    assert tools.kill_jobs_for_run(None) == 0
    assert fake_kill == []


def test_start_job_records_the_run_that_started_it(fake_kill, monkeypatch):
    monkeypatch.setattr(tools, "_launch_killable",
                        lambda argv, shell, cwd: _FakeProc(555))
    out = tools._start_job({"command": "echo hi"},
                           {"run_id": "run-a", "confirm_exec": True})
    assert out.startswith("started job"), out
    jid = max(tools._JOBS)
    assert tools._JOBS[jid]["run_id"] == "run-a"


def test_start_job_from_a_chat_records_no_run(fake_kill, monkeypatch):
    monkeypatch.setattr(tools, "_launch_killable",
                        lambda argv, shell, cwd: _FakeProc(556))
    tools._start_job({"command": "echo hi"}, {"confirm_exec": True})
    jid = max(tools._JOBS)
    assert tools._JOBS[jid]["run_id"] == ""


def test_stop_run_kills_only_its_own_jobs(tmp_path, monkeypatch, fake_kill):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    c = TestClient(serve.build_app(upstream_port=1))
    c.__enter__()
    try:
        sess = sessions.create("chat")
        rid = runs.create("mission", sess["id"])["id"]
        tools._JOBS[1] = _job(201, run_id=rid)
        tools._JOBS[2] = _job(202, run_id="")            # the chat's job

        r = c.post(f"/api/runs/{rid}/stop")
        assert r.status_code == 200
        assert fake_kill == [201], "stop_run killed the chat's job too"
        assert tools._JOBS[2]["proc"].poll() is None
    finally:
        c.__exit__(None, None, None)
