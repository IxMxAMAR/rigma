import json
import os

import psutil

from rigma import state


def _age_the_engine_identity(tmp_path, seconds=10_000.0):
    """Rewrite state.json so engine_pid names a process that started long
    before the record claims — i.e. the number was recycled."""
    s = state.read_state()
    s["engine_started_at"] = psutil.Process(int(s["engine_pid"])).create_time() - seconds
    (tmp_path / "state.json").write_text(json.dumps(s), encoding="utf-8")
    return s


def test_write_read_clear_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    assert state.read_state() is None
    state.write_state("qwen3.6-35b-a3b", "UD-Q3_K_XL", 11500,
                      engine_pid=os.getpid(), ui_pid=os.getpid())
    s = state.read_state()
    assert s["model"] == "qwen3.6-35b-a3b" and s["public_port"] == 11500
    assert state.server_running() is not None  # our own pid is alive
    state.clear_state()
    assert state.read_state() is None


def test_stale_state_is_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=999999, ui_pid=999999)
    assert state.server_running() is None
    assert state.read_state() is None  # stale file removed


def test_state_records_use_case(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), use_case="creative")
    assert state.read_state()["use_case"] == "creative"


def test_state_records_ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), ctx=4096)
    assert state.read_state()["ctx"] == 4096


# --- AUDIT F08-1: a pid is not an identity ---------------------------------
#
# Windows reallocates pids and state.json outlives the processes it names, so
# "is some process using this number" was never enough to justify terminating
# it. Before this fix the guard was exactly `pid_alive(pid)` (asserted below),
# so `rigma stop` would have killed whatever now owned the number; the test
# cannot even run against the old code, because kill_recorded did not exist.


def test_write_state_records_the_process_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    me = os.getpid()
    state.write_state("m", "q", 11500, engine_pid=me, ui_pid=me)
    s = state.read_state()
    assert abs(s["engine_started_at"] - psutil.Process(me).create_time()) < 1.0
    assert s["ui_started_at"] == s["engine_started_at"]


def test_a_recycled_pid_is_not_killed(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    me = os.getpid()
    state.write_state("m", "q", 11500, engine_pid=me, ui_pid=me)
    _age_the_engine_identity(tmp_path)
    killed = []
    monkeypatch.setattr(state, "kill_pid", lambda pid: killed.append(pid))
    # The old guard says "alive" for a number that is no longer ours. That is
    # the whole bug: it was the only question asked before terminating.
    assert state.pid_alive(me) is True
    assert state.kill_recorded(state.read_state(), "engine_pid") is False
    assert killed == []


def test_a_matching_record_is_still_killed(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    me = os.getpid()
    state.write_state("m", "q", 11500, engine_pid=me, ui_pid=me)
    killed = []
    monkeypatch.setattr(state, "kill_pid", lambda pid: killed.append(pid))
    assert state.kill_recorded(state.read_state(), "engine_pid") is True
    assert killed == [me]


def test_a_legacy_record_without_identity_is_still_killable(tmp_path, monkeypatch):
    # An old state.json has no engine_started_at; it must keep behaving as it
    # did rather than becoming unkillable.
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    me = os.getpid()
    state.write_state("m", "q", 11500, engine_pid=me, ui_pid=me)
    s = state.read_state()
    s["engine_started_at"] = 0.0
    (tmp_path / "state.json").write_text(json.dumps(s), encoding="utf-8")
    killed = []
    monkeypatch.setattr(state, "kill_pid", lambda pid: killed.append(pid))
    assert state.kill_recorded(state.read_state(), "engine_pid") is True
    assert killed == [me]


def test_server_running_ignores_a_recycled_engine_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    me = os.getpid()
    state.write_state("m", "q", 11500, engine_pid=me, ui_pid=me)
    _age_the_engine_identity(tmp_path)
    assert state.server_running() is None
    assert state.read_state() is None  # stale record cleared, nothing killed


def test_update_state_reidentifies_a_new_pid(tmp_path, monkeypatch):
    # A pid that changes hands must not carry the old process's stamp forward.
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=999_999, ui_pid=os.getpid())
    assert state.read_state()["engine_started_at"] == 0.0
    state.update_state(engine_pid=os.getpid())
    s = state.read_state()
    assert abs(s["engine_started_at"] - psutil.Process(os.getpid()).create_time()) < 1.0
