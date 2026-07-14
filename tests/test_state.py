import os

from rigma import state


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
