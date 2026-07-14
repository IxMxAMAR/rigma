import os
from unittest.mock import patch

from typer.testing import CliRunner

from rigma import sessions, state
from rigma.cli import app

runner = CliRunner()


def test_chat_persists_to_session_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    seen = {}

    def fake_stream(port, history):
        seen["history"] = history
        return "pong"

    with patch("rigma.cli._stream_chat", fake_stream), \
         patch("rigma.sessions.default_prompt", return_value="DEFAULT"):
        r = runner.invoke(app, ["chat"], input="ping\nexit\n")
    assert r.exit_code == 0
    assert seen["history"][0] == {"role": "system", "content": "DEFAULT"}
    assert seen["history"][1] == {"role": "user", "content": "ping"}
    stored = sessions.list_sessions()
    assert len(stored) == 1 and stored[0]["message_count"] == 2
    sess = sessions.load(stored[0]["id"])
    assert sess["messages"][1] == {"role": "assistant", "content": "pong"}
    assert sess["title"] == "ping"


def test_chat_unknown_session_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    r = runner.invoke(app, ["chat", "--session", "nope"])
    assert r.exit_code == 1 and "no such session" in r.output


def test_chat_exiting_immediately_leaves_no_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    r = runner.invoke(app, ["chat"], input="exit\n")
    assert r.exit_code == 0
    assert sessions.list_sessions() == []


def test_chat_survives_connection_drop_and_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    with patch("rigma.cli._stream_chat", side_effect=RuntimeError("boom")), \
         patch("rigma.sessions.default_prompt", return_value="DEFAULT"):
        r = runner.invoke(app, ["chat"], input="ping\nexit\n")
    assert r.exit_code == 0
    assert "model unreachable" in r.output
    stored = sessions.list_sessions()
    assert len(stored) == 1
    sess = sessions.load(stored[0]["id"])
    assert len(sess["messages"]) == 1
    assert sess["messages"][0] == {"role": "user", "content": "ping"}


def test_chat_resuming_rag_session_prints_ungrounded_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    sess = sessions.create()
    sess["use_rag"] = True
    sessions.save(sess)
    with patch("rigma.cli._stream_chat", return_value="pong"), \
         patch("rigma.sessions.default_prompt", return_value="DEFAULT"):
        r = runner.invoke(app, ["chat", "--session", sess["id"]], input="exit\n")
    assert r.exit_code == 0
    assert "ungrounded" in r.output
