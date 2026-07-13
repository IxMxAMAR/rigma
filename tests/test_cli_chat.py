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
