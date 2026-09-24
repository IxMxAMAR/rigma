"""A failed request must never be reported as a success.

AUDIT F08-2: `_stream_chat` swallowed every engine error — a non-streaming
error body, or a mid-stream `data: {"error": ...}` chunk — and returned "" or a
truncated string, which `chat` then persisted as a complete reply. The
run-control commands discarded the POST response entirely and echoed success, so
a 409/500 told the user an autonomous run had stopped while it kept iterating.

Everything here stubs httpx; no socket is opened.
"""
import json
import os

import httpx
import pytest
from typer.testing import CliRunner

import rigma.cli as cli
from rigma import sessions, state
from rigma.cli import app

runner = CliRunner()


class _StreamResponse:
    """What `httpx.stream(...)` yields: a context manager over SSE lines."""

    def __init__(self, status_code=200, lines=(), body=b""):
        self.status_code = status_code
        self._lines = list(lines)
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://127.0.0.1/v1"),
                response=httpx.Response(self.status_code))

    def iter_lines(self):
        yield from self._lines


def _stub_stream(monkeypatch, resp):
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: resp)


def _chunk(**delta):
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


# -- the chat stream ----------------------------------------------------------

def test_stream_chat_raises_on_an_http_error_body(monkeypatch):
    """A non-2xx reply is an error, not an empty assistant turn."""
    _stub_stream(monkeypatch, _StreamResponse(
        400, body=b'{"error":{"message":"context overflow"}}'))
    with pytest.raises(httpx.HTTPError):
        cli._stream_chat(11500, [{"role": "user", "content": "hi"}])


def test_stream_chat_raises_on_a_midstream_error_chunk(monkeypatch):
    """llama.cpp reports a released slot as a `data: {"error": ...}` chunk."""
    err = "data: " + json.dumps({"error": {"message": "slot released"}})
    _stub_stream(monkeypatch, _StreamResponse(
        200, [_chunk(content="The ans"), err, "data: [DONE]"]))
    with pytest.raises(RuntimeError, match="slot released"):
        cli._stream_chat(11500, [])


def test_stream_chat_raises_when_a_chunk_has_no_choices(monkeypatch):
    _stub_stream(monkeypatch, _StreamResponse(200, ['data: {"usage": {}}']))
    with pytest.raises(RuntimeError):
        cli._stream_chat(11500, [])


def test_stream_chat_still_returns_the_reply(monkeypatch):
    """The happy path, including the delta-less final chunk OpenAI sends."""
    _stub_stream(monkeypatch, _StreamResponse(
        200, [_chunk(content="Hel"), _chunk(content="lo"),
              'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
              "data: [DONE]"]))
    assert cli._stream_chat(11500, []) == "Hello"


def test_chat_drops_a_turn_the_engine_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid())
    _stub_stream(monkeypatch, _StreamResponse(400, body=b"{}"))
    res = runner.invoke(app, ["chat"], input="ping\nexit\n")
    assert res.exit_code == 0
    assert "model unreachable" in res.output
    assert sessions.list_sessions() == []   # the refused turn was never saved


# -- the run controls ---------------------------------------------------------

class _PostResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = {} if body is None else body

    def json(self):
        return self._body


def _stub_run(monkeypatch, response):
    monkeypatch.setattr(cli, "_run_server_base",
                        lambda: "http://127.0.0.1:11500")
    monkeypatch.setattr(cli, "_active_run_id", lambda: "run-abc")
    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


@pytest.mark.parametrize("verb,path", [
    ("stop", "stop"), ("pause", "pause"), ("resume", "resume"),
])
def test_run_control_reports_a_failed_post(tmp_path, monkeypatch, verb, path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    calls = _stub_run(monkeypatch, _PostResponse(500,
                                                 {"error": "run already finished"}))
    res = runner.invoke(app, ["run", verb])
    assert res.exit_code == 1
    assert "run already finished" in res.output
    assert calls and calls[0][0].endswith(f"/api/runs/run-abc/{path}")


def test_run_steer_reports_a_failed_post(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    calls = _stub_run(monkeypatch, _PostResponse(409, {"error": "no active run"}))
    res = runner.invoke(app, ["run", "steer", "go left"])
    assert res.exit_code == 1 and "no active run" in res.output
    assert calls[0][0].endswith("/api/runs/run-abc/inject")
    assert calls[0][1]["json"] == {"message": "go left"}


def test_run_stop_reports_success_only_on_200(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _stub_run(monkeypatch, _PostResponse(200))
    res = runner.invoke(app, ["run", "stop"])
    assert res.exit_code == 0 and "stopped run run-abc" in res.output
