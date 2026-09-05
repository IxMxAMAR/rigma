"""The Grounding card's folder suggestions, and the bridge that fetches them.

The card used to offer a bare text box: adding a folder meant typing an
absolute path from memory. raggity does the looking now; these cover the wiring
and, mostly, the ways it is allowed to fail.
"""
import json
import subprocess

from rigma import rag


def _fake_run(monkeypatch, *, stdout="", code=0, exc=None, seen=None):
    def _run(argv, **kw):
        if seen is not None:
            seen.append((argv, kw))
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")
    monkeypatch.setattr(subprocess, "run", _run)


def test_candidates_are_passed_through(monkeypatch):
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    _fake_run(monkeypatch, stdout=json.dumps({
        "complete": True,
        "candidates": [{"path": "/home/u/notes", "kind": "obsidian",
                        "file_count": 341, "exts": {".md": 341},
                        "why": "Obsidian vault - 341 notes"}]}))
    got = rag.discover()
    assert got["available"] is True
    assert got["candidates"][0]["file_count"] == 341


def test_it_never_lets_raggity_prompt(monkeypatch):
    """This runs with no terminal. A raggity that decided to ask a question
    would hang the request until the timeout, so the env var is not optional."""
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    seen = []
    _fake_run(monkeypatch, stdout='{"complete": true, "candidates": []}',
              seen=seen)
    rag.discover()
    argv, kw = seen[0]
    assert kw["env"]["RAGGITY_NONINTERACTIVE"] == "1"
    assert "--json" in argv and "--deep" not in argv


def test_deep_is_forwarded(monkeypatch):
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    seen = []
    _fake_run(monkeypatch, stdout='{"complete": true, "candidates": []}',
              seen=seen)
    rag.discover(deep=True)
    assert "--deep" in seen[0][0]


def test_no_raggity_installed_is_not_an_error(monkeypatch):
    """The card still has its text box; a missing sidecar must not 500."""
    monkeypatch.setattr(rag, "raggity_cmd", lambda: None)
    assert rag.discover() == {"complete": True, "candidates": [],
                              "available": False}


def test_an_older_raggity_without_the_command_is_not_an_error(monkeypatch):
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    _fake_run(monkeypatch, stdout="Error: No such command 'discover'.", code=2)
    assert rag.discover()["available"] is False


def test_garbage_on_stdout_is_not_an_error(monkeypatch):
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    _fake_run(monkeypatch, stdout="not json at all")
    assert rag.discover()["available"] is False


def test_a_slow_scan_is_abandoned_rather_than_hanging_the_request(monkeypatch):
    monkeypatch.setattr(rag, "raggity_cmd", lambda: ["rag"])
    _fake_run(monkeypatch,
              exc=subprocess.TimeoutExpired(cmd="rag", timeout=30.0))
    assert rag.discover()["available"] is False
