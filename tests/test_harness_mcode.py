"""The MiniMax Code adapter, against a stand-in CLI.

The event lines below are VERBATIM from a real mcode 0.5.1 turn (2026-09-21),
captured by running the shipping CLI against `tests/fake_oai_server.py`. They
are kept byte-for-byte rather than paraphrased: an adapter written against a
tidy version of a wire format is an adapter that has never met the real one, and
the whole reason this contract was measured instead of read is that the shipped
code is a minified bundle.

Nothing here needs mcode installed, a MiniMax account, or a GPU.
"""
import json
import os
import sys

import pytest

from rigma import harness, harness_mcode

ENVELOPE = {"schemaVersion": 1, "runId": "exec_turn_probe",
            "sessionId": "mvs_probe", "turnId": "turn_probe"}
MSG_ID = "eeb3d093-cccc-43cf-85f0-d1ab96cd670b:message"


def _ev(seq, type_, **rest):
    return {"sequence": seq, "timestampMs": 1789990501400 + seq,
            "type": type_, **ENVELOPE, **rest}


def _item(seq, type_, item):
    return _ev(seq, type_, item=item)


# --- a real text turn -------------------------------------------------------
TEXT_TURN = [
    _ev(1, "exec.started"),
    _ev(2, "session.started"),
    _ev(3, "turn.started"),
    _item(4, "item.started",
          {"id": MSG_ID, "type": "agent_message", "contentDelta": "hello "}),
    _item(5, "item.updated",
          {"id": MSG_ID, "type": "agent_message", "contentDelta": "from "}),
    _item(6, "item.updated",
          {"id": MSG_ID, "type": "agent_message", "contentDelta": "dsh"}),
    _item(7, "item.completed",
          {"id": MSG_ID, "type": "agent_message", "content": "hello from dsh"}),
    _ev(8, "turn.completed",
        model={"providerId": "custom_provider:rigma", "modelId": "local-test",
               "protocol": "openai-completions"},
        usage={"inputTokens": 8, "outputTokens": 3, "totalTokens": 11},
        usageSource="completed_responses",
        usageIncomplete=False, durationMs=359),
    _ev(9, "exec.completed", result={
        "schemaVersion": 1, "type": "exec.result", **ENVELOPE,
        "status": "succeeded", "output": "hello from dsh",
        "usageSource": "completed_responses", "durationMs": 359}),
]

# --- a real tool call, and the backend's answer to it -----------------------
# One id, re-emitted as it progresses: announced, then several updates with no
# result, then the result twice (updated AND completed). Exactly one `tool` and
# one `tool_result` may come out of this.
TOOL_TURN = [
    _item(1, "item.started",
          {"id": "call_probe_1", "type": "tool_call",
           "toolCall": {"id": "call_probe_1", "name": "rigma_probe_tool",
                        "status": 4}}),
    _item(2, "item.updated",
          {"id": "call_probe_1", "type": "tool_call",
           "toolCall": {"id": "call_probe_1", "name": "rigma_probe_tool",
                        "status": 5}}),
    _item(3, "item.updated",
          {"id": "call_probe_1", "type": "tool_call",
           "toolCall": {"id": "call_probe_1", "name": "rigma_probe_tool",
                        "status": 1, "input": {"probe": 1}}}),
    _item(4, "item.updated",
          {"id": "call_probe_1", "type": "tool_call",
           "toolCall": {"id": "call_probe_1", "name": "rigma_probe_tool",
                        "status": 3, "input": {"probe": 1},
                        "output": {"content": [{"type": "text",
                                                "text": "Tool not found"}],
                                   "details": {}}}}),
    _item(5, "item.completed",
          {"id": "call_probe_1", "type": "tool_call",
           "toolCall": {"id": "call_probe_1", "name": "rigma_probe_tool",
                        "status": 3, "input": {"probe": 1},
                        "output": {"content": [{"type": "text",
                                                "text": "Tool not found"}],
                                   "details": {}}}}),
]


def fold(events):
    """Run a captured stream through the adapter's mapper, as one turn."""
    seen: dict = {}
    out = []
    for obj in events:
        out.extend(harness_mcode.map_event(obj, seen))
    return out


def of_kind(events, kind):
    return [e for e in events if e.kind == kind]


# --------------------------------------------------------------------------
# the mapper: a real stream in, Rigma's vocabulary out


def test_the_reply_streams_from_the_deltas():
    got = fold(TEXT_TURN)
    assert [e.text for e in of_kind(got, "text")] == ["hello ", "from ", "dsh"]
    assert "".join(e.text for e in of_kind(got, "text")) == "hello from dsh"


def test_the_final_message_does_not_repeat_what_already_streamed():
    """mcode sends the deltas AND then the whole message. Emitting both would
    double every reply in the transcript."""
    got = of_kind(fold(TEXT_TURN), "text")
    assert "".join(e.text for e in got) == "hello from dsh"


def test_a_message_that_never_streamed_is_not_lost():
    """The other half of the same rule: a backend that reports only the final
    message must still produce a reply."""
    got = fold([_item(1, "item.completed",
                      {"id": MSG_ID, "type": "agent_message",
                       "content": "a whole answer"})])
    assert [e.text for e in of_kind(got, "text")] == ["a whole answer"]


def test_reasoning_becomes_thinking():
    got = fold([_item(1, "item.started",
                      {"id": "r:reasoning", "type": "reasoning",
                       "contentDelta": "let me think"})])
    assert [e.text for e in of_kind(got, "thinking")] == ["let me think"]


def test_a_tool_call_is_announced_once_and_answered_once():
    """mcode re-emits one item four times as it progresses. Without the memo a
    single call becomes four chips — the wrong-row class this project has
    already paid for once."""
    got = fold(TOOL_TURN)
    calls = of_kind(got, "tool")
    results = of_kind(got, "tool_result")
    assert len(calls) == 1, [e.name for e in calls]
    assert calls[0].name == "rigma_probe_tool"
    assert calls[0].args == {"probe": 1}
    assert len(results) == 1
    assert results[0].text == "Tool not found"
    assert results[0].name == "rigma_probe_tool"


def test_a_call_with_no_arguments_yet_is_held_not_announced_empty():
    """mcode announces a call BEFORE its arguments finish streaming. Announcing
    at that moment would put an empty argument box in the transcript for every
    single tool call."""
    assert fold(TOOL_TURN[:1]) == []
    assert fold(TOOL_TURN[:2]) == []


def test_the_call_is_announced_with_its_arguments():
    got = fold(TOOL_TURN[:3])
    assert [e.kind for e in got] == ["tool"]
    assert got[0].name == "rigma_probe_tool"
    assert got[0].args == {"probe": 1}


def test_a_result_without_arguments_still_announces_the_call():
    """Held, never dropped: if the arguments never arrive the call is emitted
    immediately before its result rather than vanishing."""
    got = fold([_item(1, "item.completed",
                      {"id": "c1", "type": "tool_call",
                       "toolCall": {"name": "t", "output": {"content": [
                           {"type": "text", "text": "boom"}]}}})])
    assert [e.kind for e in got] == ["tool", "tool_result"]
    assert got[0].args == {}


def test_a_failed_turn_reports_the_reason_the_server_gave():
    got = fold([_ev(1, "turn.failed", status="failed",
                    error={"category": "runtime", "retryable": True,
                           "message": 'Model "rigma/local-test" is not '
                                      'available for the configured_provider '
                                      'route'})])
    assert len(got) == 1
    assert got[0].kind == "error"
    assert "not available" in got[0].text


def test_a_failed_turn_with_no_message_still_says_something():
    got = fold([_ev(1, "turn.failed", status="failed")])
    assert got[0].kind == "error" and got[0].text


def test_the_envelope_and_lifecycle_events_are_not_items():
    """`exec.started`, `turn.started` and `exec.completed` carry no content. A
    mapper that guessed at them would put bookkeeping in the transcript."""
    assert fold([_ev(1, "exec.started"), _ev(2, "turn.started"),
                 _ev(3, "exec.completed", result={"status": "succeeded"})]) == []


def test_an_unknown_item_type_is_ignored_not_guessed_at():
    """mcode's schema is versioned. A future item kind must not become a wrong
    row in the transcript."""
    assert fold([_item(1, "item.started",
                       {"id": "x", "type": "something_new", "content": "?"})]) == []


def test_a_tool_result_that_is_not_a_content_list_is_still_text():
    got = fold([_item(1, "item.completed",
                      {"id": "c1", "type": "tool_call",
                       "toolCall": {"name": "t", "output": "plain text"}})])
    assert of_kind(got, "tool_result")[0].text == "plain text"


# --------------------------------------------------------------------------
# the driver: a stand-in CLI, so none of this needs mcode installed

FAKE_SRC = '''\
import json, os, sys
argv = sys.argv[1:]
log = os.environ.get("FAKE_MCODE_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(argv) + "\\n")
cmd = argv[0] if argv else ""
if cmd == "provider":
    action = argv[1] if len(argv) > 1 else ""
    if action == "add":
        print("Provider added: rigma")
    sys.exit(0)
if cmd == "exec":
    for line in json.loads(os.environ["FAKE_MCODE_EVENTS"]):
        sys.stdout.write(json.dumps(line) + "\\n")
    sys.stdout.flush()
    sys.exit(int(os.environ.get("FAKE_MCODE_EXIT", "0")))
sys.exit(2)
'''


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """A `mcode` on PATH that answers the two commands the adapter runs."""
    script = tmp_path / "fake_mcode.py"
    script.write_text(FAKE_SRC, encoding="utf-8")
    if os.name == "nt":
        exe = tmp_path / "mcode.cmd"
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                       encoding="utf-8")
    else:
        exe = tmp_path / "mcode"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                       encoding="utf-8")
        exe.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("RIGMA_MCODE_BIN", str(exe))
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_MCODE_LOG", str(log))
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(TEXT_TURN))
    return log


def logged(log):
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]


def test_a_turn_is_driven_through_the_cli(fake_cli):
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="local-test",
        prompt="say hi"))
    assert "".join(e.text for e in got if e.kind == "text") == "hello from dsh"
    assert not [e for e in got if e.kind == "error"], got

    execs = [a for a in logged(fake_cli) if a and a[0] == "exec"]
    assert len(execs) == 1
    argv = execs[0]
    assert "--output-format" in argv and "stream-json" in argv
    # headless: `smart` needs a TUI to ask, and `off` would disarm the tools
    assert argv[argv.index("--permission") + 1] == "full"
    # THE model reference, measured: without the `custom_provider:` prefix
    # mcode refuses the model outright
    assert argv[argv.index("--model") + 1] == "custom_provider:rigma/local-test"
    assert argv[-1] == "say hi"


def test_the_provider_is_pointed_at_rigmas_own_endpoint(fake_cli):
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="local-test",
        prompt="hi", context_window=32768, max_tokens=4096))
    adds = [a for a in logged(fake_cli) if a[:2] == ["provider", "add"]]
    assert len(adds) == 1
    argv = adds[0]
    # RIGMA's /v1, never llama-server's — the point of the seam
    assert argv[argv.index("--base-url") + 1] == "http://127.0.0.1:11500/v1"
    assert argv[argv.index("--api-format") + 1] == "openai-completions"
    assert argv[argv.index("--model") + 1] == "local-test"
    assert argv[argv.index("--context-limit") + 1] == "32768"
    assert argv[argv.index("--output-limit") + 1] == "4096"
    # a value has to EXIST for mcode to read; llama-server ignores the header
    assert argv[argv.index("--api-key-env") + 1] == harness_mcode.API_KEY_ENV


def test_the_provider_is_configured_once_not_every_turn(fake_cli):
    """`provider add` is a second Node process, and re-adding a name mcode
    already has is an error rather than an update."""
    for _ in range(3):
        list(harness_mcode.drive_turn(
            base_url="http://127.0.0.1:11500/v1", model="local-test",
            prompt="hi"))
    adds = [a for a in logged(fake_cli) if a[:2] == ["provider", "add"]]
    assert len(adds) == 1, adds


def test_a_changed_model_replaces_the_cached_provider(fake_cli):
    """mcode keys a custom provider by NAME, and the name is Rigma's, so a new
    model has to remove the old entry or the add is refused."""
    for model in ("first", "second"):
        list(harness_mcode.drive_turn(
            base_url="http://127.0.0.1:11500/v1", model=model, prompt="hi"))
    calls = [a[:2] for a in logged(fake_cli) if a and a[0] == "provider"]
    assert calls == [["provider", "add"], ["provider", "remove"],
                     ["provider", "add"]]
    adds = [a for a in logged(fake_cli) if a[:2] == ["provider", "add"]]
    assert adds[-1][adds[-1].index("--model") + 1] == "second"


def test_a_missing_cli_is_reported_and_never_raised(monkeypatch, tmp_path):
    """A machine without mcode gets an error event, not an exception in a
    worker thread that the turn loop cannot see."""
    monkeypatch.setenv("RIGMA_MCODE_BIN", str(tmp_path / "nope"))
    monkeypatch.setattr(harness_mcode.shutil, "which", lambda _n: None)
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.kind for e in got] == ["error"]
    assert "PATH" in got[0].text


def test_a_failed_turn_surfaces_the_reason(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(
        [_ev(1, "turn.failed", status="failed",
             error={"category": "runtime",
                    "message": "upstream error: connection refused"})]))
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.kind for e in got] == ["error"]
    assert "connection refused" in got[0].text


def test_the_provider_is_selected_not_merely_added(fake_cli):
    """THE bug this adapter was written wrong once already.

    Adding a provider without `--use` saves it and leaves NO provider active,
    and mcode then refuses every turn with "Sign in to MiniMax to use Agent
    features" — about the ACTIVE provider's account, even when `--model` names
    the custom provider explicitly. That reads exactly like a hard account
    prerequisite for a turn that never leaves this machine. `--use` selects it
    and the gate is gone."""
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="local-test", prompt="hi"))
    add = [a for a in logged(fake_cli) if a[:2] == ["provider", "add"]][0]
    assert "--use" in add
    # and it is the LAST thing on the command, where `provider add` expects it
    assert add[-1] == "--use"


def test_the_menu_says_why_the_provider_must_be_selected():
    """The one step an integrator can silently skip, stated where the backend
    is chosen rather than discovered as an account error at the first turn."""
    wire = harness.BACKENDS[harness.MCODE].wire
    assert "--use" in wire
    assert "MiniMax login" in wire


def test_mcode_is_selectable_and_driven_by_its_adapter(monkeypatch, tmp_path):
    """The dispatch table, exercised for the second backend: the seam names no
    adapter, so adding one must be a module and a table entry."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: True)
    seen: dict = {}

    def _fake_drive(**kw):
        # snapshot what the adapter was HANDED, before writing back to it
        seen.update(kw)
        seen.setdefault("handed", []).append(dict(kw["state"]))
        # the backend's own handle on the conversation, reported the way a real
        # one reports it: mid-stream, out of band from the text
        kw["state"]["session_id"] = "mvs_fake_session"
        yield harness.TurnEvent("notice", text="working")
        yield harness.TurnEvent("text", text="from ")
        yield harness.TurnEvent("text", text="mcode")

    monkeypatch.setattr(harness_mcode, "drive_turn", _fake_drive)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    real_read = st.read_state
    monkeypatch.setattr(st, "read_state",
                        lambda: {**(real_read() or {}),
                                 "public_port": 11500, "ctx": 32768})
    s = sessions.create(title="t")
    s["harness"] = "mcode"
    sessions.save(s)

    with TestClient(serve.build_app(upstream_port=11499)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "go"})
        assert r.status_code == 200, r.text
        assert '"delta": "from "' in r.text
        assert '"name": "mcode"' in r.text      # the harness event, up front
        # a SECOND turn, to prove Rigma remembers what the backend told it
        r2 = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "again"})
        assert r2.status_code == 200, r2.text

    assert seen["base_url"] == "http://127.0.0.1:11500/v1"
    assert seen["context_window"] == 32768
    assert seen["handed"][0] == {"session_id": ""}
    assert seen["handed"][1] == {"session_id": "mvs_fake_session"}

    stored = sessions.load(s["id"])
    assert stored["messages"][-1]["content"] == "from mcode"
    assert stored["messages"][-1]["harness"] == "mcode"
    assert stored["messages"][-1]["harness_label"] == "MiniMax Code"
    # and what the backend told us about itself is Rigma's to keep
    assert stored["harness_sessions"] == {"mcode": "mvs_fake_session"}


# --------------------------------------------------------------------------
# continuity: the difference between an agent and a stateless prompt


def test_a_fresh_chat_passes_no_session(fake_cli):
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    argv = [a for a in logged(fake_cli) if a and a[0] == "exec"][0]
    assert "--session" not in argv


def test_a_remembered_session_is_handed_back_to_continue(fake_cli):
    """mcode keeps the plan, the subagents and the goals in the session, so a
    fresh session per turn throws away most of what the agent is for."""
    state = {"session_id": "mvs_2f1e8eb0d851476cb2f9aef5eba79de5"}
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi",
        state=state))
    argv = [a for a in logged(fake_cli) if a and a[0] == "exec"][0]
    assert argv[argv.index("--session") + 1] == \
        "mvs_2f1e8eb0d851476cb2f9aef5eba79de5"


def test_the_session_id_is_read_back_out_of_the_stream(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(TEXT_TURN))
    state: dict = {}
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi",
        state=state))
    assert state["session_id"] == "mvs_probe"
    assert state["resumed"] is False
    assert state["status"] == "succeeded"
    assert state["usage"]["totalTokens"] == 11


def test_a_resumed_session_says_so(fake_cli, monkeypatch):
    """Continuity the reader cannot see is indistinguishable from a backend
    that forgot everything."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(
        [_ev(1, "exec.started"), _ev(2, "session.resumed"),
         _ev(3, "turn.started")]))
    state: dict = {}
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi",
        state=state))
    assert state["resumed"] is True
    notes = [e.text for e in got if e.kind == "notice"]
    assert notes and "continuing" in notes[0]


def test_the_final_answer_is_used_when_nothing_streamed(fake_cli, monkeypatch):
    """mcode reports the answer on `exec.completed` whether or not it streamed.
    A turn that produced a reply must not come back with an empty transcript."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([
        _ev(1, "exec.started"), _ev(2, "session.started"),
        _ev(3, "turn.started"),
        _ev(4, "turn.completed", usage={"totalTokens": 4}),
        _ev(5, "exec.completed", result={"status": "succeeded",
                                         "output": "a whole answer"}),
    ]))
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.text for e in got if e.kind == "text"] == ["a whole answer"]


def test_a_streamed_answer_is_not_repeated_from_the_result(fake_cli,
                                                           monkeypatch):
    """The other half of the same rule — the fallback must not double a reply
    that already arrived as deltas."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(TEXT_TURN))
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert "".join(e.text for e in got if e.kind == "text") == "hello from dsh"


def test_continuity_survives_a_real_second_call(fake_cli, monkeypatch):
    """End to end through the driver: turn one learns the id, turn two hands it
    back — which is the whole mechanism, with no state passed by the caller."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(TEXT_TURN))
    state: dict = {}
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="one",
        state=state))
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="two",
        state=state))
    execs = [a for a in logged(fake_cli) if a and a[0] == "exec"]
    assert "--session" not in execs[0]
    assert execs[1][execs[1].index("--session") + 1] == "mvs_probe"
