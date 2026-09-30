"""The MiniMax Code adapter, against a stand-in CLI.

The event lines below are VERBATIM from a real mcode 0.5.1 turn (2026-09-21),
captured by running the shipping CLI against `tests/fake_oai_server.py`. They
are kept byte-for-byte rather than paraphrased: an adapter written against a
tidy version of a wire format is an adapter that has never met the real one, and
the whole reason this contract was measured instead of read is that the shipped
code is a minified bundle.

Nothing here needs mcode installed, a MiniMax account, or a GPU.
"""
import io
import json
import os
import subprocess as _sp
import sys
import threading
import time
import types

import pytest

from rigma import harness, harness_mcode, rag


def _a_different_version(version: str) -> str:
    """Some version that is definitely NOT `version`, for drift tests.

    Derived rather than written out so a routine bump of `harness_mcode.VERIFIED`
    cannot turn the drift tests into failures — which is what happened when the
    pin moved 0.5.1 -> 0.5.4 and two tests here still named the old build. Bumping
    the LAST component keeps it a plausible version string, so the message the
    user sees is realistic.
    """
    head, _, tail = version.rpartition(".")
    if head and tail.isdigit():
        return f"{head}.{int(tail) + 1}"
    return version + ".1"


@pytest.fixture(autouse=True)
def _a_recorded_sidecar_answers(monkeypatch):
    """A recorded RAG sidecar counts as LIVE for this file.

    F52 made the record mean "answering" rather than "we once ran Popen", so
    `seed_docs` writing `sidecar.json` is no longer enough on its own — the port
    has to answer, and there is no real sidecar in these tests.

    Autouse because `seed_docs` is a plain helper called MID-test, where a
    fixture argument could not reach it. Harmless for a test that seeds nothing:
    with no record, the health probe is never consulted. What these tests are
    about is what gets REGISTERED, not liveness — liveness has its own tests in
    test_rag.py.
    """
    monkeypatch.setattr(rag, "sidecar_health",
                        lambda port=0, **_k: {"status": "ok", "port": port})

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


def test_a_malformed_nested_object_is_skipped_not_fatal():
    """The adapter's own promise is "a malformed line is a skipped line".

    It held for the OUTER object and stopped there: `map_event` did
    `item.get(...)` / `toolCall.get(...)` / `error.get(...)` on whatever was
    nested, so one event whose `item`, `toolCall` or `error` was a string or a
    list raised AttributeError out of the read loop. `drive_turn` catches that
    as "mcode stream failed", so a turn that had already streamed half its
    reply ended in an error and the rest of the stream was thrown away — the
    same shape 09-2 fixed on the MCP client, one level down.
    """
    for bad in ("hot", [1], 7, None):
        for obj in (
            {"type": "item.started", "item": bad},
            {"type": "item.started",
             "item": {"id": "c", "type": "tool_call", "toolCall": bad}},
        ):
            seen: dict = {}
            assert harness_mcode.map_event(obj, seen) == [], obj
        # a malformed `error` still has to say the turn failed — it just may
        # not raise on the way
        got = harness_mcode.map_event({"type": "turn.failed", "error": bad}, {})
        assert [e.kind for e in got] == ["error"], got
        assert got[0].text


def test_one_malformed_event_does_not_end_the_turn(fake_cli):
    """End to end through the real drive_turn: the reply must survive."""
    import os as _os
    events = list(TEXT_TURN)
    events.insert(4, _item(99, "item.started",
                           {"id": "c", "type": "tool_call",
                            "toolCall": "bash"}))
    _os.environ["FAKE_MCODE_EVENTS"] = json.dumps(events)
    got = list(harness_mcode.drive_turn(
        base_url=BASE, model="local-test", prompt="p", state={}, timeout=30))
    assert not [e for e in got if e.kind == "error"], got
    assert "".join(e.text for e in got if e.kind == "text") == "hello from dsh"


def test_a_stream_failure_does_not_leak_the_child(fake_cli, monkeypatch):
    """A driver failure must not outlive the turn it belonged to.

    Checked rather than assumed: `drive_turn`'s defensive `except` is above a
    `finally`, so the child IS reaped on that path — this test is what says so,
    and it fails the day the `finally` is restructured away.
    """
    import os as _os
    _os.environ["FAKE_MCODE_EVENTS"] = json.dumps(TEXT_TURN)

    def boom(obj, seen):
        raise RuntimeError("boom")

    monkeypatch.setattr(harness_mcode, "map_event", boom)
    proc_seen = {}
    real_popen = harness_mcode.subprocess.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        if p.stdout is not None:
            # The turn child is the one whose stdout is a PIPE. Anything else
            # reaching `Popen` while this spy is installed has no pipe: the
            # `taskkill` that `_stop -> kill_tree -> tools._kill_tree` spawns on
            # the failure path (stdout=DEVNULL), and mcode's setup commands
            # (subprocess.run, no pipe). Which of those is the LAST `Popen` call
            # is a child-exit RACE, so select on the pipe, not on order. Hold the
            # pipe ITSELF: Popen drops the attribute once the child is reaped,
            # and the leak this checks for is the FD, not the attribute.
            proc_seen["p"], proc_seen["out"] = p, p.stdout
        return p

    monkeypatch.setattr(harness_mcode.subprocess, "Popen", spy)
    got = list(harness_mcode.drive_turn(
        base_url=BASE, model="local-test", prompt="p", state={}, timeout=30))
    assert [e.kind for e in got] == ["error"], got
    p = proc_seen["p"]
    assert p.poll() is not None, "the mcode child outlived the failed turn"
    assert proc_seen["out"].closed, "the stdout pipe was left open"


def test_the_memo_is_per_turn_not_per_process():
    """A process-wide memo would drop a legitimate item in a later turn whose
    id happened to repeat. `map_event` must work off the caller's dict."""
    obj = _item(1, "item.completed",
                {"id": "same-id", "type": "agent_message", "content": "one"})
    first = harness_mcode.map_event(obj, {})
    second = harness_mcode.map_event(obj, {})
    assert [e.text for e in first] == ["one"]
    assert [e.text for e in second] == ["one"]


def test_a_build_we_were_not_verified_against_says_so_in_the_turn(fake_cli,
                                                                  monkeypatch):
    """The whole point of `VERIFIED` is that a released backend can change shape
    while every turn still LOOKS fine — a renamed item type drops tool calls
    from the transcript and the reply still arrives.

    `harness.conformance` knows how to say that, but it only runs from
    `rigma harness` and the menu's `?check=1`. Nothing on the TURN path
    compared anything, so installing a newer build and chatting produced a
    normal-looking turn on a build nobody measured. The turn is where the owner
    is.

    The two versions are derived from `VERIFIED` rather than written out: this
    test is about the MECHANISM, and hardcoding the pinned build made a routine
    version bump (0.5.1 -> 0.5.4) look like two broken tests. The literal that
    used to be here is exactly the failure mode
    `test_the_verified_build_is_a_fact_about_the_code` warns about.
    """
    other = _a_different_version(harness_mcode.VERIFIED)
    monkeypatch.setenv("FAKE_MCODE_VERSION", other)
    monkeypatch.setattr(harness_mcode, "_DRIFT_SAID", False)
    got = list(harness_mcode.drive_turn(
        base_url=BASE, model="local-test", prompt="p", state={}, timeout=30))
    notes = [e.text for e in got if e.kind == "notice"]
    assert any(other in n and harness_mcode.VERIFIED in n for n in notes), notes
    # the turn still runs: drift is a warning, not a refusal
    assert "".join(e.text for e in got if e.kind == "text") == "hello from dsh"


def test_a_build_we_were_verified_against_is_not_a_notice(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_MCODE_VERSION", harness_mcode.VERIFIED)
    monkeypatch.setattr(harness_mcode, "_DRIFT_SAID", False)
    got = list(harness_mcode.drive_turn(
        base_url=BASE, model="local-test", prompt="p", state={}, timeout=30))
    assert not [e for e in got if e.kind == "notice"], got


def test_the_drift_notice_is_said_once_not_every_turn(fake_cli, monkeypatch):
    """One subprocess per process, and one line in the transcript — a warning
    repeated every turn is noise the reader learns to skip."""
    monkeypatch.setenv("FAKE_MCODE_VERSION",
                       _a_different_version(harness_mcode.VERIFIED))
    monkeypatch.setattr(harness_mcode, "_DRIFT_SAID", False)
    calls = []
    real = harness_mcode.backend_version

    def counted(exe=None):
        calls.append(exe)
        return real(exe)

    monkeypatch.setattr(harness_mcode, "backend_version", counted)
    for _ in range(2):
        got = list(harness_mcode.drive_turn(
            base_url=BASE, model="local-test", prompt="p", state={}, timeout=30))
    assert len(calls) == 1, calls
    assert got  # the second turn still ran


def test_a_tool_call_that_is_announced_once_and_answered_once():
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
if cmd in ("--version", "-V"):
    # The real CLI answers a stable semver here; `harness.conformance` and the
    # drift notice both read it. Default it to the build this adapter was
    # measured against, so a test that wants drift has to ask for it.
    print(os.environ.get("FAKE_MCODE_VERSION", "0.5.1"))
    sys.exit(0)
if cmd == "provider":
    action = argv[1] if len(argv) > 1 else ""
    store = os.environ.get("FAKE_MCODE_PROVIDERS_FILE") or ""

    def load():
        try:
            with open(store, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save(provs):
        with open(store, "w", encoding="utf-8") as f:
            json.dump(provs, f)

    if action == "list":
        code = int(os.environ.get("FAKE_MCODE_LIST_EXIT", "0"))
        if code:
            sys.stderr.write("provider list is unhappy\\n")
            sys.exit(code)
        if store:
            provs = load()
        else:
            raw = os.environ.get("FAKE_MCODE_PROVIDERS")
            provs = (json.loads(raw).get("providers") if raw else [{
                "providerId": "custom_provider:rigma", "kind": "custom",
                "active": True, "enabled": True,
                "baseUrl": os.environ.get(
                    "FAKE_MCODE_BASEURL", "http://127.0.0.1:11500/v1")}])
        sys.stdout.write(json.dumps({"providers": provs}))
        sys.exit(0)
    if action == "remove":
        pid = argv[2] if len(argv) > 2 else ""
        # THE REAL BEHAVIOUR, and the bug this fake exists to catch: a remove
        # without --yes does nothing and says so on stderr. Rigma called it that
        # way, the removal silently failed, and the leftover made the next add
        # dedupe to rigma-2 while --model kept naming rigma.
        if "--yes" not in argv:
            sys.stderr.write("Refusing to remove a provider without --yes.\\n")
            sys.exit(2)
        if store:
            save([p for p in load() if p.get("providerId") != pid])
        print("Provider removed: " + pid)
        sys.exit(0)
    if action == "add":
        name = argv[argv.index("--name") + 1]
        url = argv[argv.index("--base-url") + 1]
        if store:
            provs = load()
            taken = {p.get("providerId") for p in provs}
            pid, n = "custom_provider:" + name, 2
            while pid in taken:          # mcode dedupes; it does not update
                pid = "custom_provider:%s-%d" % (name, n)
                n += 1
            for p in provs:
                p["active"] = False
            provs.append({"providerId": pid, "kind": "custom", "active": True,
                          "enabled": True, "baseUrl": url})
            save(provs)
        print("Provider added and selected: " + name)
        sys.exit(0)
    sys.exit(0)
if cmd == "exec":
    for line in json.loads(os.environ["FAKE_MCODE_EVENTS"]):
        sys.stdout.write(json.dumps(line) + "\\n")
    sys.stdout.flush()
    if os.environ.get("FAKE_MCODE_HANG"):
        import time
        time.sleep(600)
    sys.exit(int(os.environ.get("FAKE_MCODE_EXIT", "0")))
sys.exit(2)
'''

BASE = "http://127.0.0.1:11500/v1"


def test_an_mcode_child_is_detached_on_posix_only(monkeypatch):
    """mcode must be in its OWN process group on POSIX.

    `kill_tree` reaches a tree with `killpg` there, and a child left in RIGMA's
    group would make a stop take the server down with it. Windows is unchanged.

    The setup commands are stubbed so the only `Popen` the spy sees is the turn
    child's — a `subprocess.run` helper would be captured first otherwise.
    """
    captured = []

    class _Stop(Exception):
        pass

    def spy(argv, **kw):
        captured.append(kw)
        raise _Stop

    monkeypatch.setattr(harness_mcode, "bin_path", lambda: "mcode")
    monkeypatch.setattr(harness_mcode, "_drift_notice", lambda exe: None)
    monkeypatch.setattr(harness_mcode, "ensure_provider",
                        lambda *a, **k: ("pid", ""))
    monkeypatch.setattr(harness_mcode, "ensure_agents_md", lambda: None)
    monkeypatch.setattr(harness_mcode, "ensure_mcp", lambda cwd: None)
    monkeypatch.setattr(harness_mcode.subprocess, "Popen", spy)

    def drive():
        return list(harness_mcode.drive_turn(
            base_url=BASE, model="local-test", prompt="p", state={}, timeout=5))

    monkeypatch.setattr(harness_mcode._harness, "_DETACH_CHILDREN", True)
    with pytest.raises(_Stop):
        drive()
    assert captured[-1].get("start_new_session") is True, captured[-1]

    monkeypatch.setattr(harness_mcode._harness, "_DETACH_CHILDREN", False)
    with pytest.raises(_Stop):
        drive()
    assert "start_new_session" not in captured[-1], captured[-1]


def _prov(pid="custom_provider:rigma", *, active=True, enabled=True, url=BASE):
    return {"providerId": pid, "kind": "custom", "active": active,
            "enabled": enabled, "baseUrl": url}


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """A `mcode` on PATH that answers the commands the adapter runs.

    Its provider store is a FILE, and `add` dedupes names and `remove` refuses
    without `--yes` exactly as the real CLI does. A stateless fake would let the
    dedupe bug back in: the whole failure was that `add` created `rigma-2` while
    `--model` still named `rigma`, and a fake that always answers "rigma" cannot
    tell those apart.
    """
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
    store = tmp_path / "providers.json"
    store.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("RIGMA_MCODE_BIN", str(exe))
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_MCODE_LOG", str(log))
    monkeypatch.setenv("FAKE_MCODE_PROVIDERS_FILE", str(store))
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps(TEXT_TURN))
    # AUDIT 13-6: the child's environment is an explicit allowlist now, so a
    # wrapper that needs extra names declares them through the passthrough hook
    # (this is also the escape hatch a real owner would use).
    monkeypatch.setenv("RIGMA_MCODE_ENV_PASSTHROUGH", "FAKE_MCODE_*")
    # R3-HARN-7: the fake CLI reports 0.5.1 when `FAKE_MCODE_VERSION` is unset —
    # that was the pinned build when its event lines were captured. It used to
    # inherit a real value from the ambient environment by accident, because the
    # adapter copied the whole environment into the child. Now that the child's
    # environment is an allowlist, the fixture has to say so itself: otherwise the
    # first turn to run emits a drift notice and latches `_DRIFT_SAID`, and every
    # later test that asserts on the exact notice list fails by test ORDER.
    monkeypatch.setenv("FAKE_MCODE_VERSION", harness_mcode.VERIFIED)
    # module-level state, so it must not leak between tests
    monkeypatch.setattr(harness_mcode, "_VERIFIED", {})
    return log


def seed_providers(tmp_path, provs):
    (tmp_path / "providers.json").write_text(json.dumps(provs),
                                             encoding="utf-8")


def seed_marker(tmp_path, base_url=BASE,
                model="local-test", ctx=32768, out=4096):
    """What Rigma leaves behind after a successful setup, on a later process."""
    d = tmp_path / "home" / "mcode"
    d.mkdir(parents=True, exist_ok=True)
    (d / "provider.json").write_text(json.dumps(
        {"base_url": base_url, "model": model, "context_limit": ctx,
         "output_limit": out}), encoding="utf-8")


def provider_calls(log):
    return [a for a in logged(log) if a[:2] == ["provider", "add"]]


def remove_calls(log):
    return [a for a in logged(log) if a[:2] == ["provider", "remove"]]


def list_calls(log):
    return [a for a in logged(log) if a[:2] == ["provider", "list"]]


def model_arg(call):
    return call[call.index("--model") + 1]


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
    """`provider add` is a second Node process, so it must not run per turn."""
    for _ in range(3):
        _one_turn()
    assert len(provider_calls(fake_cli)) == 1


def test_the_remove_says_yes_or_it_does_nothing(fake_cli, tmp_path):
    """THE BUG THIS FAKE EXISTS TO CATCH.

    `provider remove` refuses without `--yes` — "Refusing to remove a provider
    without --yes" — and exits 2. Rigma called it without, so the removal
    silently did nothing, the old provider stayed, the next `add` deduped to
    `rigma-2`, and `--use` activated `rigma-2` while `--model
    custom_provider:rigma/…` kept resolving to the stale one. Every change
    leaked a provider and pointed the turn one step further from the id it
    named.
    """
    seed_providers(tmp_path, [_prov(active=False, url="http://127.0.0.1:9999/v1")])
    _one_turn()
    removes = remove_calls(fake_cli)
    assert removes, "a provider at the wrong URL has to be replaced"
    assert all("--yes" in a for a in removes), removes
    # and the proof it WORKED: with the old entry gone the name is free again,
    # so the turn names `rigma` and not `rigma-2`
    exe = [a for a in logged(fake_cli) if a and a[0] == "exec"][0]
    assert model_arg(exe) == "custom_provider:rigma/local-test"


def test_the_provider_id_is_read_back_rather_than_assumed(fake_cli, tmp_path):
    """What mcode CALLED the provider is what `--model` has to name.

    This is the post-dedupe state: `rigma` exists but is inactive and points
    somewhere else, and the live one is `rigma-2`. Naming `custom_provider:rigma`
    because that is the name Rigma asked for sends the turn to a dead port.
    """
    seed_providers(tmp_path, [
        _prov(active=False, url="http://127.0.0.1:9999/v1"),
        _prov("custom_provider:rigma-2", active=True, url=BASE),
    ])
    seed_marker(tmp_path)
    _one_turn()
    assert provider_calls(fake_cli) == []      # nothing to configure
    assert remove_calls(fake_cli) == []        # and nothing to clean up
    exe = [a for a in logged(fake_cli) if a and a[0] == "exec"][0]
    assert model_arg(exe) == "custom_provider:rigma-2/local-test"


def test_a_changed_model_replaces_the_cached_provider(fake_cli):
    """A new model has to clear the old entry, or the add dedupes and the turn
    keeps naming the provider that holds the OLD model."""
    for model in ("first", "second"):
        _one_turn(model=model)
    verbs = [a[:2] for a in logged(fake_cli) if a and a[0] == "provider"]
    assert verbs.count(["provider", "add"]) == 2, verbs
    assert ["provider", "remove"] in verbs, verbs
    # the turn must name the provider that actually HOLDS the new model, and
    # the replace has to happen between the two turns rather than after both
    execs = [a for a in logged(fake_cli) if a and a[0] == "exec"]
    assert model_arg(execs[0]) == "custom_provider:rigma/first"
    assert model_arg(execs[-1]) == "custom_provider:rigma/second"


def test_leftovers_are_cleared_before_adding(fake_cli, tmp_path):
    """Every entry Rigma created is Rigma's to remove — otherwise they pile up
    one per change and each `--use` moves further from the id `--model` names."""
    seed_providers(tmp_path, [
        _prov(active=False, url="http://127.0.0.1:9998/v1"),
        _prov("custom_provider:rigma-2", active=False, url="http://127.0.0.1:9999/v1"),
    ])
    _one_turn()
    gone = [a[2] for a in remove_calls(fake_cli)]
    assert gone == ["custom_provider:rigma", "custom_provider:rigma-2"], gone


def test_the_permission_mode_comes_from_the_chat(fake_cli):
    """How much the agent may do unasked is the OWNER's call, so the value has
    to reach the CLI instead of being decided here."""
    for mode in ("full", "smart", "off"):
        list(harness_mcode.drive_turn(
            base_url=BASE, model="local-test", prompt="hi", permission=mode))
    execs = [a for a in logged(fake_cli) if a and a[0] == "exec"]
    assert [a[a.index("--permission") + 1] for a in execs] == \
        ["full", "smart", "off"]


def test_the_default_is_the_mode_that_does_not_ask(fake_cli):
    """Headless has nobody to ask, so a mode that wants to ask FAILS the run.
    The default has to be the one that works; the trade belongs in the UI, not
    in a turn that dies."""
    _one_turn()
    exe = [a for a in logged(fake_cli) if a and a[0] == "exec"][0]
    assert exe[exe.index("--permission") + 1] == "full"


# --------------------------------------------------------------------------
# What happens when mcode updates underneath us


def _one_turn(model="local-test", base=BASE):
    return list(harness_mcode.drive_turn(base_url=base, model=model,
                                         prompt="hi"))


def test_a_marker_mcode_agrees_with_costs_one_ask_per_process(fake_cli,
                                                              tmp_path):
    """The marker saves a Node process per turn, so it must not become a Node
    process per turn itself."""
    seed_marker(tmp_path)
    seed_providers(tmp_path, [_prov()])
    for _ in range(3):
        _one_turn()
    assert provider_calls(fake_cli) == []
    assert len(list_calls(fake_cli)) == 1


def test_an_upstream_that_moves_its_provider_config_is_noticed(fake_cli,
                                                               tmp_path):
    """THE update case, and the reason the marker is only a hint.

    Rigma leaves a marker saying "configured". An mcode release then changes
    where custom providers live — which this project has already done once, from
    `~/.minimax-code` to `~/.minimax`. The marker still says configured, so
    Rigma skips the setup step, and `exec` fails with a model-not-available
    error that names nothing about the real cause. Asking mcode once per process
    is what turns that into a re-add instead of a mystery.
    """
    seed_marker(tmp_path)
    seed_providers(tmp_path, [])
    _one_turn()
    adds = provider_calls(fake_cli)
    assert len(adds) == 1, "a provider mcode no longer has must be configured again"
    assert adds[0][-1] == "--use"


def test_a_provider_that_lost_its_selection_is_configured_again(fake_cli,
                                                                tmp_path):
    """Present but not selected is precisely the state that makes mcode demand a
    MiniMax login for a turn that never leaves this machine."""
    seed_marker(tmp_path)
    seed_providers(tmp_path, [_prov(active=False)])
    _one_turn()
    assert len(provider_calls(fake_cli)) == 1


def test_a_provider_pointing_at_another_port_is_configured_again(fake_cli,
                                                                 tmp_path):
    """Rigma's port can move. A provider aimed at the old one fails in a way
    that reads like a model problem."""
    seed_marker(tmp_path)
    seed_providers(tmp_path, [_prov(url="http://127.0.0.1:9999/v1")])
    _one_turn()
    assert len(provider_calls(fake_cli)) == 1


def test_a_provider_list_that_fails_is_not_taken_as_absent(fake_cli, tmp_path,
                                                           monkeypatch):
    """A transient failure must not be read as "mcode lost the provider" — the
    re-add would dedupe, and `--use` would then activate a provider that
    `--model` does not name."""
    seed_marker(tmp_path)
    monkeypatch.setenv("FAKE_MCODE_LIST_EXIT", "4")
    _one_turn()
    assert provider_calls(fake_cli) == []
    assert remove_calls(fake_cli) == []


def test_a_fresh_process_re_asks_even_though_the_marker_is_there(fake_cli,
                                                                 tmp_path):
    """The marker survives a restart, so it is exactly the thing that can be
    stale across one."""
    seed_marker(tmp_path)
    seed_providers(tmp_path, [_prov()])
    _one_turn()
    assert len(list_calls(fake_cli)) == 1
    # a second turn in the SAME process must not ask again
    _one_turn()
    assert len(list_calls(fake_cli)) == 1


# --------------------------------------------------------------------------
# Stopping a turn


def test_a_cancel_stops_a_silent_turn_without_losing_the_session(fake_cli,
                                                                 monkeypatch):
    """An interrupt must cost the TURN, not the thread.

    The child says one thing and then goes quiet — which is what mcode does
    while it thinks, and therefore the state a stop has to work in. A stop that
    only takes effect at the next output line would not be a stop at all.

    The session id is recorded before the silence, so the next turn continues
    this conversation instead of starting the agent over. Losing it would make
    "stop" mean "forget everything", which is the opposite of what it is for.
    """
    monkeypatch.setenv("FAKE_MCODE_HANG", "1")
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([
        {"type": "session.started", "sessionId": "mvs_interrupted"}]))
    # R3-HARN-7: this asserts the EXACT notice list, so it must not depend on
    # whether an earlier test in the same process already latched the one-shot
    # drift notice. `_DRIFT_SAID` is process-global; without this the test passes
    # alone and fails in a full run.
    monkeypatch.setattr(harness_mcode, "_DRIFT_SAID", False)
    cancel = threading.Event()
    state: dict = {}
    got: list = []

    def _run():
        got.extend(harness_mcode.drive_turn(
            base_url="http://127.0.0.1:11500/v1", model="local-test",
            prompt="hi", state=state, cancel=cancel))

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(1.5)
    cancel.set()
    t.join(timeout=25)

    assert not t.is_alive(), "a cancel must land without the child speaking"
    assert state.get("session_id") == "mvs_interrupted"
    assert [e.text for e in got if e.kind == "notice"] == ["stopped"]
    assert [e.text for e in got if e.kind == "error"] == []


def test_stopping_is_not_reported_as_a_failure(fake_cli, monkeypatch):
    """Nothing failed, so nothing may be reported as an error — a red line in
    the transcript would make a deliberate stop look like a broken backend."""
    monkeypatch.setenv("FAKE_MCODE_HANG", "1")
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([]))
    cancel = threading.Event()
    got: list = []

    def _run():
        got.extend(harness_mcode.drive_turn(
            base_url="http://127.0.0.1:11500/v1", model="local-test",
            prompt="hi", cancel=cancel))

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(1.0)
    cancel.set()
    t.join(timeout=25)
    assert not t.is_alive()
    assert [e.kind for e in got] == ["notice"], got


def test_no_cancel_means_the_turn_runs_to_the_end(fake_cli):
    """The event is opt-in: a caller that does not pass one must see the turn
    finish normally, not be cut short by a stray default."""
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="local-test", prompt="hi"))
    # `text` and `state` only. The `state`/`usage` event is deliberate: mcode has
    # always reported a turn's token cost, Rigma wrote it into the adapter state
    # and then read only `session_id` back out, so the number was collected and
    # discarded. This assertion used to be `== {"text"}`, which would have made
    # that fix look like a regression.
    assert {e.kind for e in got} == {"text", "state"}, got
    assert "".join(e.text for e in got) == "hello from dsh"
    usage = [e for e in got if e.kind == "state"]
    assert len(usage) == 1
    assert usage[0].event == "usage"
    # R5-MCODE-ERR: the envelope also carries `durationMs`/`model`/
    # `usageIncomplete` as SIBLINGS of `usage`, and all three were discarded while
    # the token count survived. So this asserts the token keys rather than an exact
    # dict — an exact match would now fail for the right reason and make a real
    # improvement look like a regression.
    for k, v in {"inputTokens": 8, "outputTokens": 3, "totalTokens": 11}.items():
        assert usage[0].data.get(k) == v, (k, usage[0].data)
    # And the extras must actually be there, or the fix did nothing.
    assert usage[0].data.get("durationMs") is not None, usage[0].data
    assert usage[0].data.get("model"), usage[0].data


def test_the_turns_usage_is_reported_rather_than_only_remembered(fake_cli):
    """mcode reports `usage` on `turn.completed`. The adapter stored it in
    `state` and nothing ever read it, so a long agent turn's token cost was
    collected and thrown away — the one number a reader most wants."""
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="local-test", prompt="hi",
        state={}))
    usage = [e for e in got if e.kind == "state" and e.event == "usage"]
    assert len(usage) == 1
    assert usage[0].data["totalTokens"] == 11


def test_a_turn_with_no_usage_reports_none(fake_cli, monkeypatch):
    """An empty or absent usage block must not emit a blank event, which would
    render as a row of zeros."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([
        _ev(1, "exec.started"), _ev(2, "session.started"), _ev(3, "turn.started"),
        _ev(4, "turn.completed"),
        _ev(5, "exec.completed", result={"status": "succeeded", "output": "ok"}),
    ]))
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e for e in got if e.kind == "state"] == []


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


# --------------------------------------------------------------------------
# Rigma's one channel into the agent: the agent's OWN instruction file


# --------------------------------------------------------------------------
# Handing Rigma's own tools to the arm (MCP)


def seed_docs(tmp_path):
    """Documents are indexed — the sidecar has recorded its port."""
    d = tmp_path / "home" / "rag"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sidecar.json").write_text('{"port": 11699}', encoding="utf-8")


def mcp_file(tmp_path):
    return tmp_path / "home" / "mcode" / "mcp.json"


def test_the_server_is_registered_even_with_no_documents(fake_cli, tmp_path):
    """This is the whole point of item 5: Rigma's own tool reaching an external
    agent WITHOUT Rigma touching a message the agent sends to its model. MCP
    makes Rigma a tool PROVIDER; the arm decides whether to call it.

    The gate used to be `is a RAG sidecar live`, which is only one of four roster
    entries — so with nothing indexed the arm also lost `remember`, `recall` and
    `undo_last_change`, none of which need documents. Registration now asks what
    would actually be offered.

    That does change the cost: the server is spawned on every turn rather than
    only when documents exist. It is worth it — memory and undo are the two
    things the arm has no equivalent of — but it is a real trade, not a free one.
    """
    harness_mcode.ensure_mcp(str(tmp_path))
    entry = json.loads(mcp_file(tmp_path).read_text(encoding="utf-8"))
    assert "rigma" in entry["mcpServers"]


def test_the_registration_points_at_rigmas_own_interpreter(fake_cli, tmp_path):
    """A bare `python` would be whatever the arm's PATH resolves to. The server
    imports Rigma, and the interpreter that has Rigma installed is the one
    already running this."""
    seed_docs(tmp_path)
    harness_mcode.ensure_mcp(r"C:\work")
    spec = json.loads(mcp_file(tmp_path).read_text(encoding="utf-8"))["mcpServers"]["rigma"]
    assert spec["command"] == sys.executable
    assert spec["args"] == ["-m", "rigma.mcp_server"]
    # passed, not guessed: a tool on the wrong directory is worse than a refusal
    assert spec["env"]["RIGMA_MCP_WORKSPACE"] == r"C:\work"
    assert spec["env"]["RIGMA_HOME"] == str(tmp_path / "home")
    # granted at the registration site rather than assumed by the server, which
    # is pessimistic by default. Only `undo_last_change` needs it.
    assert spec["env"]["RIGMA_MCP_ALLOW_CODE"] == "1"


def test_the_registration_is_taken_away_when_there_is_nothing_to_offer(
        fake_cli, tmp_path, monkeypatch):
    """An MCP server with an empty roster still costs a process launch on every
    turn. Leaving the pointer behind would pay that for nothing.

    Tested by emptying the roster rather than by removing the documents, because
    the roster no longer depends on documents: `remember` and `recall` are always
    offerable, so no on-disk state makes it empty. The mechanism still has to
    work, which is what this pins.
    """
    from rigma import mcp_server

    seed_docs(tmp_path)
    harness_mcode.ensure_mcp(str(tmp_path))
    assert "rigma" in json.loads(
        mcp_file(tmp_path).read_text(encoding="utf-8"))["mcpServers"]

    monkeypatch.setattr(mcp_server, "offered", lambda **kw: [])
    harness_mcode.ensure_mcp(str(tmp_path))
    left = json.loads(mcp_file(tmp_path).read_text(encoding="utf-8"))
    assert "rigma" not in (left.get("mcpServers") or {})


def test_another_mcp_server_in_the_file_is_left_alone(fake_cli, tmp_path):
    """The file is Rigma's to MANAGE, not Rigma's to own exclusively — it is the
    same `mcpServers` notation the client side reads, so a user may well have put
    their own server there."""
    mcp_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    mcp_file(tmp_path).write_text(json.dumps({"mcpServers": {
        "theirs": {"command": "npx", "args": ["-y", "something"]}}}),
        encoding="utf-8")
    seed_docs(tmp_path)
    harness_mcode.ensure_mcp(str(tmp_path))
    entry = json.loads(mcp_file(tmp_path).read_text(encoding="utf-8"))
    assert entry["mcpServers"]["theirs"]["command"] == "npx"
    assert "rigma" in entry["mcpServers"]


def test_the_registration_is_not_rewritten_when_it_is_already_right(fake_cli,
                                                                    tmp_path):
    """This runs on EVERY turn, so it must not churn the file's mtime each time
    — and it must not fight a file that is already correct."""
    seed_docs(tmp_path)
    harness_mcode.ensure_mcp(str(tmp_path))
    before = mcp_file(tmp_path).stat().st_mtime_ns
    for _ in range(3):
        harness_mcode.ensure_mcp(str(tmp_path))
    assert mcp_file(tmp_path).stat().st_mtime_ns == before


def test_a_file_that_is_not_json_is_replaced_rather_than_crashing(fake_cli,
                                                                  tmp_path):
    """A half-written file from a killed process must cost the registration, not
    the turn."""
    mcp_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    mcp_file(tmp_path).write_text("{not json", encoding="utf-8")
    seed_docs(tmp_path)
    harness_mcode.ensure_mcp(str(tmp_path))
    assert "rigma" in json.loads(
        mcp_file(tmp_path).read_text(encoding="utf-8"))["mcpServers"]


def test_a_turn_registers_the_server(fake_cli, tmp_path):
    """The hook is `drive_turn`, so the registration cannot be something a
    caller has to remember."""
    seed_docs(tmp_path)
    _one_turn()
    assert "rigma" in json.loads(
        mcp_file(tmp_path).read_text(encoding="utf-8"))["mcpServers"]


def test_rigmas_note_lands_in_the_agents_own_data_dir(fake_cli, tmp_path):
    """mcode has no append/override mechanism for its prompt — no flag, no
    config key, no env var. `<dataDir>/AGENTS.md` is the documented channel, and
    Rigma owns the data dir, so that is where a note about the agent's world
    goes. Never into the prompt, and never into the user's project."""
    list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    note = tmp_path / "home" / "mcode" / "AGENTS.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    assert harness_mcode._AGENTS_MARKER in text
    # the one fact the agent cannot discover for itself
    assert "not a frontier cloud model" in text


def test_the_note_describes_the_world_and_claims_nothing_else(fake_cli,
                                                              tmp_path):
    """The agent's instruction set is the thing worth borrowing. A note that
    told it HOW to work would dilute the benchmark this backend was chosen
    for."""
    harness_mcode.ensure_agents_md()
    text = (tmp_path / "home" / "mcode" / "AGENTS.md").read_text(encoding="utf-8")
    assert "environment description" in text
    assert "does not rewrite" in text
    assert "your tool roster" in text


def test_a_users_edit_to_the_note_is_not_fought(fake_cli, tmp_path):
    """The file sits in a directory Rigma owns, but what it says about the
    agent's world is worth being able to correct by hand."""
    note = tmp_path / "home" / "mcode" / "AGENTS.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# my own note, no marker\n", encoding="utf-8")
    harness_mcode.ensure_agents_md()
    assert note.read_text(encoding="utf-8") == "# my own note, no marker\n"


def test_the_note_is_refreshed_while_rigma_still_owns_it(fake_cli, tmp_path):
    note = tmp_path / "home" / "mcode" / "AGENTS.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(harness_mcode._AGENTS_MARKER + "\n<!-- version: 0 -->\n"
                    "stale\n", encoding="utf-8")
    harness_mcode.ensure_agents_md()
    assert "stale" not in note.read_text(encoding="utf-8")


def test_a_failure_reports_what_the_exit_code_means(fake_cli, monkeypatch):
    """The number alone sends a reader to look it up."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", "[]")
    monkeypatch.setenv("FAKE_MCODE_EXIT", "4")
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.kind for e in got] == ["error"]
    assert "exited 4" in got[0].text and "the run failed" in got[0].text


def test_a_run_that_ends_unsuccessfully_is_not_called_a_success(
        fake_cli, monkeypatch):
    """A step limit or a cancellation is reported in the RESULT while the
    process still exits 0. The documented advice is to check both."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([
        _ev(1, "exec.started"), _ev(2, "session.started"),
        _ev(3, "turn.started"),
        _ev(4, "item.completed", item={"id": "m:message",
                                       "type": "agent_message",
                                       "content": "half a job"}),
        _ev(5, "exec.completed", result={"status": "limit_exceeded"}),
    ]))
    monkeypatch.setenv("FAKE_MCODE_EXIT", "0")
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.text for e in got if e.kind == "text"] == ["half a job"]
    errs = [e.text for e in got if e.kind == "error"]
    assert errs and "limit_exceeded" in errs[0]


def test_a_turn_that_reported_its_own_failure_is_not_reported_twice(
        fake_cli, monkeypatch):
    """`turn.failed` already said what went wrong. A second error line for the
    same event would read as two failures."""
    monkeypatch.setenv("FAKE_MCODE_EVENTS", json.dumps([
        _ev(1, "turn.failed", status="failed",
            error={"message": "upstream refused"}),
        _ev(2, "exec.completed", result={"status": "failed"}),
    ]))
    monkeypatch.setenv("FAKE_MCODE_EXIT", "0")
    got = list(harness_mcode.drive_turn(
        base_url="http://127.0.0.1:11500/v1", model="m", prompt="hi"))
    assert [e.kind for e in got] == ["error"]
    assert "upstream refused" in got[0].text


# --------------------------------------------------------------------------
# a fake child, so the read loop can be driven without mcode or a process


class _RecordingIO(io.StringIO):
    """A stdout/stderr that records the size each readline asked for."""
    def __init__(self, text):
        super().__init__(text)
        self.sizes = []

    def readline(self, size=-1):
        self.sizes.append(size)
        return super().readline(size)


def _fake_exec(monkeypatch, *, stdout="", stderr="", code=0):
    """Drive the real `drive_turn` against a fake child: no process starts."""
    class FakeProc:
        def __init__(self):
            self.stdout = _RecordingIO(stdout)
            self.stderr = _RecordingIO(stderr)
            self.returncode = code

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    fake = FakeProc()
    monkeypatch.setattr(harness_mcode, "bin_path", lambda: "mcode")
    monkeypatch.setattr(harness_mcode, "ensure_provider",
                        lambda *a, **k: ("custom_provider:rigma", ""))
    monkeypatch.setattr(harness_mcode, "ensure_agents_md", lambda: None)
    monkeypatch.setattr(harness_mcode, "ensure_mcp", lambda cwd="": None)
    monkeypatch.setattr(harness_mcode, "subprocess",
                        types.SimpleNamespace(Popen=lambda *a, **k: fake,
                                              DEVNULL=_sp.DEVNULL,
                                              PIPE=_sp.PIPE))
    return fake


def test_a_line_over_the_bound_is_skipped_not_buffered(monkeypatch):
    """`for line in proc.stdout` held one whole line in memory before parsing,
    and json.loads made a second copy; mcode puts tool results inside those
    lines, so one large result lived twice inside the process that also holds
    every chat (09-4)."""
    monkeypatch.setattr(harness_mcode, "_FRAME_MAX", 200)
    stream = io.StringIO("x" * 400 + "\n" + json.dumps(
        {"type": "exec.completed",
         "result": {"status": "succeeded", "output": "ok"}}) + "\n")
    got = list(harness_mcode._bounded_lines(stream))
    assert got[0] == ("", True)
    assert got[1][1] is False
    assert json.loads(got[1][0])["result"]["output"] == "ok"


def test_an_unterminated_oversize_line_ends_the_stream(monkeypatch):
    """An endless line must not spin the drain forever: bounded memory is the
    point, and the watchdog is what ends the process."""
    monkeypatch.setattr(harness_mcode, "_FRAME_MAX", 8)

    class Endless(io.StringIO):
        def readline(self, size=-1):
            return "x" * (size if size and size > 0 else 16)

    assert list(harness_mcode._bounded_lines(Endless(""))) == [("", True)]


def test_drive_turn_reads_mcode_with_a_hard_frame_bound(monkeypatch):
    """The F55 bound was applied to the MCP client and not to mcode's stdout
    (09-4). Every read must carry it."""
    monkeypatch.setattr(harness_mcode, "_FRAME_MAX", 200)
    fake = _fake_exec(monkeypatch, stdout=(
        "x" * 400 + "\n" + json.dumps(
            {"type": "exec.completed",
             "result": {"status": "succeeded", "output": "ok"}}) + "\n"))
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    assert fake.stdout.sizes and all(s == 201 for s in fake.stdout.sizes), \
        fake.stdout.sizes
    assert [e.text for e in got if e.kind == "text"] == ["ok"]
    assert any(e.kind == "notice" and "skipped" in e.text for e in got), got


def test_an_exit_zero_with_no_stream_is_an_error_not_an_empty_success(
        monkeypatch):
    """mcode can exit 0 without ever emitting `exec.completed` (a shim that
    fails silently, an output-format mismatch). The tail checked only stopped/
    killed/exit code/final_status, so the generator ended silently and the
    caller saved an empty assistant reply as a finished turn (09-7)."""
    _fake_exec(monkeypatch, stdout="", stderr="shim said nothing\n", code=0)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    assert [e.kind for e in got] == ["error"], got
    assert "0" in got[0].text and "without completing" in got[0].text


def test_a_completed_stream_is_not_reported_as_unfinished(monkeypatch):
    """The other half of the same rule: `exec.completed` is the end marker, and
    a turn that has one must not gain a spurious error."""
    _fake_exec(monkeypatch, stdout=json.dumps(
        {"type": "exec.completed",
         "result": {"status": "succeeded", "output": "hi"}}) + "\n")
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    assert [e.kind for e in got] == ["text"], got


# --- R4-MCODE-1: the SERVER hop, which had no test at all --------------------
#
# WHY THIS EXISTS. The adapter has emitted `event="goal"` and `event="todos"`
# since it was written, and `serve.py`'s state chain matched DSH's slash names
# only (`goal/change`, `todo/write`). Every mcode goal and todo list therefore
# fell through to the "an event name from a newer DSH than this Rigma knows is
# dropped" branch and reached the UI as NOTHING.
#
# The adapter tests could not catch it, because they stop at the adapter: they
# assert the TurnEvent is produced, which was always true. A typo anywhere in
# that elif chain would have left every Python test green, so these drive a real
# request through the real app and read the SSE that comes back.


def _drive_events(monkeypatch, tmp_path, events):
    """Run one chat turn whose adapter yields exactly `events`."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: True)

    def _fake_drive(**kw):
        for e in events:
            yield e

    monkeypatch.setattr(harness_mcode, "drive_turn", _fake_drive)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
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
    return r.text


def test_an_mcode_goal_reaches_the_ui_as_a_goal_event(monkeypatch, tmp_path):
    """`event="goal"` must be forwarded, not dropped as an unknown name."""
    body = _drive_events(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="goal",
                          data={"goalId": "g1", "objective": "ship it",
                                "status": "active", "tokensUsed": 5}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "event: goal" in body, body
    assert "ship it" in body


def test_an_mcode_todo_list_reaches_the_ui_as_a_todos_event(monkeypatch, tmp_path):
    """`event="todos"` must be forwarded, and the list preserved."""
    body = _drive_events(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="todos",
                          data={"todos": [{"content": "step one",
                                           "status": "in_progress"}]}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "event: todos" in body, body
    assert "step one" in body


def test_an_mcode_subagent_reaches_the_ui_with_its_ids(monkeypatch, tmp_path):
    """The wrapped payload keeps the ids the fold needs to key a row by."""
    body = _drive_events(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="subagent",
                          data={"taskId": "bg_1", "subSessionId": "s1",
                                "name": "explore", "status": "started"}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "event: subagent" in body, body
    assert "s1" in body and "explore" in body


def test_an_unknown_state_event_is_still_dropped(monkeypatch, tmp_path):
    """The other half of the contract: a name this build does not know is not
    guessed at. A wrong rendering is worse than none."""
    body = _drive_events(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="something/from-the-future",
                          data={"x": 1}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "something/from-the-future" not in body, body


# --- R5-PERSIST: the agent's durable state must survive the turn -------------
#
# Goals, todos and plan mode OUTLIVE the turn that reported them, and the UI drew
# them from the live turn only — so the whole panel vanished on reload even though
# the agent's state had not changed. The server now stores them on the session row.
#
# These drive a real turn through the real app and then read the session back, so
# they cover the hop that had no coverage at all before R4: adapter -> server ->
# stored row.


def _turn_then_read_session(monkeypatch, tmp_path, events):
    """Run one chat turn, then return (sse_body, stored_session)."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: True)

    def _fake_drive(**kw):
        for e in events:
            yield e

    monkeypatch.setattr(harness_mcode, "drive_turn", _fake_drive)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
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
        got = c.get(f"/api/sessions/{s['id']}").json()
    return r.text, got


def test_a_goal_is_stored_on_the_session_and_returned_by_get(monkeypatch, tmp_path):
    """The reload path: what the panel needs must come back from the server."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="goal",
                          data={"goalId": "g1", "objective": "ship it",
                                "status": "active"}),
        harness.TurnEvent("text", text="done"),
    ])
    assert got["agent_state"]["goal"]["objective"] == "ship it"


def test_the_stored_goal_is_the_raw_payload_not_a_reshaped_one(monkeypatch, tmp_path):
    """`chat/goal.ts` owns the two backends' field names. Reshaping here would mean
    the server guessing at a schema it does not own, and the stored copy drifting
    from the live one."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="goal",
                          data={"goalId": "g1", "objective": "ship it",
                                "status": "active", "tokensUsed": 7}),
        harness.TurnEvent("text", text="done"),
    ])
    # mcode's own key names survive verbatim, including ones Rigma never reads.
    assert got["agent_state"]["goal"]["goalId"] == "g1"
    assert got["agent_state"]["goal"]["tokensUsed"] == 7


def test_a_todo_list_is_stored_as_the_whole_list(monkeypatch, tmp_path):
    """`todo_write` REPLACES the list every call, so the stored copy is an
    assignment, never a merge."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="todos",
                          data={"todos": [{"content": "one", "status": "completed"},
                                          {"content": "two", "status": "pending"}]}),
        harness.TurnEvent("text", text="done"),
    ])
    stored = got["agent_state"]["todos"]
    assert [x["content"] for x in stored] == ["one", "two"]
    assert stored[1]["status"] == "pending"


def test_a_second_todo_write_replaces_rather_than_appends(monkeypatch, tmp_path):
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="todos",
                          data={"todos": [{"content": "old", "status": "pending"}]}),
        harness.TurnEvent(kind="state", event="todos",
                          data={"todos": [{"content": "new", "status": "pending"}]}),
        harness.TurnEvent("text", text="done"),
    ])
    assert [x["content"] for x in got["agent_state"]["todos"]] == ["new"]


def test_plan_mode_is_stored(monkeypatch, tmp_path):
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="plan/mode", data={"active": True}),
        harness.TurnEvent("text", text="done"),
    ])
    assert got["agent_state"]["plan_mode"] is True


def test_a_goal_clear_does_not_resurrect_the_goal_on_reload(monkeypatch, tmp_path):
    """The clear tombstone carries no objective. Storing it as-is would mean a
    reload normalises it, finds no objective, and ... shows nothing — but only by
    accident. Storing an explicit None is what the panel actually showed, and it
    keeps the tombstone out of the durable copy."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="goal",
                          data={"goalId": "g1", "objective": "ship it"}),
        harness.TurnEvent(kind="state", event="goal",
                          data={"operation": "clear", "cleared": True}),
        harness.TurnEvent("text", text="done"),
    ])
    assert got["agent_state"]["goal"] is None


def test_subagents_are_deliberately_not_persisted(monkeypatch, tmp_path):
    """A subagent row names a child process belonging to the turn that spawned it.
    Restoring one after a reload would draw "running" for a child that is long
    gone, so it is turn-scoped ON PURPOSE — the same decision the frontend makes.
    This test exists so that adding it later is a decision, not an accident."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="subagent",
                          data={"taskId": "bg_1", "subSessionId": "s1",
                                "name": "explore", "status": "started"}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "subagents" not in got["agent_state"]
    assert "subagent" not in got["agent_state"]


def test_usage_is_not_persisted(monkeypatch, tmp_path):
    """A per-step number is not a fact about the conversation, and it arrives every
    step — persisting it would turn one turn into a write storm."""
    _body, got = _turn_then_read_session(monkeypatch, tmp_path, [
        harness.TurnEvent(kind="state", event="usage", data={"promptTokens": 5}),
        harness.TurnEvent("text", text="done"),
    ])
    assert "usage" not in got["agent_state"]


def test_a_fresh_session_starts_with_an_empty_agent_state(monkeypatch, tmp_path):
    """Named in the defaults rather than left absent, so a reader that subscripts
    it on a chat that predates the field does not KeyError."""
    from rigma import sessions

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create(title="fresh")
    assert s["agent_state"] == {}


# --- R5-TOOLID: the call id must reach serve.py ------------------------------
#
# serve.py falls back to the tool NAME when no id arrives, and says in its own
# comment what that costs: "Falling back to the tool NAME collapses two calls to
# the same tool in one turn onto a single chip, because the frontend matches a
# result to the first still-running chip with that id." mcode runs its tools in
# PARALLEL, so two `read`s in one turn is ordinary — and every result landed on
# whichever chip came first. The frontend also keys chips by that id
# (`Transcript.tsx` `key={c.id}`), so the fallback produced duplicate React keys.
#
# The sibling adapter always passed it; this one did not. Pinned so it cannot
# regress, because nothing else in the suite would notice: the events are still
# produced, they are just anonymous.


def test_a_tool_call_and_its_result_carry_the_same_call_id():
    got = fold(TOOL_TURN)
    call = next(e for e in got if e.kind == "tool")
    result = next(e for e in got if e.kind == "tool_result")
    assert call.data and result.data, got
    assert call.data["id"] == "call_probe_1"
    assert result.data["id"] == call.data["id"]


def test_two_parallel_calls_to_one_tool_stay_distinguishable():
    """The case the id exists for. Without it serve.py names both events by the
    tool, and the frontend folds them onto one chip."""
    two = [
        _item(1, "item.started",
              {"id": "call_a", "type": "tool_call",
               "toolCall": {"id": "call_a", "name": "read", "status": 4}}),
        _item(2, "item.started",
              {"id": "call_b", "type": "tool_call",
               "toolCall": {"id": "call_b", "name": "read", "status": 4}}),
        _item(3, "item.completed",
              {"id": "call_a", "type": "tool_call",
               "toolCall": {"id": "call_a", "name": "read", "status": 3,
                            "input": {"path": "a"},
                            "output": {"content": [{"type": "text", "text": "A"}]}}}),
        _item(4, "item.completed",
              {"id": "call_b", "type": "tool_call",
               "toolCall": {"id": "call_b", "name": "read", "status": 3,
                            "input": {"path": "b"},
                            "output": {"content": [{"type": "text", "text": "B"}]}}}),
    ]
    got = fold(two)
    ids = [e.data["id"] for e in got if e.kind in ("tool", "tool_result")]
    # Each call's result follows its own call, in the order mcode finished them —
    # NOT grouped by call. What matters is that the four events carry two distinct
    # ids rather than collapsing onto the tool name.
    assert ids == ["call_a", "call_a", "call_b", "call_b"], ids
    assert len(set(ids)) == 2


# --- R5-MCODE-DEADEND: a blocked chat must say so, and say what works --------
#
# `mcode exec` has no interaction host, so a turn that asks the user something
# cannot be answered — a limitation. The DEAD END is that the question is never
# cleared: mcode's guard (`run-exec-command-EXGGKYPR.js`, `hr`) refuses to start a
# session that still has a pending questionnaire OR pending permission request, and
# it runs BEFORE the stream projector is constructed. So every LATER turn in that
# chat fails the same way, in seconds, with no stream-json output at all.
#
# Verified in the 0.5.4 bundle:
#   let[n,s]=await Promise.all([e.getPendingQuestionnaire(r,t.sessionId),
#                               e.listPendingPermissions()]),i=s.some(...)
#   if(!n&&!i)return;
#   throw new h("runtime",`Session ${t.sessionId} has ${o} and requires an
#               interactive host. Continue it in the TUI or ACP.`)
#
# Rigma cannot answer the question (that needs mcode's TUI or ACP, and ACP needs
# `mcode login`). What it CAN do is stop reporting a cryptic exit code and say the
# one thing that works: start a new chat.

_DEAD_END_STDERR = ("mcode exec failed: Session sess_1 has a pending questionnaire "
                    "and requires an interactive host. Continue it in the TUI or ACP.")


def test_a_blocked_chat_is_told_it_cannot_continue(monkeypatch):
    _fake_exec(monkeypatch, stdout="", stderr=_DEAD_END_STDERR, code=1)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    errs = [e for e in got if e.kind == "error"]
    assert errs, [e.kind for e in got]
    text = errs[0].text
    # It must name the recovery, and be explicit that a retry cannot work —
    # otherwise the obvious response is to send the message again, which fails
    # identically and looks like a broken product.
    assert "NEW chat" in text or "new chat" in text, text
    assert "cannot" in text.lower(), text


def test_the_dead_end_is_not_reported_as_a_bare_exit_code(monkeypatch):
    """The whole point: `mcode exited 1` is what the user used to get."""
    _fake_exec(monkeypatch, stdout="", stderr=_DEAD_END_STDERR, code=1)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    text = next(e.text for e in got if e.kind == "error")
    assert "mcode exited" not in text, text


def test_a_pending_permission_request_is_the_same_dead_end(monkeypatch):
    """The same guard covers `--permission smart` raising a permission request,
    so the message names the mode that avoids that half."""
    _fake_exec(monkeypatch, stdout="",
               stderr=("mcode exec failed: Session s has a pending permission "
                       "request and requires an interactive host."), code=1)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    text = next(e.text for e in got if e.kind == "error")
    assert "full" in text, text


def test_the_dead_end_is_detected_on_the_exit_zero_path_too(monkeypatch):
    """The guard throws before the projector exists, so this usually lands as
    exit 0 with no `exec.completed` and the reason on stderr — not as a non-zero
    exit. Both paths are checked because which one it is depends on how the CLI
    surfaced the throw."""
    _fake_exec(monkeypatch, stdout="", stderr=_DEAD_END_STDERR, code=0)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    text = next(e.text for e in got if e.kind == "error")
    assert "new chat" in text.lower(), text


def test_an_ordinary_failure_is_still_a_plain_exit_message(monkeypatch):
    """The detection must not swallow unrelated failures — a real crash has to
    keep reporting its exit code and stderr, or diagnosis gets worse."""
    _fake_exec(monkeypatch, stdout="", stderr="something else went wrong", code=1)
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    text = next(e.text for e in got if e.kind == "error")
    assert "mcode exited 1" in text, text


def test_the_detector_does_not_fire_on_ordinary_prose():
    """A false positive would replace a real error with recovery advice for a
    problem the user does not have."""
    for prose in ("the tool asked for input and continued",
                  "interactive host is fine",
                  "no pending questionnaire"):
        assert harness_mcode._interaction_dead_end(prose) == "", prose


# --- R5-MCODE-ERR / CONTENT: fields the projector sends and the adapter dropped
#
# The projector's own event construction (run-exec-command-EXGGKYPR.js) spreads
# more into each envelope than the adapter read back. None of these change what a
# turn DOES; all of them change what a reader can tell about it.


def _turn_failed(**err):
    return {"type": "turn.failed", "status": "failed", "error": err,
            "durationMs": 1234}


def test_a_failed_turn_keeps_the_error_code_and_category():
    """`code` is the only machine-readable way to tell a dead end from a
    transient fault. mcode rewrites an awaiting-user-continuation turn to
    code INTERACTION_NOT_AVAILABLE, retryable false — and the adapter reported
    just the message, so nothing downstream could branch on it."""
    ev = harness_mcode.map_event(_turn_failed(
        category="runtime", code="INTERACTION_NOT_AVAILABLE",
        message="The Runtime requested interaction from a non-interactive Exec host.",
        retryable=False), {})
    assert ev[0].kind == "error"
    assert "INTERACTION_NOT_AVAILABLE" in ev[0].text, ev[0].text
    assert "runtime" in ev[0].text, ev[0].text


def test_a_failed_turn_says_when_mcode_calls_it_not_retryable():
    """mcode's own answer to "is trying again worth anything"."""
    ev = harness_mcode.map_event(_turn_failed(
        category="runtime", code="X", message="boom", retryable=False), {})
    assert "not retryable" in ev[0].text, ev[0].text


def test_a_retryable_failure_does_not_claim_otherwise():
    ev = harness_mcode.map_event(_turn_failed(
        category="internal", code="Y", message="boom", retryable=True), {})
    assert "not retryable" not in ev[0].text, ev[0].text


def test_a_bare_failure_stays_a_bare_message():
    """No code, no category, no noise — the message must not grow brackets."""
    ev = harness_mcode.map_event({"type": "turn.failed", "status": "failed"}, {})
    assert ev[0].kind == "error"
    assert ev[0].text == "the mcode turn failed", ev[0].text
    assert "[" not in ev[0].text, ev[0].text


def test_a_non_dict_error_is_still_survived():
    """09-R3-1: `.get` on a string raised straight out of the read loop."""
    ev = harness_mcode.map_event({"type": "turn.failed", "error": "plain"}, {})
    assert ev[0].kind == "error"
    assert ev[0].text == "the mcode turn failed", ev[0].text


def _stream(*objs):
    """mcode's stream-json: one JSON object per line."""
    return "".join(json.dumps(o) + "\n" for o in objs)


def test_the_turn_duration_and_model_reach_the_usage_state(monkeypatch):
    """`durationMs` and `model` are SIBLINGS of `usage` in the envelope (the
    projector spreads them in), and both were discarded while the token count
    survived."""
    _fake_exec(monkeypatch, stdout=_stream(
        {"type": "turn.completed", "usage": {"inputTokens": 5},
         "durationMs": 4200, "model": {"modelId": "MiniMax-M2"}},
        {"type": "exec.completed", "result": {"status": "succeeded",
                                              "output": "done"}}))
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    usage = [e for e in got if e.kind == "state" and e.event == "usage"][0]
    assert usage.data["durationMs"] == 4200, usage.data
    assert usage.data["model"] == {"modelId": "MiniMax-M2"}, usage.data
    assert usage.data["inputTokens"] == 5, usage.data


def test_an_incomplete_usage_count_is_flagged_rather_than_trusted(monkeypatch):
    """`usageIncomplete` is mcode saying the token count is not the whole story.
    A reader who cannot see it trusts a number that is low."""
    _fake_exec(monkeypatch, stdout=_stream(
        {"type": "turn.completed", "usage": {"inputTokens": 5},
         "usageIncomplete": True},
        {"type": "exec.completed", "result": {"status": "succeeded",
                                              "output": "done"}}))
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    usage = [e for e in got if e.kind == "state" and e.event == "usage"][0]
    assert usage.data["usageIncomplete"] is True, usage.data


def test_a_usage_state_without_the_extras_is_unchanged(monkeypatch):
    """The common case must not grow keys."""
    _fake_exec(monkeypatch, stdout=_stream(
        {"type": "turn.completed", "usage": {"inputTokens": 5}},
        {"type": "exec.completed", "result": {"status": "succeeded",
                                              "output": "done"}}))
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    usage = [e for e in got if e.kind == "state" and e.event == "usage"][0]
    assert set(usage.data) == {"inputTokens"}, usage.data


def test_a_failed_turn_reports_its_code_through_a_real_turn(monkeypatch):
    """End to end, not just `map_event`: the code has to survive the read loop."""
    _fake_exec(monkeypatch, stdout=_stream(
        {"type": "turn.failed", "status": "failed",
         "error": {"category": "runtime", "code": "INTERACTION_NOT_AVAILABLE",
                   "message": "The Runtime requested interaction from a "
                              "non-interactive Exec host.",
                   "retryable": False}}))
    got = list(harness_mcode.drive_turn(base_url=BASE, model="m", prompt="hi"))
    errs = [e for e in got if e.kind == "error"]
    assert errs, [e.kind for e in got]
    assert "INTERACTION_NOT_AVAILABLE" in errs[0].text, errs[0].text
    assert "not retryable" in errs[0].text, errs[0].text


def test_a_non_text_content_block_is_named_not_dropped():
    """`content` is a block list and only text blocks survived, so a tool that
    returned a picture read as if it had returned nothing."""
    out = {"content": [{"type": "text", "text": "here it is"},
                       {"type": "image", "data": "AAAA"}]}
    text = harness_mcode._flatten(out)
    assert "here it is" in text, text
    assert "image" in text, text


def test_an_image_only_result_does_not_read_as_empty():
    out = {"content": [{"type": "image", "data": "AAAA"}]}
    text = harness_mcode._flatten(out)
    assert text.strip(), "an image-only result must not flatten to nothing"
    assert "image" in text, text


def test_a_text_only_result_does_not_grow_a_note():
    out = {"content": [{"type": "text", "text": "plain"}]}
    assert harness_mcode._flatten(out) == "plain"


def test_the_image_payload_is_not_inlined():
    """Base64 in an SSE text frame would put megabytes into the transcript."""
    out = {"content": [{"type": "image", "data": "A" * 5000}]}
    text = harness_mcode._flatten(out)
    assert "A" * 100 not in text, "the payload must not be inlined"


# --- R6-OKOF: the tool-call status enum, resolved from its definition ---------
#
# WHY THIS IS SETTLED NOW AND WAS NOT BEFORE. `_ok_of` returned None for every
# real call, so every mcode tool chip rendered UNKNOWN — never a tick, never a
# cross. It refused to guess, and that refusal was right: the captured fixture
# pairs status 3 with "Tool not found", but the projector contains no `status: 3`
# literal, so mapping 3 to a cross could have inverted every chip in the
# transcript if 3 had meant success.
#
# The enum's DEFINITION settles it. In the installed @minimax-ai/code 0.5.4,
# `chunk-QWAB5G2D.js` byte 6132401:
#
#     zu = {Start: 1, Finished: 2, Failed: 3, Preparing: 4, Prepared: 5}
#
# is the value domain of the thrift field `tool_call_status`, and the same module
# pins the polarity with `isError: e.tool_call_status === zu.Failed`. The projector
# does not invent the number — it passes the runtime tool-call object through
# verbatim — so this is the authoritative source.
#
# The gap that remains, stated rather than hidden: NO capture in this repo contains
# status 2, so "2 means success" rests on the enum's names plus the runtime's own
# `isError` predicate, not on an observed successful call.

_ACCEPTED = {
    1: None,        # Start      — in flight
    2: True,        # Finished   — the non-error terminal
    3: False,       # Failed     — the error terminal
    4: None,        # Preparing  — in flight
    5: None,        # Prepared   — in flight
}


def test_the_numeric_status_enum_maps_from_its_own_definition():
    """Every value of `zu`, including the two that must NOT become a tick.

    1/4/5 return None deliberately: they are in-flight states, and a tick on a call
    that has not finished is a lie about the transcript.
    """
    for code, want in _ACCEPTED.items():
        got = harness_mcode._ok_of({"status": code})
        assert got is want, f"status {code}: expected {want}, got {got}"


def test_status_three_is_a_failure_and_two_is_a_success():
    """The polarity, stated separately because inverting it is the specific error
    the earlier note was right to fear."""
    assert harness_mcode._ok_of({"status": 3}) is False
    assert harness_mcode._ok_of({"status": 2}) is True


def test_the_string_form_of_the_same_enum_still_works():
    """`Dwn` proves these strings are the SAME enum's other spelling:
        e==="preparing"?Preparing : e==="prepared"?Prepared
        : e==="completed"?Finished : e==="failed"?Failed : Start
    so keeping both arms is not two vocabularies — it is one.
    """
    assert harness_mcode._ok_of({"status": "completed"}) is True
    assert harness_mcode._ok_of({"status": "failed"}) is False
    assert harness_mcode._ok_of({"status": "preparing"}) is None
    assert harness_mcode._ok_of({"status": "prepared"}) is None
    assert harness_mcode._ok_of({"status": "started"}) is None


def test_an_unknown_status_stays_unknown_rather_than_becoming_a_tick():
    """A future item kind must not become a wrong row. The safe half is UNKNOWN."""
    for junk in (None, "", "nonsense", 0, 6, 99, -1, [], {}, "7"):
        assert harness_mcode._ok_of({"status": junk}) is None, junk


def test_a_boolean_status_is_not_read_as_the_enum():
    """`True == 1` in Python, so a careless `int()` arm would map a JSON `true` to
    Start. A bool is not one of this enum's values."""
    assert harness_mcode._ok_of({"status": True}) is None
    assert harness_mcode._ok_of({"status": False}) is None


def test_a_stringified_number_is_still_read_as_the_enum():
    """The runtime accepts string-or-number for this field, so "2" and 2 must not
    disagree. It does not: `str` falls through to the string arms, and "2" is not
    one of them — so this pins the CURRENT behaviour honestly rather than claiming
    a coercion that is not there."""
    assert harness_mcode._ok_of({"status": "2"}) is None, (
        "a stringified code is NOT silently coerced; it is unknown, which is safe")
    assert harness_mcode._ok_of({"status": 2}) is True


def test_the_captured_fixture_pairs_status_three_with_a_failure():
    """The repo's own capture, end to end: status 4 -> 5 -> 1 -> 3, where 3 carries
    "Tool not found". Read from the real fixture rather than restated."""
    import pathlib as _p
    import re
    src = _p.Path(__file__).with_name("test_harness_mcode.py").read_text(encoding="utf-8")
    block = src[src.index("TOOL_TURN"):src.index("TOOL_TURN") + 3000]
    codes = [int(m) for m in re.findall(r'"status":\s*(\d+)', block)]
    assert codes[:4] == [4, 5, 1, 3], codes
    assert "Tool not found" in block
    assert harness_mcode._ok_of({"status": codes[3]}) is False
    for code in codes[:3]:
        assert harness_mcode._ok_of({"status": code}) is None
