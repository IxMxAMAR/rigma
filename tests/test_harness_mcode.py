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
import threading
import time

import pytest

from rigma import harness, harness_mcode, rag


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
    assert {e.kind for e in got} == {"text"}, got
    assert "".join(e.text for e in got) == "hello from dsh"


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
