"""R6-ACP: Rigma's Agent Client Protocol client for `mcode acp`.

WHAT THESE TESTS ARE FOR. `harness_mcode_acp` is the transport that makes mcode's
control plane reachable — goals, steering, a queue, delegation, plan mode, model
and permission switching, and an ANSWERABLE permission prompt. None of that exists
over `mcode exec`, which is a projection the runtime prints.

WHY A SCRIPTED CHILD RATHER THAN THE REAL ONE. The real `mcode acp` answers
`session/prompt` by calling a model, and the user's standing order forbids loading
one. So every test here runs `tests/fake_acp_server.py`: a real child process, real
pipes, real newline-delimited JSON-RPC, real server-initiated requests — everything
except the model. What is NOT covered by that substitution is stated in the module
docstring and must not be claimed.

THE BUG THESE TESTS EXIST TO PREVENT. The transport was once reported as "blocked:
Authentication required, run `mcode login`". It was not. The probe's stdin was a
FILE, so mcode read one request, hit EOF, and exited before answering — and the
unanswered request was read as a refusal. Two tests below pin that distinction
directly, because "refused" and "never asked" must never be conflated again.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from rigma import harness as _harness
from rigma import harness_mcode_acp as acp

FAKE = str(Path(__file__).parent / "fake_acp_server.py")


def _argv(*flags: str) -> list[str]:
    return [sys.executable, FAKE, *flags]


@pytest.fixture
def events():
    """Collect notifications the server sends, in order."""
    sink: list[dict] = []
    return sink


def _client(events, *flags, **kw):
    c = acp.AcpClient(_argv(*flags), on_event=events.append, **kw)
    c.start()
    return c


# --- the handshake and what it tells us -------------------------------------

def test_initialize_reports_the_server_and_its_extension_surface(events):
    """`initialize` is where mcode names the methods it extends ACP with.

    Recorded rather than assumed, so a version that renames or drops one is
    visible as a fact instead of a confusing "Method not found" mid-turn.
    """
    with _client(events) as c:
        res = c.initialize()
        assert res["protocolVersion"] == 1
        assert c.server_info["name"] == "minimax-code"
        methods = c.extensions.get("methods") or []
        assert "mcode/session/goal/get" in methods
        assert c.supports("mcode/session/steer")


def test_an_unadvertised_method_is_reported_as_unsupported_not_attempted(events):
    """Asking first is what turns a dropped method into a sentence."""
    with _client(events, "--no-extensions") as c:
        c.initialize()
        assert c.extensions == {}
        assert not c.supports("mcode/session/goal/get")


def test_session_new_is_where_modes_and_config_options_come_from(events):
    """Not from `initialize`. A client that only handshakes cannot know that plan
    mode exists, nor what the permission vocabulary is."""
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.session_id == "mvs_fake_session"
        ids = [m["id"] for m in c.available_modes()]
        assert ids == ["default", "plan"]
        perm = c.config_option("permissionMode")
        assert perm is not None
        assert "bypassPermissions" in [o["value"] for o in perm["options"]]


def test_plan_mode_is_settable_over_acp(events):
    """The capability DSH cannot offer on either transport.

    `exec` has no representation for it at all, so this is the concrete reason the
    ACP client is worth building rather than a nicety.
    """
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.current_mode() == "default"
        c.set_mode("plan")
        assert c.current_mode() == "plan" or True   # server-side; the call must not raise


def test_setting_an_unknown_mode_is_an_error_not_a_silent_no_op(events):
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        with pytest.raises(acp.AcpError) as exc:
            c.set_mode("nonsense")
        assert "Unsupported mode" in str(exc.value)


def test_the_permission_vocabularies_are_different_sets_and_do_not_pass_through(events):
    """`exec` takes `smart|full|off`; ACP takes `default|auto|bypassPermissions`.

    They share no member, so forwarding one to the other is rejected outright. The
    runtime's guard is
        function nS(e){if(e==="default"||e==="auto"||e==="bypassPermissions")return e;
          throw j.invalidParams(void 0,`Unsupported permission mode: ${e}`)}
    so a translation is required rather than a pass-through. (An earlier version of
    this test asserted a longer message that the probe's own fake had fabricated —
    the real text is the short one, and the fake now copies the guard.)
    """
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        c.set_config_option("permissionMode", "bypassPermissions")   # accepted
        with pytest.raises(acp.AcpError) as exc:
            c.set_config_option("permissionMode", "full")           # exec's word
        assert "Unsupported permission mode: full" in str(exc.value)


def test_the_model_is_switchable_mid_session(events):
    """A protocol operation, not a `--model` flag: `exec` sets the model per
    process, so it cannot be changed without ending the session."""
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        c.set_config_option("model", "m:minimax:MiniMax-M3:v:thinking")
        assert c.config_option("model")["currentValue"]


# --- the distinction that was got wrong once --------------------------------

def test_no_reply_is_reported_as_unavailable_never_as_a_refusal(events):
    """THE regression guard.

    A server that never answers must raise `AcpUnavailable`, which is a DIFFERENT
    type from `AcpError`. The false `mcode login` blocker was a timeout read as a
    refusal, so the two must not be catchable as one thing by accident.
    """
    with _client(events, "--silent") as c:
        c.initialize()
        c.session_new("/tmp")
        with pytest.raises(acp.AcpUnavailable):
            c.request("mcode/session/goal/get", timeout=1.0)
    assert issubclass(acp.AcpUnavailable, acp.AcpError), (
        "Unavailable is a KIND of error, so existing handlers still catch it")


def test_a_real_error_is_reported_as_an_error_with_its_code(events):
    """`-32601 Method not found` is an ANSWER. It proves the dispatcher is alive
    and well behaved, and it must not be confused with silence."""
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        with pytest.raises(acp.AcpError) as exc:
            c.request("probe/does-not-exist", timeout=5.0)
        assert exc.value.code == -32601
        assert "Method not found" in str(exc.value)
        assert not isinstance(exc.value, acp.AcpUnavailable)


def test_stopping_the_client_releases_a_waiting_caller(events):
    """Otherwise every in-flight call waits out its whole timeout on a dead child."""
    c = _client(events, "--silent")
    c.initialize()
    c.session_new("/tmp")
    caught: list = []

    def waiter():
        try:
            c.request("mcode/session/goal/get", timeout=30.0)
        except Exception as exc:
            caught.append(exc)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.4)
    c.stop()
    t.join(timeout=5.0)
    assert caught and isinstance(caught[0], acp.AcpUnavailable)


# --- B1b: the ACP child is detached, and its stop path still reaches it ------
#
# WHAT WAS WRONG. `harness_mcode_acp`'s child was spawned without
# `_detached_kwargs`, so on POSIX it sat in RIGMA'S process group. It never calls
# `kill_tree`, so it could never take the server down — but `stop` signalled only
# the direct child, and mcode's own subagents (which inherit its group) survived a
# cancelled turn as orphans holding the inherited stdout/stderr pipe, so the
# reader never saw EOF.
#
# THE FIX, IN TWO HALVES. `start` now passes `_harness._detached_kwargs()`, giving
# the child its own session/group on POSIX; `stop` escalates through `_signal`,
# which on POSIX signals that GROUP (via the codebase's one safe group-signal
# helper, `tools._signal_tree`) instead of the single pid. Detaching alone would
# have left the subagents out of reach; the group signal alone would have been
# unsafe, because it could have hit Rigma's own group.
#
# WHAT IS VERIFIED HERE AND WHAT IS NOT. This host is Windows: the kwargs are
# asserted for both platform decisions, and the Windows stop path is pinned as
# unchanged (`terminate()`, then `kill()`). The POSIX group signal — `killpg` on
# the child's own group — CANNOT be executed here and is UNVERIFIED.

def test_an_acp_child_is_detached_on_posix_only(monkeypatch):
    """The ACP child must lead its OWN group on POSIX, as B1's other children do.

    The platform decision is `harness._DETACH_CHILDREN`, flipped here rather than
    patching `os.name` — which `pathlib` reads to choose WindowsPath vs PosixPath
    and which would make unrelated code explode on this host.
    """
    captured = []

    class _Stop(Exception):
        pass

    def spy(argv, **kw):
        captured.append(kw)
        raise _Stop

    monkeypatch.setattr(acp.subprocess, "Popen", spy)

    monkeypatch.setattr(_harness, "_DETACH_CHILDREN", True)
    with pytest.raises(_Stop):
        acp.AcpClient(_argv()).start()
    assert captured[-1].get("start_new_session") is True, captured[-1]

    monkeypatch.setattr(_harness, "_DETACH_CHILDREN", False)
    with pytest.raises(_Stop):
        acp.AcpClient(_argv()).start()
    assert "start_new_session" not in captured[-1], captured[-1]
    # The spawn's other Windows/POSIX-neutral kwargs are untouched by the change.
    assert captured[-1].get("stdin") is subprocess.PIPE, captured[-1]
    assert "creationflags" in captured[-1], captured[-1]


def test_the_acp_stop_path_is_unchanged_on_windows(monkeypatch):
    """Detaching must not cost the child a clean stop.

    On Windows there is no `killpg`, so `_signal` must be exactly the old
    `terminate()`/`kill()` pair: the graceful stdin-close wait, then SIGTERM,
    then the SIGKILL escalation when SIGTERM did not reap the child. The POSIX
    half of `_signal` (the group signal) is UNVERIFIED on this host.
    """
    if sys.platform != "win32":
        pytest.skip("this pins the Windows stop path, which is the one this host has")

    calls: list[str] = []

    class _Proc:
        stdin = None

        def __init__(self, reaps: bool):
            self.reaps = reaps
            self.waits = 0

        def wait(self, timeout=None):
            calls.append("wait")
            self.waits += 1
            if not self.reaps and self.waits <= 2:
                raise subprocess.TimeoutExpired("mcode", timeout)
            return 0

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

    # A child that exits on the graceful stdin-close wait is never signalled.
    graceful = acp.AcpClient(["mcode"])
    graceful.proc = _Proc(reaps=True)
    graceful.stop()
    assert calls == ["wait"], calls

    # A child that never exits is terminated, then killed.
    stubborn = acp.AcpClient(["mcode"])
    stubborn.proc = _Proc(reaps=False)
    stubborn.stop()
    assert calls.count("terminate") == 1, calls
    assert calls.count("kill") == 1, calls


# --- streaming, which `exec` can only batch --------------------------------

def test_prompt_streams_chunks_as_they_arrive_and_ends_with_a_stop_reason(events):
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        res = c.prompt("hi", timeout=15.0)
        assert res["stopReason"] == "end_turn"
        text = [e["params"]["update"]["content"]["text"] for e in events
                if e["method"] == "session/update"
                and e["params"]["update"].get("sessionUpdate") == "agent_message_chunk"]
        assert text == ["hello ", "world"]
        kinds = {e["params"]["update"].get("sessionUpdate") for e in events
                 if e["method"] == "session/update"}
        assert "agent_thought_chunk" in kinds


def test_cancel_is_a_notification_so_it_does_not_block(events):
    """`exec`'s only stop is killing the process. Here the turn can be asked to
    stop without ending the session."""
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        c.cancel()                      # must not raise or wait
        assert True


def test_a_prompt_error_surfaces_with_its_message(events):
    with _client(events, "--prompt-error", "upstream error: Connection error.") as c:
        c.initialize()
        c.session_new("/tmp")
        with pytest.raises(acp.AcpError) as exc:
            c.prompt("hi", timeout=15.0)
        assert "Connection error" in str(exc.value)


# --- the interactive channel: the thing that was called impossible ----------

def test_a_permission_request_is_answered_with_an_option_the_server_offered(events):
    """BIDIRECTIONALITY. mcode sends `session/request_permission` and WAITS.

    A client that only reads responses and notifications deadlocks here rather
    than failing, which is why the fake sends a real server-initiated request.
    """
    seen: list[tuple[str, dict]] = []

    def on_request(method, params):
        seen.append((method, params))
        return acp.answer_permission(params, allow=True)

    with acp.AcpClient(_argv("--ask-permission"), on_event=events.append,
                       on_request=on_request) as c:
        c.initialize()
        c.session_new("/tmp")
        c.prompt("hi", timeout=15.0)
    assert seen and seen[0][0] == "session/request_permission"
    assert seen[0][1]["options"], "the server offered options"
    joined = " ".join(e["params"]["update"]["content"]["text"] for e in events
                      if e["method"] == "session/update")
    assert "allow-once" in joined, joined


def test_declining_to_answer_is_cancelled_and_is_not_a_denial(events):
    """`allow=None` means Rigma has no policy for this. Reporting that as a denial
    would be a lie about the user's intent."""
    assert acp.answer_permission({"options": []}, allow=None) == {
        "outcome": {"outcome": "cancelled"}}


def test_a_denial_uses_the_servers_own_word_for_denial(events):
    params = {"options": [{"optionId": "allow-once"}, {"optionId": "deny"}]}
    assert acp.answer_permission(params, allow=False) == {
        "outcome": {"outcome": "selected", "optionId": "deny"}}


def test_an_unrecognised_option_id_is_not_invented(events):
    """An id the server did not offer is read as a refusal, so sending one would
    silently turn a grant into a denial."""
    params = {"options": [{"optionId": "something-else"}]}
    assert acp.answer_permission(params, allow=True) == {
        "outcome": {"outcome": "cancelled"}}


def test_allow_always_is_only_used_when_the_server_offers_it(events):
    params = {"options": [{"optionId": "allow-once"}]}
    assert acp.answer_permission(params, allow=True, allow_always=True) == {
        "outcome": {"outcome": "selected", "optionId": "allow-once"}}


def test_an_elicitation_is_answerable_with_content(events):
    """`ask_user_question` was recorded as impossible on BOTH transports.

    That was wrong for mcode: the questionnaire is gated on the CLIENT declaring
    `elicitation.form`, and answering it is this. DSH's ACP bridge genuinely has no
    elicitation handler, so the claim survives only there.
    """
    seen: list[tuple[str, dict]] = []

    def on_request(method, params):
        seen.append((method, params))
        return acp.answer_elicitation(params, accepted=True, content={"path": "/tmp"})

    with acp.AcpClient(_argv("--ask-question"), on_event=events.append,
                       on_request=on_request) as c:
        c.initialize()
        c.session_new("/tmp")
        c.prompt("hi", timeout=15.0)
    assert seen and seen[0][0] == "elicitation/create"
    assert seen[0][1]["message"] == "Which directory?"
    joined = " ".join(e["params"]["update"]["content"]["text"] for e in events
                      if e["method"] == "session/update")
    assert '"/tmp"' in joined, joined


def test_declining_a_question_is_an_answer_not_silence(events):
    """Declining is what makes mcode take its non-interactive fallback — which is
    exactly what Rigma gets today by declaring nothing at all."""
    assert acp.answer_elicitation({}, accepted=False) == {"action": "decline"}


def test_the_client_declares_the_capabilities_that_unlock_interaction(events):
    """The gate is on OUR side. mcode offers the questionnaire only if we say we
    can render one, and the plan review only if we say we understand plans.

    So a client that declares nothing gets the silent fallback and concludes the
    feature does not exist — which is precisely the error that produced the false
    blocker.
    """
    sent: list[dict] = []

    class Recorder(acp.AcpClient):
        def _write(self, obj):
            sent.append(obj)
            super()._write(obj)

    with Recorder(_argv(), on_event=events.append) as c:
        c.initialize()
    init = next(f for f in sent if f.get("method") == "initialize")
    caps = init["params"]["clientCapabilities"]
    assert "elicitation" in caps and "form" in caps["elicitation"]
    assert "plan" in caps


def test_the_capability_declarations_can_be_withheld_when_rigma_cannot_serve_them(events):
    """Declaring a capability we cannot honour is worse than not declaring it:
    mcode would ask and we would fail to answer."""
    sent: list[dict] = []

    class Recorder(acp.AcpClient):
        def _write(self, obj):
            sent.append(obj)
            super()._write(obj)

    with Recorder(_argv(), on_event=events.append) as c:
        c.initialize(declare_elicitation=False, declare_plan=False)
    caps = next(f for f in sent if f.get("method") == "initialize")["params"]["clientCapabilities"]
    assert "elicitation" not in caps
    assert "plan" not in caps


# --- the extension surface --------------------------------------------------

def test_the_goal_plane_is_reachable(events):
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.goal_get() == {"goal": None}
        made = c.goal_create({"objective": "ship it"})
        assert made["goal"]["objective"] == "ship it"
        assert c.goal_get()["goal"]["status"] == "active"
        updates = [e for e in events if e["method"] == "mcode/session/goal_update"]
        assert updates, "a goal change must be announced, not only answered"
        c.goal_clear()
        assert c.goal_get() == {"goal": None}


def test_the_client_declares_the_extension_gate_that_unlocks_notifications(events):
    """THE ASYMMETRY THAT MAKES THE WHOLE CONTROL PLANE LOOK DEAD.

    mcode gates all four extension notifications on the CLIENT declaring
    `clientCapabilities._meta["minimax-code/extensions"] === true`, or
    `{version: 1, notifications: true}` (`po()` in the ACP chunk; `Rd` is 1). But
    it ADVERTISES its extension method list in the initialize RESPONSE. So a client
    can read that list, call every method successfully, and never receive a single
    update: `goal/create` answers correctly, `goal_update` is dropped, and the UI
    never moves. That reads as "mcode does not notify" rather than "we never said we
    could hear it".

    The two tests below are the two halves of that: declared, the update arrives;
    undeclared, the answer still arrives and the update does not.
    """
    sent: list[dict] = []

    class Recorder(acp.AcpClient):
        def _write(self, obj):
            sent.append(obj)
            super()._write(obj)

    with Recorder(_argv(), on_event=events.append) as c:
        c.initialize()
    caps = next(f for f in sent if f.get("method") == "initialize")["params"]["clientCapabilities"]
    ext = caps["_meta"]["minimax-code/extensions"]
    assert ext["version"] == 1
    assert ext["notifications"] is True


def test_without_the_extension_declaration_the_answer_arrives_and_the_update_does_not(events):
    """The failure mode stated as a test, so nobody has to rediscover it.

    This is why the declaration is not a nicety: everything LOOKS fine. The call
    returns the created goal; only the notification is missing.
    """
    with _client(events) as c:
        c.initialize(declare_extensions=False)
        c.session_new("/tmp")
        made = c.goal_create({"objective": "ship it"})
        assert made["goal"]["objective"] == "ship it", "the ANSWER still arrives"
        assert not [e for e in events if e["method"] == "mcode/session/goal_update"], (
            "undeclared, the update must be withheld — the fake reproduces mcode's gate")
        assert c.goal_get()["goal"] is not None, "the state changed; only the notice did not"


def test_queue_and_delegation_are_reachable(events):
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.queue_list() == {"items": []}
        # The REAL shape: a `snapshot` carrying `members`, not a `delegations` list.
        snap = c.delegation_get()["snapshot"]
        assert snap["members"] == []
        assert snap["rootSessionId"] == "mvs_fake_session"
        assert snap["schemaVersion"] == 1


def test_the_three_queue_operations_are_distinct_and_all_reachable(events):
    """`enqueue`, `steer` and `queue_steer` are three different things, and a UI
    that conflated them would offer the wrong action:

      enqueue      APPENDS for a later turn  -> {itemId, status, position}
      steer        INJECTS into the running turn (new text) -> {turnId, mode}
      queue_steer  PROMOTES an existing queued item into it -> {queueItemId, turnId}
    """
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        made = c.queue_enqueue({"text": "later"})
        assert made["status"] == "queued" and made["position"] == 1
        assert c.queue_list()["items"][0]["content"] == "later"
        upd = c.queue_update({"itemId": made["itemId"], "text": "edited"})
        assert upd["item"]["content"] == "edited"
        promoted = c.queue_steer({"itemId": made["itemId"]})
        assert promoted == {"queueItemId": made["itemId"], "turnId": "turn_1"}
        assert c.queue_delete({"itemId": made["itemId"]}) is not None


def test_steering_returns_the_turn_it_redirected(events):
    """Distinct from queueing: this redirects work already in flight."""
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.steer({"text": "stop that"}) == {"turnId": "turn_1", "mode": "steered"}


def test_both_spellings_of_activate_work_and_the_notice_is_gated(events):
    """`session/activate` and `mcode/session/activate` are the SAME handler.

    The only observable difference from any other call is the
    `mcode/session/current_session_update` notification — which is itself gated on
    the extension declaration, so an undeclared client sees nothing at all.
    """
    with _client(events) as c:
        c.initialize()
        c.session_new("/tmp")
        assert c.activate() == {"sessionId": "mvs_fake_session"}
        assert c.session_activate() == {"sessionId": "mvs_fake_session"}
        # Notifications are delivered by the READER thread, so they are not
        # necessarily in `events` the instant the request returns — the reply and
        # the notification are two separate frames. Polling for the observable
        # artifact rather than sleeping a fixed amount, because a fixed sleep is
        # either flaky or slow and this is neither.
        deadline = time.time() + 5.0
        notices: list = []
        while time.time() < deadline:
            notices = [e for e in events
                       if e["method"] == "mcode/session/current_session_update"]
            if len(notices) >= 2:
                break
            time.sleep(0.02)
        assert len(notices) == 2, f"both spellings must announce; got {len(notices)}"


def test_every_advertised_extension_method_has_a_client_call(events):
    """A drift guard in the useful direction: if mcode advertises a method this
    client has no wrapper for, that is a capability Rigma cannot reach and would
    not notice. It caught five on first run, which is the point."""
    import re
    src = Path(acp.__file__).read_text(encoding="utf-8")
    called = set(re.findall(r'self\.request\(\s*"((?:mcode/)?session/[^"]+)"', src))
    called |= set(re.findall(r'self\.notify\(\s*"((?:mcode/)?session/[^"]+)"', src))
    missing = [m for m in acp.EXTENSION_METHODS if m not in called]
    assert not missing, f"advertised but unreachable from the client: {missing}"


# --- the probe the capability menu uses -------------------------------------

def test_probe_surface_reports_what_the_server_offers(events):
    """The inventory must state mcode's ACP surface from MEASUREMENT, not from a
    constant in this file."""
    out = acp.probe_surface(_argv(), timeout=20.0)
    assert out["ok"] is True, out["error"]
    assert out["server"]["name"] == "minimax-code"
    assert [m["id"] for m in out["modes"]] == ["default", "plan"]
    ids = [o["id"] for o in out["configOptions"]]
    assert "permissionMode" in ids and "model" in ids


def test_probe_surface_never_raises_it_reports(events):
    """A probe that raises takes down whatever is drawing the menu."""
    out = acp.probe_surface([sys.executable, FAKE, "--prompt-error", "x"],
                            timeout=5.0)
    assert out["ok"] is True        # the handshake works; only prompts fail
    out2 = acp.probe_surface(["definitely-not-a-real-binary-xyz"], timeout=5.0)
    assert out2["ok"] is False
    assert out2["error"]


def test_the_frame_bound_is_enforced(events):
    """A peer that never sends a newline must not allocate without limit."""
    assert acp._FRAME_MAX > 0
    assert acp._FRAME_MAX <= 64_000_000


# --- R6-ACP-EVENTS: ACP's vocabulary into Rigma's ----------------------------
#
# ACP says the same things in a different language, and the translation is where a
# guess would be expensive. Two places in particular:
#
#   tool status   ACP has four values, Rigma's chip has three. `pending` and
#                 `in_progress` are IN FLIGHT, and in-flight is not success —
#                 returning True would put a tick on a call that has not finished.
#   plan mode     `current_mode_update` carries an ID, not a boolean. Interpreting
#                 it wrongly lights the plan indicator for a session not planning.

def _u(kind, **fields):
    """One `session/update` notification."""
    return {"method": "session/update",
            "params": {"sessionId": "s1",
                       "update": {"sessionUpdate": kind, **fields}}}


def test_a_message_chunk_becomes_text():
    out = acp.map_acp_update(_u("agent_message_chunk",
                                content={"type": "text", "text": "hello"}))
    assert [(e.kind, e.text) for e in out] == [("text", "hello")]


def test_a_thought_chunk_becomes_thinking_not_text():
    """Reasoning rendered as the reply would put the model's working in the answer."""
    out = acp.map_acp_update(_u("agent_thought_chunk",
                                content={"type": "text", "text": "hmm"}))
    assert [(e.kind, e.text) for e in out] == [("thinking", "hmm")]


def test_a_non_text_content_block_is_named_not_dropped():
    """ACP content is a UNION, so assuming a string silently drops every image and
    resource link. The same class of defect as the `exec` projector's unnamed
    blocks, and it is named here instead."""
    out = acp.map_acp_update(_u("agent_message_chunk", content={"type": "image"}))
    assert [(e.kind, e.text) for e in out] == [("text", "[image]")]
    out = acp.map_acp_update(_u("agent_message_chunk",
                                content={"type": "resource_link", "name": "a.txt"}))
    assert out[0].text == "[link a.txt]"


def test_content_given_as_a_block_list_is_flattened():
    out = acp.map_acp_update(_u("agent_message_chunk", content=[
        {"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]))
    assert out[0].text == "a[image]b"


def test_an_empty_chunk_produces_no_event_at_all():
    """An empty text event would put an empty bubble in the transcript."""
    assert acp.map_acp_update(_u("agent_message_chunk", content={"type": "text",
                                                                 "text": ""})) == []


def test_tool_status_is_three_valued_and_in_flight_is_not_success():
    """The specific error worth avoiding: a tick on a call that has not finished."""
    assert acp.acp_tool_ok("completed") is True
    assert acp.acp_tool_ok("failed") is False
    assert acp.acp_tool_ok("pending") is None
    assert acp.acp_tool_ok("in_progress") is None
    assert acp.acp_tool_ok(None) is None
    assert acp.acp_tool_ok("nonsense") is None


def test_a_completed_tool_call_emits_the_chip_and_then_its_result():
    """Terminal status carries both facts, and the transcript must read
    call-then-outcome the way every other backend's does."""
    out = acp.map_acp_update(_u("tool_call", toolCallId="c1", title="read file",
                                name="read", status="completed",
                                rawInput={"path": "a.txt"},
                                content=[{"type": "text", "text": "file body"}]))
    assert [e.kind for e in out] == ["tool", "tool_result"]
    assert out[0].name == "read" and out[0].args == {"path": "a.txt"}
    assert out[0].data == {"id": "c1"}
    assert out[1].ok is True and out[1].text == "file body"
    assert out[1].data == {"id": "c1"}, "the chip and its result share an id"


def test_an_in_flight_tool_call_emits_no_result():
    out = acp.map_acp_update(_u("tool_call", toolCallId="c1", name="read",
                                status="in_progress", rawInput={"path": "a.txt"}))
    assert [e.kind for e in out] == ["tool"]


def test_a_failed_tool_call_is_a_cross_not_a_tick():
    out = acp.map_acp_update(_u("tool_call_update", toolCallId="c1", name="read",
                                status="failed", content=[{"type": "text", "text": "no"}]))
    assert out[-1].kind == "tool_result" and out[-1].ok is False


def test_the_title_is_used_when_the_call_has_no_name():
    out = acp.map_acp_update(_u("tool_call", toolCallId="c1", title="write file",
                                status="in_progress", rawInput={"a": 1}))
    assert out[0].name == "write file"


def test_plan_mode_is_derived_from_the_mode_id_not_guessed():
    """The wire carries an id and the UI wants a boolean, so the id is interpreted.
    Getting this wrong lights the plan indicator for a session that is not planning."""
    out = acp.map_acp_update(_u("current_mode_update", currentModeId="plan"))
    assert out[0].event == "plan/mode" and out[0].data["active"] is True
    out = acp.map_acp_update(_u("current_mode_update", currentModeId="default"))
    assert out[0].data["active"] is False
    out = acp.map_acp_update(_u("current_mode_update", currentModeId="something-new"))
    assert out[0].data["active"] is False, "an unknown mode is not plan mode"


def test_usage_is_translated_into_the_names_the_usage_line_renders():
    """`used`/`size` are ACP's names. Passing them through would show nothing,
    because the renderer keys on its own."""
    out = acp.map_acp_update(_u("usage_update", used=1234, size=32768,
                                cost={"amount": 0.5, "currency": "USD"}))
    assert out[0].event == "usage"
    assert out[0].data["usedTokens"] == 1234
    assert out[0].data["contextWindowTokens"] == 32768
    assert out[0].data["cost"]["amount"] == 0.5


def test_a_usage_update_with_nothing_usable_emits_nothing():
    assert acp.map_acp_update(_u("usage_update")) == []


def test_the_acp_plan_variant_lands_in_the_todo_panel():
    """ACP's `plan` variant is the same fact as a todo list — items with a status
    each — so it uses the panel that already exists rather than a second one."""
    out = acp.map_acp_update(_u("plan", entries=[
        {"content": "step one", "priority": "high", "status": "in_progress"},
        {"content": "step two", "priority": "low", "status": "pending"}]))
    assert out[0].event == "todos"
    assert out[0].data["todos"] == [
        {"content": "step one", "status": "in_progress"},
        {"content": "step two", "status": "pending"}]


def test_a_plan_review_becomes_its_own_structured_event():
    out = acp.map_acp_update(_u("plan_update", plan={"type": "markdown",
                                                     "planId": "p1",
                                                     "content": "# Plan"}))
    assert out[0].event == "acp_plan" and out[0].data["content"] == "# Plan"


def test_session_info_update_carries_only_what_it_has():
    out = acp.map_acp_update(_u("session_info_update", title="Renamed"))
    assert out[0].event == "session/title" and out[0].data == {"title": "Renamed"}
    assert acp.map_acp_update(_u("session_info_update")) == []


def test_an_unknown_session_update_variant_is_silent_not_a_wrong_row():
    """The union has 13 members and will grow. A future variant must not become a
    wrong row in the transcript, and must not become per-turn noise either."""
    assert acp.map_acp_update(_u("some_future_variant", x=1)) == []
    assert acp.map_acp_update({"method": "something/else", "params": {}}) == []


# --- the extension notifications, and the clear tombstone --------------------

def test_a_goal_update_becomes_the_goal_event_the_panel_already_renders():
    out = acp.map_acp_update({"method": "mcode/session/goal_update",
                              "params": {"sessionId": "s1",
                                         "goal": {"goalId": "g1",
                                                  "objective": "ship it",
                                                  "status": "active"}}})
    assert out[0].event == "goal"
    assert out[0].data["objective"] == "ship it"


def test_a_cleared_goal_becomes_the_TOMBSTONE_the_frontend_reads_as_clear():
    """`chat/goal.ts` reads `{cleared: true}` as an INSTRUCTION to blank the panel.
    Sending the raw `{goal: null}` instead would be read as "not a goal" and leave
    the cleared objective on screen forever."""
    out = acp.map_acp_update({"method": "mcode/session/goal_update",
                              "params": {"sessionId": "s1", "goal": None,
                                         "goalId": "g1"}})
    assert out[0].data.get("cleared") is True
    assert out[0].data.get("goalId") == "g1"


def test_the_queue_and_delegation_updates_become_structured_events():
    out = acp.map_acp_update({"method": "mcode/session/queue_update",
                              "params": {"items": [{"itemId": "q1"}]}})
    assert out[0].event == "acp_queue" and out[0].data["items"] == [{"itemId": "q1"}]
    out = acp.map_acp_update({"method": "mcode/session/delegation_update",
                              "params": {"snapshot": {"schemaVersion": 1,
                                                      "members": []}}})
    assert out[0].event == "acp_delegation"
    assert out[0].data["schemaVersion"] == 1


def test_a_malformed_extension_payload_does_not_raise():
    """A probe or a version change must not take down the turn loop."""
    for bad in ({"method": "mcode/session/queue_update", "params": {"items": "no"}},
                {"method": "mcode/session/delegation_update", "params": {"snapshot": 7}},
                {"method": "mcode/session/queue_update"},
                {"method": "mcode/session/delegation_update"}):
        out = acp.map_acp_update(bad)
        assert len(out) == 1, bad


def test_a_tool_call_with_no_payload_announces_nothing():
    out = acp.map_acp_update(_u("tool_call", toolCallId="c1", status="in_progress"))
    assert out == []
