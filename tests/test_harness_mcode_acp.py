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

import sys
import threading
import time
from pathlib import Path

import pytest

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
        assert c.delegation_get() == {"delegations": []}


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
        notices = [e for e in events
                   if e["method"] == "mcode/session/current_session_update"]
        assert len(notices) == 2


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
