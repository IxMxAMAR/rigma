"""R6-ACP-TURN: driving a chat turn over the Agent Client Protocol.

WHY THIS FILE EXISTS. Everything in `harness_mcode_acp` up to now was transport: a
client that can talk to `mcode acp` and a mapper that turns its notifications into
Rigma's vocabulary. A control plane nobody can reach DURING a turn is still
unreachable, so the features that motivated the client — answering a permission
prompt, plan mode, steering, the queue — only become live when a chat turn runs
here instead of through `mcode exec`.

These tests drive `drive_turn_acp` against `tests/fake_acp_server.py`, which is a
real subprocess speaking real JSON-RPC over real pipes. No model is loaded: the fake
answers the handshake and the prompt itself, so the standing order is respected and
the transport is still genuinely exercised.

WHAT IS *NOT* TESTED HERE, and must not be implied by a green run: that mcode's own
`session/prompt` behaves this way against a live engine. That calls a model.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from rigma import harness_mcode_acp as acp

FAKE = str(pathlib.Path(__file__).resolve().parent / "fake_acp_server.py")


@pytest.fixture
def fake_acp(monkeypatch):
    """Point `drive_turn_acp` at the fake instead of the real mcode.

    The client's argv is rewritten rather than `bin_path`, because `drive_turn_acp`
    builds `[exe, "acp"]` and the fake is a script, not an executable named `acp`.
    """
    holder = {"argv_extra": []}
    real_client = acp.AcpClient

    class _Redirect(real_client):
        def __init__(self, argv, **kwargs):
            super().__init__([sys.executable, FAKE, *holder["argv_extra"]], **kwargs)

    monkeypatch.setattr(acp, "AcpClient", _Redirect)
    return holder


def _drive(prompt="hello", *, permission="full", state=None, on_permission=None,
           on_question=None, timeout=30.0, **kw):
    """Run one turn over the fake and return every event it produced."""
    return list(acp.drive_turn_acp(
        prompt, exe=sys.executable, state=state, permission=permission,
        on_permission=on_permission, on_question=on_question, timeout=timeout,
        **kw,
    ))


# --- the turn runs at all -----------------------------------------------------


def test_a_turn_produces_text(fake_acp):
    events = _drive("say hi")
    text = "".join(e.text or "" for e in events if e.kind == "text")
    assert text, f"a turn must produce text; got {[e.kind for e in events]}"


def _map(update):
    """Run one `session/update` through the mapper and return the events it produced.

    `map_acp_update` takes the WHOLE notification frame and reads `params.update` itself.
    """
    return acp.map_acp_update(
        {"method": "session/update", "params": {"update": update}})


def test_all_three_plan_arms_are_handled():
    """`PlanUpdateContent` is a THREE-way union and only two arms were drawn.

    READ from `@agentclientprotocol/sdk@1.4.0`:

        PlanUpdateContent = (PlanItems  & {type:"items"})
                          | (PlanFile   & {type:"file"})
                          | (PlanMarkdown & {type:"markdown"})

    `PlanItems` carries `entries` and NEITHER `content` NOR `uri`, so the plan panel could
    not draw it — and its entries were dropped rather than routed anywhere. mcode today
    sends only `markdown`, which is why this looked fine: the defect is one conformant
    server away, and the SDK is the authority this module already cites.
    """
    # markdown -> the plan panel, which draws `content`.
    md = _map({"sessionUpdate": "plan_update",
               "plan": {"type": "markdown", "planId": "p1", "content": "# Plan"}})
    assert [e.event for e in md] == ["acp_plan"], [e.event for e in md]
    assert md[0].data["content"] == "# Plan"

    # file -> the plan panel too, which draws `uri`.
    fl = _map({"sessionUpdate": "plan_update",
               "plan": {"type": "file", "planId": "p1", "uri": "file:///p.md"}})
    assert [e.event for e in fl] == ["acp_plan"], [e.event for e in fl]

    # items -> the TODOS channel, the same place the SDK's sibling `plan` variant sends the
    # same data. This is the arm that used to vanish.
    it = _map({"sessionUpdate": "plan_update",
               "plan": {"type": "items", "planId": "p1", "entries": [
                   {"content": "first", "status": "completed"},
                   {"content": "second", "status": "in_progress"}]}})
    assert [e.event for e in it] == ["todos"], [e.event for e in it]
    assert [t["content"] for t in it[0].data["todos"]] == ["first", "second"]
    assert [t["status"] for t in it[0].data["todos"]] == ["completed", "in_progress"]


def test_a_plan_update_with_no_plan_says_nothing():
    """`or {}` used to make this emit `acpPlan = {}`, which is NON-NULL.

    The outer panel gated on non-null and the inner one on a string `content`/`uri`, so the
    container was drawn around a child that rendered nothing: a visible empty box. An update
    carrying no plan has nothing to say, and saying nothing is the correct answer.
    """
    for empty in ({"sessionUpdate": "plan_update"},
                  {"sessionUpdate": "plan_update", "plan": None},
                  {"sessionUpdate": "plan_update", "plan": {}}):
        events = _map(empty)
        assert events == [], f"{empty} produced {[e.event for e in events]}"


def test_a_real_delegation_tree_reaches_the_panel(fake_acp):
    """The delegation path had NO end-to-end coverage: the fake always sent an empty list.

    `Se(e)` in mcode's `chunks/chunk-M5QJG5VT.js` is the member mapper, so this drives the
    fake's seeded tree through a real turn and checks that every field the panel reads
    survives the trip — including `errorMessage`, which is emitted only when present and
    which nothing in `tests/` had ever produced.
    """
    fake_acp["argv_extra"] = ["--delegation", "tree"]
    state: dict = {}
    _drive("hello", state=state)
    delegations = [e for e in _drive("again", state=state)
                   if e.kind == "state" and e.event == "acp_delegation"]
    assert delegations, "the turn must surface the delegation snapshot"
    snap = delegations[0].data
    assert snap.get("schemaVersion") == 1, snap
    assert snap.get("rootSessionId"), "the snapshot names its root"
    members = {m["sessionId"]: m for m in snap["members"]}
    assert set(members) == {"mvs_child", "mvs_grandchild", "mvs_bg"}, sorted(members)

    # THE TREE. `parentSessionId` is what makes this a tree rather than a flat list, and it
    # is emitted unconditionally by mcode.
    assert members["mvs_child"]["parentSessionId"] == snap["rootSessionId"]
    assert members["mvs_grandchild"]["parentSessionId"] == "mvs_child"

    # `task` holds the session's TITLE, not an id.
    assert members["mvs_child"]["task"] == "map the call sites"
    # The status vocabulary, normalised by mcode.
    assert members["mvs_child"]["status"] == "running"
    assert members["mvs_bg"]["status"] == "failed"
    # The two conditional fields. Both are REAL and neither had a producer in this repo.
    assert members["mvs_bg"]["errorMessage"] == "the worker exited with code 3"
    assert members["mvs_bg"]["backgroundTaskId"] == "bg-7"
    # And they are absent, not empty, when the server did not send them.
    assert "errorMessage" not in members["mvs_child"]
    assert "backgroundTaskId" not in members["mvs_child"]


def test_the_stop_receipt_names_what_it_stopped_and_what_it_could_not(fake_acp):
    """The receipt shape is MEASURED, and the double used to invent a different one.

    mcode's `stop()` returns `{schemaVersion, rootSessionId, rootStopped, stoppedSessionIds,
    activeSessionIds, failedSessionIds}` — there is no `stopped`. The fake answered
    `{"receipt": {"stopped": [...]}}` and the client read that same invented field, so the
    two agreed with each other about a field the server does not have, and the UI reported
    "all delegated work stopped" for every outcome.
    """
    fake_acp["argv_extra"] = ["--delegation", "tree"]
    state: dict = {}
    _drive("hello", state=state)
    sid = state.get("session_id")
    assert sid, "the first turn must establish a session"

    from rigma import harness_mcode_acp as acp_mod
    res = acp_mod.drive_control("delegation_stop", None, exe=sys.executable,
                                session_id=sid, timeout=30.0)
    receipt = (res or {}).get("result", {}).get("receipt")
    assert receipt, res
    # The names mcode actually uses.
    assert "stopped" not in receipt, f"the invented field is back: {receipt}"
    assert set(receipt["stoppedSessionIds"]) == {"mvs_child", "mvs_grandchild"}
    # The member that had already FAILED is reported as unstoppable rather than counted as
    # stopped — which is what makes the UI's partial-failure sentence possible.
    assert receipt["failedSessionIds"] == ["mvs_bg"]
    assert receipt["rootStopped"] is True


def test_the_handshakes_own_option_list_reaches_the_ui(fake_acp):
    """The settings block is gated on this list, so it has to arrive on a turn.

    The handshake already ANSWERS with `configOptions`; the only thing that could put them
    on the wire as an `acp_config` event was a `config_option_update` notification. So on a
    server that does not push one — and the fake does not — the panel's session-settings
    controls rendered NOTHING, gated on a list that never arrived.
    """
    events = _drive("hello")
    # `event` is a TOP-LEVEL field on TurnEvent — "the backend's own event name" — and the
    # payload is `data` beside it.
    config = [e for e in events if e.kind == "state" and e.event == "acp_config"]
    assert config, ("the handshake's configOptions must be forwarded as an acp_config "
                    f"event; got {[e.event for e in events if e.kind == 'state']}")
    opts = config[0].data.get("configOptions")
    assert isinstance(opts, list) and opts, f"the option list was {opts!r}"
    ids = {str(o.get("id")) for o in opts if isinstance(o, dict)}
    # The two the fake's handshake declares. A list that arrives but names nothing the
    # panel can draw is the same defect one step later.
    assert {"permissionMode", "model"} <= ids, f"the list named {sorted(ids)}"


def test_the_option_list_is_sent_even_when_it_is_empty(fake_acp, monkeypatch):
    """An empty list has to be SAID, not omitted.

    If the event were skipped when there is nothing to report, the panel would keep
    drawing the PREVIOUS turn's options after a session that has none — stale settings
    presented as current, which is worse than no settings at all.
    """
    real_resume = acp.AcpClient.session_resume

    def _no_options(self, *a, **kw):
        res = real_resume(self, *a, **kw)
        self.config_options = []
        return res

    monkeypatch.setattr(acp.AcpClient, "session_resume", _no_options)
    state = {"session_id": "mvs_fake_session"}
    events = _drive("hello", state=state)
    config = [e for e in events if e.kind == "state" and e.event == "acp_config"]
    assert config, "an empty list must still be reported"
    assert config[0].data.get("configOptions") == []


def test_the_session_id_is_written_back_for_the_next_turn(fake_acp):
    """Continuity is the whole reason to hand a turn to an agent rather than a
    prompt, so a turn that forgets its session id is a turn that starts over."""
    state: dict = {}
    _drive("first", state=state)
    assert state.get("session_id"), "the session id must reach the caller's state"


def test_a_second_turn_continues_the_first_session(fake_acp):
    """The ACP equivalent of `exec --session <id>`."""
    state: dict = {}
    _drive("first", state=state)
    first = state["session_id"]
    events = _drive("second", state=state)
    assert state["session_id"] == first
    assert not [e for e in events if e.kind == "notice" and "new one" in (e.text or "")]


def test_a_renamed_session_is_recorded_as_the_server_named_it(fake_acp):
    """The server may answer with a DIFFERENT id than the one asked for.

    The same rule `exec`'s `--model` follows: use the id mcode actually gave us, not
    the one we asked for. A client that trusts its own id continues the wrong
    session — or none.
    """
    fake_acp["argv_extra"] = ["--resume-renames"]
    state = {"session_id": "mvs_older_session"}
    _drive("again", state=state)
    assert state["session_id"] == "mvs_renamed_session", (
        "the id the SERVER named must win, not the one requested")


def test_an_unknown_session_starts_a_new_one_and_says_so(fake_acp):
    """A session mcode no longer has is not a failed turn.

    But continuing a conversation whose earlier half is gone, silently, would make
    the transcript describe a turn that did not happen — so it is reported.
    """
    fake_acp["argv_extra"] = ["--resume-unknown"]
    state = {"session_id": "mvs_never_existed"}
    events = _drive("again", state=state)
    notices = [e.text or "" for e in events if e.kind == "notice"]
    assert any("new one" in n for n in notices), notices
    assert not [e for e in events if e.kind == "error"], "this is not an error"
    assert state.get("session_id"), "a fresh session id must still be recorded"


# --- the permission policy, which is the reason for the whole module ----------


def test_permission_modes_translate_because_the_vocabularies_share_no_member():
    """`exec` says smart|full|off; ACP says default|auto|bypassPermissions.

    Passing `full` straight through is REJECTED by the real server with
    `Unsupported permission mode: full`, so this is a translation, not a rename.
    """
    assert acp.acp_permission_mode("full") == "bypassPermissions"
    assert acp.acp_permission_mode("smart") == "auto"
    assert acp.acp_permission_mode("off") == "default"


def test_an_unrecognised_mode_does_not_become_the_most_permissive_one():
    """A fallback that landed on `bypassPermissions` would turn a typo into a grant."""
    assert acp.acp_permission_mode("nonsense") == "auto"
    assert acp.acp_permission_mode("") == "auto"


def _decided(events):
    """The DECISION events, in the vocabulary the governance trail already reads.

    `drive_turn_acp` reports an ACP permission decision as `approval/asked` +
    `approval/decided` rather than a private event name, because `chat/governance.ts`
    already folds that pair into a panel with a UI — so answering a permission prompt
    needed no new UI at all.
    """
    return [e for e in events if e.event == "approval/decided"]


def _asked(events):
    return [e for e in events if e.event == "approval/asked"]


def test_a_permission_request_is_ANSWERED_so_the_server_never_waits(fake_acp):
    """The `exec` dead-end this exists to remove.

    `mcode exec` has no interaction host, so when `smart` asks, nobody can answer
    and mcode's guard blocks the chat permanently. Here the server asks and waits,
    so an unanswered request would hang the turn — and every branch must reply.
    """
    fake_acp["argv_extra"] = ["--ask-permission"]
    seen: list = []

    def _answer(method, params):
        seen.append((method, params))
        return True

    events = _drive("do it", permission="auto", on_permission=_answer)
    assert seen, "the server asked and the policy must have been consulted"
    assert seen[0][0] == "session/request_permission"
    # And the decision is REPORTED rather than silently taken.
    assert _asked(events), "the question must reach the governance trail"
    got = _decided(events)
    assert got, "a permission decision must be visible in the transcript"
    assert got[0].data["outcome"] == "allowed-once", got[0].data


def test_a_declined_permission_is_declined_not_ignored(fake_acp):
    """`outcome: "selected"` means "the server offered options and one was chosen".

    It is NOT a synonym for granted. The DECISION is the optionId, so that is what a
    refusal has to be asserted on — asserting on `outcome` would have passed a deny
    as a grant, which is the worst possible direction to get this wrong.
    """
    fake_acp["argv_extra"] = ["--ask-permission"]
    events = _drive("do it", permission="auto", on_permission=lambda m, p: False)
    got = _decided(events)
    assert got, "the request was answered"
    assert got[0].data["outcome"] == "rejected", got[0].data


def test_with_no_policy_at_all_the_request_is_still_answered(fake_acp):
    """THE IMPORTANT ONE. No policy must mean "no", and must still be an ANSWER.

    An unanswered request is worse than a decline: mcode waits forever and the chat
    is blocked. So this asserts a reply was produced, and that `auto` mode does not
    silently grant.
    """
    fake_acp["argv_extra"] = ["--ask-permission"]
    events = _drive("do it", permission="smart", on_permission=None)
    got = _decided(events)
    assert got, "the request was answered even with no policy"
    assert got[0].data["outcome"] == "rejected", (
        f"with no policy, `auto` must NOT grant; got {got[0].data}")
    asked = _asked(events)
    assert asked and asked[0].data["auto"] is True, (
        "the trail must say the decision was automatic")


def test_bypass_permissions_grants_and_says_so(fake_acp):
    """`full` maps to `bypassPermissions`, which grants — and a grant that is not
    disclosed reads as a turn that simply did nothing unusual."""
    fake_acp["argv_extra"] = ["--ask-permission"]
    events = _drive("do it", permission="full", on_permission=None)
    got = _decided(events)
    assert got and got[0].data["outcome"] == "allowed-once", (
        got[0].data if got else "no decision")
    notices = [e.text or "" for e in events if e.kind == "notice"]
    assert any("bypassed" in n for n in notices), notices


# --- elicitation -------------------------------------------------------------


def test_a_question_with_no_answer_continues_instead_of_hanging(fake_acp):
    """Declining is what makes mcode take its non-interactive fallback — the place
    it already lands today — so this changes nothing except that the server is TOLD
    rather than left waiting."""
    fake_acp["argv_extra"] = ["--ask-question"]
    events = _drive("ask me", permission="auto", on_question=None)
    notices = [e.text or "" for e in events if e.kind == "notice"]
    assert any("question" in n for n in notices), notices


def test_a_question_with_an_answer_is_returned(fake_acp):
    fake_acp["argv_extra"] = ["--ask-question"]
    asked: list = []

    def _answer(method, params):
        asked.append(method)
        return {"answer": "yes"}

    _drive("ask me", permission="auto", on_question=_answer)
    assert asked == ["elicitation/create"], asked


# --- what the turn reports ----------------------------------------------------


def test_a_prompt_error_is_an_error_not_a_silent_empty_turn(fake_acp):
    fake_acp["argv_extra"] = ["--prompt-error", "engine is down"]
    events = _drive("go")
    errors = [e.text or "" for e in events if e.kind == "error"]
    assert any("engine is down" in e for e in errors), errors


def test_a_missing_binary_is_reported_rather_than_crashing(monkeypatch):
    """A FALSE STATEMENT ABOUT THE MACHINE is the failure mode being avoided: an
    empty exe must not become a traceback, and must not become a claim that mcode
    is absent when the resolver was simply wrong."""
    monkeypatch.setattr("rigma.harness_mcode.bin_path", lambda: None)
    events = list(acp.drive_turn_acp("hi", exe="", timeout=5.0))
    assert [e.kind for e in events] == ["error"]


# --- the translation is real, and it is necessary -----------------------------


def test_the_translated_permission_mode_REACHES_the_server(fake_acp, tmp_path):
    """The translation is end-to-end, not just a table in this module.

    The fake validates `permissionMode` with the runtime's OWN guard and refuses
    anything outside default|auto|bypassPermissions, recording every refusal. A turn
    that sent `full` unchanged would leave a refusal behind.
    """
    record = tmp_path / "rejected.json"
    fake_acp["argv_extra"] = ["--record", str(record)]
    _drive("go", permission="full")
    assert record.exists(), "the fake did not run its recorder"
    assert record.read_text(encoding="utf-8") == "[]", (
        "`full` must be TRANSLATED; the server refused it")


def test_sending_the_raw_exec_mode_WOULD_be_rejected(fake_acp, monkeypatch, tmp_path):
    """AND the translation is NECESSARY, which the test above cannot show alone.

    If `acp_permission_mode` were a no-op the turn would STILL succeed, because the
    driver deliberately does not fail a turn over a refused config option — so
    "no error" is not evidence the value was accepted. This removes the translation
    and reads the refusal the fake recorded, which is the only way to tell "the
    server refused `full`" from "the driver never asked".
    """
    record = tmp_path / "rejected.json"
    fake_acp["argv_extra"] = ["--record", str(record)]
    monkeypatch.setattr(acp, "acp_permission_mode", lambda _p: "full")
    _drive("go", permission="full")
    assert record.exists(), "the fake recorded nothing — the assertion is vacuous"
    refused = record.read_text(encoding="utf-8")
    assert "full" in refused, (
        f"a pass-through mode must be REFUSED by the server; recorded {refused!r}")


def test_the_outcome_uses_the_vocabulary_the_governance_panel_reads():
    """A THIRD vocabulary is the failure mode here.

    `outcomeTone`/`outcomeLabel` in `chat/governance.ts` know exactly four words:
    allowed-once, rejected, cancelled, unavailable. Anything else is coloured grey
    and shown as a bare identifier, so a granted permission would look like an
    unexplained neutral event rather than a grant.

    The rule is copied from DSH's own bridge
    (`packages/acp/acp/src/index.ts:171`), so both transports produce the same
    words: cancelled if the server said so, allowed-once only for `allow-once`,
    rejected otherwise.
    """
    f = acp._acp_approval_outcome
    assert f({"outcome": "selected"}, "allow-once") == "allowed-once"
    assert f({"outcome": "selected"}, "deny") == "rejected"
    assert f({"outcome": "cancelled"}, "") == "cancelled"
    # Only `allow-once` is a grant. `allow-always` is NOT, matching DSH — a durable
    # grant is not something either transport infers from a one-shot answer.
    assert f({"outcome": "selected"}, "allow-always") == "rejected"
    # And every value produced is one the panel knows.
    for outcome, opt in (({"outcome": "selected"}, "allow-once"),
                         ({"outcome": "selected"}, "deny"),
                         ({"outcome": "cancelled"}, ""),
                         ({"outcome": "selected"}, "reject-once")):
        assert f(outcome, opt) in ("allowed-once", "rejected", "cancelled",
                                   "unavailable")
