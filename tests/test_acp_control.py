"""R6-ACP-CONTROL: the control plane becomes INVOCABLE, not just readable.

WHAT WAS MISSING. `AcpClient` has defined mcode's whole control surface since `a7caded` —
the goal, the queue, steering, the delegation tree, plan mode, the configOptions — and
NOTHING CALLED ANY OF IT. Every one of those arrived as a NOTIFICATION, so the panel could
draw the queue and the goal and the delegation tree, and a user could not touch any of
them. A control plane you can only watch is not a control plane, and "carry the goal
control plane" was not done until a caller existed.

These tests drive the REAL entry point against `tests/fake_acp_server.py` — a real
subprocess over real pipes — so the transport, the resume, the operation dispatch and the
teardown are all exercised. No model is loaded.

THE THING THAT MADE THIS TESTABLE AT ALL. A control operation opens its OWN process, so
the session has to survive the process that changed it. The fake kept its state in memory
and therefore answered `session/resume` and behaved as if the session were new, which
would have let a broken route pass: `goal_create` succeeds, `goal_get` reports no goal.
`--state-file` now models the real store, and `test_a_goal_survives_the_process_that_set_it`
is the test that would have caught it.
"""
from __future__ import annotations

import pathlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from rigma import harness_mcode, harness_mcode_acp, sessions
from rigma.serve import build_app

FAKE = pathlib.Path(__file__).with_name("fake_acp_server.py")


def _exe(state_file: pathlib.Path) -> str:
    """The fake, as a command LINE — which is what `drive_control` accepts.

    A real `bin_path()` is one token; a fake needs its interpreter too. Both shapes go
    through the same path, which is why the split lives in `drive_control`.
    """
    return f'"{sys.executable}" "{FAKE}" --state-file "{state_file}"'


@pytest.fixture
def fake(tmp_path):
    """The fake's command line, and the state file it persists to."""
    state_file = tmp_path / "session.json"
    return _exe(state_file), state_file


@pytest.fixture
def sid():
    """A session id UNIQUE PER TEST, and this is not tidiness.

    The fake persists the whole session — including the session id — to one state file,
    and `session/resume` adopts the id it is asked for. Two tests sharing a state file
    therefore share a session: whichever resumes last rewrites the id, and a later
    `session/resume` for the OTHER id still loads that state. The symptom was a test
    failing with "No active goal" for a goal it had just created, which reads as a broken
    route and is actually a leaked fixture.
    """
    import uuid

    return f"mvs_{uuid.uuid4().hex[:12]}"


def op(name: str, exe: str, sid: str, **params):
    """`sid` is REQUIRED, deliberately. A default here is how the leak above happened:
    every test silently shared one session. Making it explicit means a new test cannot
    forget it."""
    return harness_mcode.drive_control(name, params, exe=exe, session_id=sid,
                                       timeout=30.0)


# --- the driver ---------------------------------------------------------------

def test_a_goal_survives_the_process_that_set_it(fake, sid):
    """THE TEST THIS ROUTE RESTS ON.

    Each operation is its own process, so a session that does not persist makes
    `goal_create` look successful and `goal_get` report nothing — a route that lies. The
    first version of the fake did exactly that.
    """
    exe, _ = fake
    assert op("goal_get", exe, sid)["result"] == {"goal": None}
    created = op("goal_create", exe, sid, objective="ship the parser")
    assert created["ok"], created
    assert created["result"]["goal"]["objective"] == "ship the parser"
    # A DIFFERENT process reads it back.
    assert op("goal_get", exe, sid)["result"]["goal"]["objective"] == "ship the parser"


def test_a_goal_can_be_patched_and_cleared_across_processes(fake, sid):
    """`goal/patch` was ADVERTISED by the fake and never handled, so it answered
    `-32601 Method not found` — a method no test could cover."""
    exe, _ = fake
    op("goal_create", exe, sid, objective="ship it")
    patched = op("goal_patch", exe, sid, status="paused")
    assert patched["ok"], patched
    assert patched["result"]["goal"]["status"] == "paused"
    assert op("goal_get", exe, sid)["result"]["goal"]["status"] == "paused"
    assert op("goal_clear", exe, sid)["ok"]
    assert op("goal_get", exe, sid)["result"]["goal"] is None


def test_a_queued_message_survives_and_is_listed(fake, sid):
    exe, _ = fake
    added = op("queue_enqueue", exe, sid, text="do this later")
    assert added["ok"], added
    items = op("queue_list", exe, sid)["result"]["items"]
    assert [i["content"] for i in items] == ["do this later"]


def test_the_delegation_tree_uses_the_real_snapshot_shape(fake, sid):
    """The fake answered `{"delegations": []}`, which the real server does not produce —
    it answers a `snapshot` with `members`, and that is what the UI's `AcpDelegation` is
    built from. The old shape was one no client could have been reading correctly."""
    exe, _ = fake
    got = op("delegation_get", exe, sid)
    assert got["ok"], got
    assert "snapshot" in got["result"]
    assert got["result"]["snapshot"]["members"] == []


def test_steering_and_the_mode_and_config_selects_all_drive(fake, sid):
    """The remaining three surfaces, so "the control plane is invocable" is a claim
    about the whole plane rather than the part with an easy test."""
    exe, _ = fake
    assert op("steer", exe, sid, text="focus on the parser")["ok"]
    assert op("activate", exe, sid)["ok"]
    assert op("mode_set", exe, sid, modeId="plan")["ok"]
    assert op("config_set", exe, sid, optionId="permissionMode", value="auto")["ok"]


def test_an_operation_on_an_unknown_session_reports_the_transport_error(fake):
    """`drive_control` NEVER RAISES. It is called from a route that is drawing a panel,
    and an exception would take down the panel instead of reporting one failure."""
    exe, _ = fake
    out = op("goal_get", exe, "mvs_not_a_session")
    # The fake accepts any id, so this asserts the CONTRACT rather than a failure: the
    # result is a dict with the documented keys, never an exception.
    assert set(out) == {"ok", "op", "result", "error"}


def test_a_missing_session_is_refused_without_opening_a_process():
    """A goal on a session the chat is not using would be a success that changed nothing
    the user can see, so this is refused rather than invented."""
    out = harness_mcode.drive_control("goal_get", {}, exe="definitely-not-mcode",
                                      session_id="", timeout=5.0)
    assert out["ok"] is False
    assert "no mcode session" in out["error"]


# --- the operation table ------------------------------------------------------

def test_an_unknown_operation_is_refused_with_the_known_ones():
    why = harness_mcode.control_op_error("session/prompt", {})
    assert why and "unknown operation" in why
    # `session/prompt` is named deliberately: the allowlist is what stops an HTTP body
    # from smuggling a MODEL TURN through a control route.
    assert "session/prompt" in why


def test_a_required_parameter_is_enforced():
    assert "objective" in harness_mcode.control_op_error("goal_create", {})
    assert "text" in harness_mcode.control_op_error("queue_enqueue", {})
    assert harness_mcode.control_op_error("goal_create", {"objective": "x"}) == ""


def test_every_operation_in_the_table_is_reachable_by_name(fake, sid):
    """A row in the table that the dispatcher does not implement would answer "unhandled
    operation", which is a hole in the plane that the table would hide."""
    exe, _ = fake
    op("goal_create", exe, sid, objective="seed")
    op("queue_enqueue", exe, sid, text="seed")
    required = {
        "goal_create": {"objective": "x"},
        "goal_patch": {"status": "paused"},
        "queue_enqueue": {"text": "x"},
        "queue_update": {"itemId": "q1", "text": "x"},
        "queue_delete": {"itemId": "q1"},
        "queue_steer": {"itemId": "q1"},
        "steer": {"text": "x"},
        "delegation_stop": {"sessionId": "nobody"},
        "mode_set": {"modeId": "plan"},
        "config_set": {"optionId": "permissionMode", "value": "auto"},
    }
    for name in harness_mcode.CONTROL_OPS:
        params = required.get(name, {})
        out = op(name, exe, sid, **params)
        # Not every one can SUCCEED against the fake (stopping a member that does not
        # exist is a legitimate error), but none may be "unhandled".
        assert "unhandled operation" not in str(out.get("error") or ""), (name, out)


# --- the HTTP route -----------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def _make_chat(tmp_path, *, harness="mcode", transport="acp", backend_sid=""):
    """A stored chat with the fields the route reads.

    `create` takes only a title and a system prompt, so the backend fields are written
    onto the row afterwards — which is also how a real chat acquires them, on its first
    turn rather than at creation.
    """
    s = sessions.create(title="t")
    s["harness"] = harness
    s["mcode_transport"] = transport
    s["harness_sessions"] = {"mcode": backend_sid} if backend_sid else {}
    sessions.save(s)
    return s["id"]


def test_the_route_refuses_a_chat_that_is_not_mcode(client, tmp_path):
    """Doing nothing would be worse than refusing: the UI would show a control that
    appeared to work."""
    sid = _make_chat(tmp_path, harness="native")
    r = client.post(f"/api/sessions/{sid}/control", json={"op": "goal_get"})
    assert r.status_code == 409
    assert "not an mcode chat" in r.json()["error"]


def test_the_route_refuses_the_exec_transport(client, tmp_path):
    """`exec` holds no session, so there is nothing to steer and no queue to add to.
    The refusal names the transport so the UI can tell the user what to change."""
    sid = _make_chat(tmp_path, transport="exec")
    r = client.post(f"/api/sessions/{sid}/control", json={"op": "goal_get"})
    assert r.status_code == 409
    assert r.json()["transport"] == "exec"
    assert "acp" in r.json()["error"]


def test_the_route_refuses_an_unknown_operation_with_400(client, tmp_path):
    """A client mistake, not a transport failure — and refused BEFORE a process is
    opened, so a bad request does not cost an mcode launch."""
    sid = _make_chat(tmp_path)
    r = client.post(f"/api/sessions/{sid}/control", json={"op": "session/prompt"})
    assert r.status_code == 400
    assert "unknown operation" in r.json()["error"]


def test_the_route_404s_an_unknown_chat(client):
    r = client.post("/api/sessions/nope/control", json={"op": "goal_get"})
    assert r.status_code == 404


def test_the_route_reports_a_missing_session_rather_than_lying(client, tmp_path):
    """A chat with no backend session yet is the normal first-turn state, and it must
    say so rather than report an empty goal as if it had read one."""
    sid = _make_chat(tmp_path, backend_sid="")
    r = client.post(f"/api/sessions/{sid}/control", json={"op": "goal_get"})
    assert r.status_code == 502
    assert "no mcode session" in r.json()["error"]


def test_the_route_returns_the_operation_result(client, tmp_path, monkeypatch):
    """The success path, with the transport stubbed: the route's job is to resolve the
    session and the executable and hand off, so that is what is asserted."""
    sid = _make_chat(tmp_path, backend_sid="mvs_fake_session")
    seen = {}

    def fake_drive(op, params, **kw):
        seen.update({"op": op, "params": params, **kw})
        return {"ok": True, "op": op, "result": {"goal": {"objective": "x"}}, "error": ""}

    from rigma import harness_mcode as hm
    monkeypatch.setattr(hm, "drive_control", fake_drive, raising=True)
    monkeypatch.setattr(hm, "bin_path", lambda: "mcode-fake", raising=True)
    r = client.post(f"/api/sessions/{sid}/control",
                    json={"op": "goal_create", "params": {"objective": "x"}})
    assert r.status_code == 200, r.text
    assert r.json()["result"] == {"goal": {"objective": "x"}}
    # The BACKEND session id is what reaches the adapter, not Rigma's chat id — passing
    # the chat id would resume a session mcode has never heard of.
    assert seen["session_id"] == "mvs_fake_session"
    assert seen["op"] == "goal_create"
    assert seen["params"] == {"objective": "x"}


# --- the per-row actions, and where they must NOT appear ----------------------

FRONTEND = pathlib.Path(__file__).resolve().parents[1] / "frontend-v2" / "src" / "chat"


def test_the_live_turn_passes_the_row_handler():
    """A row action needs a turn to act on, and only the LIVE render has one."""
    src = (FRONTEND / "Transcript.tsx").read_text(encoding="utf-8")
    assert "onRowOp={rowOp}" in src, (
        "the live AgentState does not receive the row handler, so no per-row control "
        "can be pressed")


def test_the_durable_turn_does_NOT_pass_the_row_handler():
    """THE RULE, and it is a correctness rule rather than a style one.

    A durable turn is HISTORY. Its queue and its delegation tree name a session that may
    be long gone, so a "stop this child" button on a turn from an hour ago would offer to
    act on something that no longer exists. `AcpBlock` renders the buttons only when it is
    handed a handler, so the durable render must not hand it one.

    Read as source because this project has no `*.test.tsx` — the render layer is covered
    by typecheck plus pure-function tests, and this pins the ONE wiring decision that
    typecheck cannot catch: both call sites are type-valid, and only one is correct.
    """
    src = (FRONTEND / "Transcript.tsx").read_text(encoding="utf-8")
    # The durable builder is the one that reads from `SavedAgentState`.
    i = src.find("durable")
    assert i > 0, "the durable render site is not named `durable` any more"
    # Every `onRowOp=` occurrence must be inside the live `LiveTurn`, which is the only
    # component that defines `rowOp`. There is exactly one.
    assert src.count("onRowOp={rowOp}") == 1, (
        "more than one render site passes the row handler; the durable one must not")


def test_the_row_handler_is_defined_only_where_a_turn_is_live():
    src = (FRONTEND / "Transcript.tsx").read_text(encoding="utf-8")
    # `rowOp` is defined beside `answerApproval`, which is also live-only.
    assert "const rowOp = useCallback" in src
    assert "const answerApproval = useCallback" in src
    # And it reads the session id at CLICK time, not from a closure captured at render —
    # a stale closure would send the operation to whichever chat was open before.
    i = src.index("const rowOp = useCallback")
    body = src[i:i + 900]
    assert "useChat.getState().currentId" in body, (
        "the row handler must read the current chat at click time")


def test_the_row_handler_surfaces_a_refusal():
    """These actions are DESTRUCTIVE — drop a queued message, stop a delegated child.
    A swallowed failure would leave the user believing something happened."""
    src = (FRONTEND / "Transcript.tsx").read_text(encoding="utf-8")
    i = src.index("const rowOp = useCallback")
    body = src[i:i + 900]
    assert "catch" in body and "lastError" in body, (
        "a refused row action would be silent")


def test_the_frontend_offers_only_operations_the_server_allows():
    """Pinned across the language boundary. A typo here is a button that 400s, and it
    would only appear when a user pressed it."""
    plane = (FRONTEND / "controlPlane.ts").read_text(encoding="utf-8")
    import re

    offered = set(re.findall(r'op: "([a-z_]+)"', plane))
    assert offered, "no operations found in controlPlane.ts"
    for name in offered:
        assert name in harness_mcode.CONTROL_OPS, (
            f"the UI offers {name!r}, which the server's allowlist does not have")


def test_the_panel_and_the_row_actions_do_not_overlap():
    """A control in both places is a control a user cannot predict: the same operation
    reached from a form and from a row behaves differently, because the form has no id.

    Read from the ARRAYS rather than from every `op:` literal in the file — the first
    version of this test matched the panel's own table as well and reported an overlap
    that was not there. A test that cannot tell the two tables apart is not checking the
    thing it names.
    """
    import re

    plane = (FRONTEND / "controlPlane.ts").read_text(encoding="utf-8")
    panel = (FRONTEND / "ControlPanel.tsx").read_text(encoding="utf-8")
    form = re.search(r"CONTROL_OPS: ControlOp\[\] = \[(.*?)\n\];", plane, re.S)
    rows = re.search(r"QUEUE_ACTIONS: RowAction\[\] = \[(.*?)\n\];", plane, re.S)
    assert form and rows, "the operation tables were not found"
    form_ops = set(re.findall(r'op: "([a-z_]+)"', form.group(1)))
    row_ops = set(re.findall(r'op: "([a-z_]+)"', rows.group(1)))
    assert form_ops and row_ops, (form_ops, row_ops)

    # AND THE OPERATIONS THE PANEL SENDS DIRECTLY. `config_set` is not in either table —
    # `ControlPanel` passes it as a string literal, because it is driven by the
    # session-settings selects rather than by the operation dropdown. Reading only
    # `controlPlane.ts` left the newest operation outside the guard.
    direct = set(re.findall(r'api\.control\(\s*sessionId,\s*"([a-z_]+)"', panel))
    assert direct, "no direct api.control call was found in ControlPanel"
    form_ops |= direct
    # The overlap that must not exist: the form has no id, so an id-taking operation
    # offered there would be a button that always fails.
    assert not (form_ops & row_ops), (
        f"offered both as a form operation and as a row action: {form_ops & row_ops}")

    # THE REVERSE DIRECTION, which was unchecked. Every operation the UI can send has to be
    # one the server's allowlist declares — a typo in an operation name is otherwise a 400
    # the user discovers by clicking. This is the guard that would have caught it.
    server = set(harness_mcode_acp.CONTROL_OPS)
    unknown = (form_ops | row_ops) - server
    assert not unknown, (
        f"the UI can send operations the server's allowlist does not have: {sorted(unknown)}")
    # And no per-member delegation table, because the protocol has no member-scoped stop.
    assert "MEMBER_ACTIONS" not in plane, (
        "a per-member action table is back; mcode's delegation/stop is session-wide")


# --- the capability disclosure, which went stale once already ----------------

def test_the_menu_does_not_claim_the_turn_cannot_use_acp():
    """R6-ACP-CONTROL. `drives` said the chat turn "does not use it yet", which was true
    when the client landed and stopped being true at R6-ACP-SEAM, when the transport
    selector arrived. The menu was describing a limitation the product no longer had —
    the same defect class as the `mcode login` claim, in the same direction.

    Pinned because a disclosure that lags the code is worse than no disclosure: it tells
    a user not to look for something that is there.
    """
    from rigma import harness

    mcode = next(h for h in harness.list_harnesses() if h["name"] == "mcode")
    assert "does not use it yet" not in mcode["drives"], (
        "the menu still says the turn cannot run over ACP")
    # And it must name the control that actually switches it, or the user cannot act.
    assert "transport" in mcode["drives"]


def test_the_menu_says_the_control_plane_can_be_OPERATED():
    """The disclosure said the control plane was "reachable", which was half true: every
    operation arrived as a notification, so the panel could DRAW it and a user could not
    touch it. "Reachable" is not the same claim as "usable", and the menu's job is to say
    which one is true."""
    from rigma import harness

    mcode = next(h for h in harness.list_harnesses() if h["name"] == "mcode")
    joined = " ".join(mcode["capabilities"])
    assert "control plane" in joined
    assert "OPERATE" in joined or "operate" in joined, (
        "the menu does not say the control plane can be operated, only that it exists")


# --- the method tables, and the guard the code claimed to have ----------------
#
# The comment above `STANDARD_METHODS` said the tables were "kept beside the extension
# list so the drift guard in the tests covers both". There was no such guard: both
# `STANDARD_METHODS` and `EXTENSION_NOTIFICATIONS` appeared exactly once in the tree,
# at their own definition. These are that guard. They matter because every one of these
# strings is a wire fact — a renamed method is a `-32601` at the point of use, and the
# only place the rename can be caught cheaply is here.

def test_every_control_op_dispatches_to_a_method_the_tables_declare():
    """The three lists must agree: the ops the route accepts, the client methods it
    calls, and the mcode methods those wrap.

    This is the guard `harness_mcode_acp.py` claimed to have. It reads the dispatch chain
    as SOURCE rather than calling it, because calling it needs a live process per op and
    the mapping is a static fact.
    """
    import re

    from rigma import harness_mcode_acp as acp

    src = (pathlib.Path(acp.__file__)).read_text(encoding="utf-8")
    start = src.index("def drive_control(")
    end = src.index("def drive_turn_acp(")
    chain = src[start:end]

    # op -> the client method it calls, straight out of the if/elif chain.
    pairs = re.findall(r'op == "([a-z_]+)":\s*\n\s*res = client\.([a-z_]+)\(', chain)
    assert pairs, "the dispatch chain was not found"
    dispatched = dict(pairs)

    # `mode_set` and `config_set` pass positional args, so they are matched separately.
    assert 'client.set_mode(' in chain, "mode_set's call is missing"
    assert 'client.set_config_option(' in chain, "config_set's call is missing"
    dispatched.setdefault("mode_set", "set_mode")
    dispatched.setdefault("config_set", "set_config_option")

    assert set(dispatched) == set(acp.CONTROL_OPS), (
        "the operation table and the dispatch chain disagree: "
        f"table only {set(acp.CONTROL_OPS) - set(dispatched)}, "
        f"dispatch only {set(dispatched) - set(acp.CONTROL_OPS)}")

    # And each of those methods must exist on the client, so the chain cannot name a
    # method that was renamed or never written.
    for op, meth in dispatched.items():
        assert hasattr(acp.AcpClient, meth), (
            f"operation {op!r} dispatches to client.{meth}(), which does not exist")


def test_the_extension_method_table_matches_the_client_methods():
    """Every `mcode/session/...` method the client can send is in `EXTENSION_METHODS`,
    and every entry in that table is really sent by some client method.

    The table's stated job is to make "a version that drops or renames one ... visible
    instead of producing a confusing `Method not found` at the point of use" — which it
    can only do if it is complete and has no entries nothing sends.
    """
    import re

    from rigma import harness_mcode_acp as acp

    src = pathlib.Path(acp.__file__).read_text(encoding="utf-8")
    # Every `self.request("<method>", ...)` literal in the client.
    sent = set(re.findall(r'self\.request\(\s*"([a-z_/]+)"', src))
    # `activate` sends one of two names depending on the server; both must be declared.
    assert "session/activate" in acp.EXTENSION_METHODS
    assert "mcode/session/activate" in acp.EXTENSION_METHODS

    declared = set(acp.EXTENSION_METHODS)
    sent_mcode = {m for m in sent if m.startswith("mcode/") or m == "session/activate"}

    undeclared = sent_mcode - declared
    assert not undeclared, (
        f"the client sends {sorted(undeclared)}, which EXTENSION_METHODS does not declare")
    unused = declared - sent_mcode
    assert not unused, (
        f"EXTENSION_METHODS declares {sorted(unused)}, which no client method sends")


def test_every_standard_method_in_the_table_is_one_the_client_can_send():
    """`STANDARD_METHODS` is documentation-with-a-guard: it lists the standard ACP
    session methods Rigma uses. Every entry must be reachable from a client method, or
    the table is describing a surface the code does not have."""
    import re

    from rigma import harness_mcode_acp as acp

    src = pathlib.Path(acp.__file__).read_text(encoding="utf-8")
    sent = set(re.findall(r'self\.request\(\s*"([a-z_/]+)"', src))
    unreachable = set(acp.STANDARD_METHODS) - sent
    # `session/new` and `session/prompt` are sent by the turn path through
    # `_session_new`/`prompt`, so accept them if the string appears anywhere.
    unreachable = {m for m in unreachable if f'"{m}"' not in src}
    assert not unreachable, (
        f"STANDARD_METHODS lists {sorted(unreachable)}, which no code path sends")


def test_every_extension_notification_is_actually_mapped():
    """The four notifications are gated on a capability we DECLARE, so declaring interest
    in one and then not mapping it is a fact the UI silently never sees. The check is
    against `map_acp_update`'s own source: each method name must appear there as a
    literal, because a notification reaching the mapper and falling through is the
    failure mode this guards.
    """
    import inspect

    from rigma import harness_mcode_acp as acp

    src = inspect.getsource(acp.map_acp_update)
    for note in acp.EXTENSION_NOTIFICATIONS:
        assert f'"{note}"' in src, (
            f"{note} is declared in EXTENSION_NOTIFICATIONS but map_acp_update "
            "does not handle it")


def test_the_docstring_does_not_claim_a_union_size_it_cannot_know():
    """`map_acp_update` said "the protocol's union has 13 members". The ACP SDK resolved
    in the DSH checkout (`@agentclientprotocol/sdk@1.4.0`) declares 15 — it adds
    `compaction_update` and `compaction_summary_chunk`.

    The number is a wire fact that moves when the SDK does, so the docstring must not
    freeze it. Pinned as a PROHIBITION on the specific stale count rather than as a
    requirement to name a new one: naming 15 would go stale the same way.
    """
    import inspect

    from rigma import harness_mcode_acp as acp

    src = inspect.getsource(acp.map_acp_update)
    assert "union has 13 members" not in src, (
        "the docstring still states a union size; the SDK in this checkout declares 15 "
        "and the number will keep moving")


# --- the session id is CONTEXT, not a parameter ------------------------------

def test_a_request_body_cannot_redirect_the_operation_to_another_session(fake, sid, tmp_path):
    """The route's invariant is that the backend session comes from the CHAT ROW, not the
    request body. It was bypassable: every client method builds `{"sessionId":
    self.session_id}` and then `body.update(params)`, so a body carrying `sessionId`
    replaced the resumed session — and `delegation_stop` REQUIRES a `sessionId`, which
    made the override look like an ordinary argument.

    The fake now records every session id a request names, which is what makes this
    observable at all: its state is one blob keyed by nothing, so before that ledger a
    client that honoured the override and one that ignored it were indistinguishable.
    """
    exe, state_file = fake

    # Create the goal, naming SOMEONE ELSE'S session in the params.
    out = op("goal_create", exe, sid, objective="mine", sessionId="someone-elses")
    assert out["ok"], out

    named = json.loads(state_file.read_text(encoding="utf-8"))["named"]
    assert "someone-elses" not in named, (
        "a request body replaced the resumed session id — the operation ran against a "
        f"session the caller chose rather than the one the route resolved: {named}")
    assert sid in named, f"the resumed session id was never named: {named}"


def test_delegation_stop_takes_no_member_id_because_the_server_has_none(fake, sid):
    """`delegation_stop` is ROOT-scoped, measured from mcode's own handler:

        onRequest("mcode/session/delegation/stop", xt, async({params:i}) => {
            let s = r(i.sessionId), a = await mo(e.runtime, s);
            return {receipt: await e.runtime.stopDelegation(a)}})

    `i.sessionId` is resolved as the root session and the whole delegation tree for it is
    stopped. There is no member id, and the answer is a `receipt`.

    This test exists because the operation used to REQUIRE a `sessionId` param, which was
    not a member id at all — it was the root id spelled the same way as the context key.
    That collision is what let a request body redirect the operation, and it also made the
    UI offer a per-child "stop" button for a control the protocol does not have.
    """
    exe, state_file = fake
    out = op("delegation_stop", exe, sid)
    assert out["ok"], out
    # A receipt, not a member: the shape is part of the contract.
    assert "receipt" in json.dumps(out), out


def test_delegation_stop_cannot_be_aimed_at_another_session(fake, sid):
    """And the root id it uses is the RESUMED one, not anything the body supplied."""
    exe, state_file = fake
    out = op("delegation_stop", exe, sid, sessionId="someone-elses")
    assert out["ok"], out
    named = json.loads(state_file.read_text(encoding="utf-8"))["named"]
    assert "someone-elses" not in named, named
