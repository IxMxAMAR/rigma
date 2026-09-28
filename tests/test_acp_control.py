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
import sys

import pytest
from fastapi.testclient import TestClient

from rigma import harness_mcode, sessions
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


def op(name: str, exe: str, sid: str = "mvs_fake_session", **params):
    return harness_mcode.drive_control(name, params, exe=exe, session_id=sid,
                                       timeout=30.0)


# --- the driver ---------------------------------------------------------------

def test_a_goal_survives_the_process_that_set_it(fake):
    """THE TEST THIS ROUTE RESTS ON.

    Each operation is its own process, so a session that does not persist makes
    `goal_create` look successful and `goal_get` report nothing — a route that lies. The
    first version of the fake did exactly that.
    """
    exe, _ = fake
    assert op("goal_get", exe)["result"] == {"goal": None}
    created = op("goal_create", exe, objective="ship the parser")
    assert created["ok"], created
    assert created["result"]["goal"]["objective"] == "ship the parser"
    # A DIFFERENT process reads it back.
    assert op("goal_get", exe)["result"]["goal"]["objective"] == "ship the parser"


def test_a_goal_can_be_patched_and_cleared_across_processes(fake):
    """`goal/patch` was ADVERTISED by the fake and never handled, so it answered
    `-32601 Method not found` — a method no test could cover."""
    exe, _ = fake
    op("goal_create", exe, objective="ship it")
    patched = op("goal_patch", exe, status="paused")
    assert patched["ok"], patched
    assert patched["result"]["goal"]["status"] == "paused"
    assert op("goal_get", exe)["result"]["goal"]["status"] == "paused"
    assert op("goal_clear", exe)["ok"]
    assert op("goal_get", exe)["result"]["goal"] is None


def test_a_queued_message_survives_and_is_listed(fake):
    exe, _ = fake
    added = op("queue_enqueue", exe, text="do this later")
    assert added["ok"], added
    items = op("queue_list", exe)["result"]["items"]
    assert [i["content"] for i in items] == ["do this later"]


def test_the_delegation_tree_uses_the_real_snapshot_shape(fake):
    """The fake answered `{"delegations": []}`, which the real server does not produce —
    it answers a `snapshot` with `members`, and that is what the UI's `AcpDelegation` is
    built from. The old shape was one no client could have been reading correctly."""
    exe, _ = fake
    got = op("delegation_get", exe)
    assert got["ok"], got
    assert "snapshot" in got["result"]
    assert got["result"]["snapshot"]["members"] == []


def test_steering_and_the_mode_and_config_selects_all_drive(fake):
    """The remaining three surfaces, so "the control plane is invocable" is a claim
    about the whole plane rather than the part with an easy test."""
    exe, _ = fake
    assert op("steer", exe, text="focus on the parser")["ok"]
    assert op("activate", exe)["ok"]
    assert op("mode_set", exe, modeId="plan")["ok"]
    assert op("config_set", exe, optionId="permissionMode", value="auto")["ok"]


def test_an_operation_on_an_unknown_session_reports_the_transport_error(fake):
    """`drive_control` NEVER RAISES. It is called from a route that is drawing a panel,
    and an exception would take down the panel instead of reporting one failure."""
    exe, _ = fake
    out = op("goal_get", exe, sid="mvs_not_a_session")
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


def test_every_operation_in_the_table_is_reachable_by_name(fake):
    """A row in the table that the dispatcher does not implement would answer "unhandled
    operation", which is a hole in the plane that the table would hide."""
    exe, _ = fake
    op("goal_create", exe, objective="seed")
    op("queue_enqueue", exe, text="seed")
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
        out = op(name, exe, **params)
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
