"""B6a: the ACP session's own mode list, surfaced on the session-settings event.

WHY THIS FILE EXISTS. `mode_set` is reachable over HTTP and has no UI, because the
valid `modeId`s arrive in `session/new`'s `modes.availableModes` and Rigma never put
them on the wire — the only place they appeared was `probe_surface`, which is
tests-only. Hardcoding `plan`/`default` was rejected, correctly: a control whose
values stop matching the server on the next mcode version is worse than no control.

WHAT IS TESTED HERE is the adapter half: the client RECORDS the modes the session
advertises (`session/new`, `session/resume`, and every `current_mode_update`), and
the turn path MIRRORS them on the `acp_config` event the panel already draws for the
session's `configOptions`. That is the SAME event, not a second channel — an
`acp_config` payload may carry `configOptions`, `modes`, or both, and a consumer
replaces only the field it was sent.

WHAT IS *NOT* TESTED HERE, and must not be implied by a green run: any frontend
control. `chatStore.ts` has no `acpModes` field yet, so nothing renders this today;
that work is named in the commit message and the B6a report. Nor is a live mcode turn
run: every test drives `tests/fake_acp_server.py`, a real subprocess speaking real
JSON-RPC over real pipes, which loads no model.
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


def _drive(prompt="hello", *, state=None, timeout=30.0, **kw):
    return list(acp.drive_turn_acp(prompt, exe=sys.executable, state=state,
                                   permission="full", timeout=timeout, **kw))


def _settings(events):
    """Every `acp_config` event, in order. The modes ride this event."""
    return [e for e in events if e.kind == "state" and e.event == "acp_config"]


# --- the payload is the server's, never a constant ----------------------------


def test_the_payload_never_invents_a_mode():
    """The whole reason `mode_set` had no UI: a hardcoded list is a false control.

    Every "no list here" shape must come out as an EXPLICIT unknown — an empty
    `availableModes` and an empty `currentModeId` — rather than the `plan`/`default`
    pair that was rejected. `known` separates "the session advertised an empty list"
    from "no list was advertised at all", which is a fact the UI needs to decide
    whether to draw the control.
    """
    for raw in (None, {}, {"availableModes": None}, {"availableModes": []}, "plan", []):
        payload = acp.acp_modes_payload(raw)
        assert payload["availableModes"] == [], (raw, payload)
        assert payload["currentModeId"] == "", (raw, payload)

    # A real list is passed through, and only dict entries survive.
    payload = acp.acp_modes_payload({
        "availableModes": [{"id": "default"}, "junk", {"id": "plan", "name": "Plan"}],
        "currentModeId": "plan",
    })
    assert payload == {
        "availableModes": [{"id": "default"}, {"id": "plan", "name": "Plan"}],
        "currentModeId": "plan",
        "known": True,
    }, payload


# --- the handshake's list reaches the event -----------------------------------


def test_the_modes_the_session_advertises_reach_the_event(fake_acp):
    """`session/new` answered with `availableModes` and nothing forwarded it.

    The list is asserted to be the one the SERVER sent — a different set from the
    default is used so a hardcoded `default`/`plan` list cannot pass this.
    """
    fake_acp["argv_extra"] = ["--modes", "default,plan,acceptEdits"]
    events = _drive("hello")
    settings = _settings(events)
    assert settings, ("the session's advertised modes must be mirrored on acp_config; "
                      f"got {[e.event for e in events if e.kind == 'state']}")
    modes = settings[0].data.get("modes")
    assert isinstance(modes, dict), modes
    assert modes.get("known") is True, modes
    assert [m["id"] for m in modes["availableModes"]] == [
        "default", "plan", "acceptEdits"], modes
    assert modes["currentModeId"] == "default", modes


def test_an_empty_advertised_list_is_known_and_empty(fake_acp):
    """An empty list the server really sent is `known: True` with no entries."""
    fake_acp["argv_extra"] = ["--modes", ""]
    events = _drive("hello")
    modes = _settings(events)[0].data["modes"]
    assert modes["availableModes"] == [], modes
    assert modes["known"] is True, modes


def test_a_session_that_advertises_nothing_says_unknown(fake_acp, monkeypatch):
    """No list at all is `known: False`, not a fabricated default.

    The handshake is made to answer without a `modes` object, which is what a
    server that does not support modes looks like.
    """
    real_resume = acp.AcpClient.session_resume

    def _strip_modes(self, *a, **kw):
        res = real_resume(self, *a, **kw)
        self.modes = {}
        return res

    monkeypatch.setattr(acp.AcpClient, "session_resume", _strip_modes)
    events = _drive("hello", state={"session_id": "mvs_fake_session"})
    modes = _settings(events)[0].data["modes"]
    assert modes["availableModes"] == [], modes
    assert modes["currentModeId"] == "", modes
    assert modes["known"] is False, modes


# --- the list updates when the session changes modes --------------------------


def test_a_mid_session_mode_change_is_mirrored(fake_acp, monkeypatch):
    """The modes must update DURING a session, not only at the handshake.

    ACP's `current_mode_update` carries only the new `currentModeId`; the available
    list came from the handshake. The notification is dispatched through the client's
    REAL `_handle_notification`, so this exercises the same path a server push takes:
    the client records the new id and the turn path re-mirrors the whole settings
    snapshot. Without the re-mirror the panel would keep drawing the handshake's
    `currentModeId` for the rest of the session.
    """
    fake_acp["argv_extra"] = ["--modes", "default,plan"]
    real_prompt = acp.AcpClient.prompt

    def _prompt_after_a_mode_change(self, text, timeout=None):
        self._handle_notification({"method": "session/update", "params": {
            "sessionId": self.session_id,
            "update": {"sessionUpdate": "current_mode_update",
                       "currentModeId": "plan"}}})
        return real_prompt(self, text, timeout=timeout)

    monkeypatch.setattr(acp.AcpClient, "prompt", _prompt_after_a_mode_change)
    settings = _settings(_drive("hello"))
    assert len(settings) >= 2, (
        "a mode change must re-mirror the settings, not leave the handshake's value "
        f"on screen; got {len(settings)} acp_config event(s)")
    first, last = settings[0].data["modes"], settings[-1].data["modes"]
    assert first["currentModeId"] == "default", first
    assert last["currentModeId"] == "plan", last
    # And the advertised list is NOT blanked by a change that does not carry it.
    assert [m["id"] for m in last["availableModes"]] == ["default", "plan"], last
    assert last["known"] is True, last


def test_the_client_records_the_current_mode_it_is_told():
    """The RECORD half, without a turn: `AcpClient` keeps its own view current."""
    client = acp.AcpClient(["true"])
    client.modes = {"availableModes": [{"id": "default"}, {"id": "plan"}],
                    "currentModeId": "default"}
    client._handle_notification({"method": "session/update", "params": {
        "update": {"sessionUpdate": "current_mode_update", "currentModeId": "plan"}}})
    assert client.current_mode() == "plan"
    assert [m["id"] for m in client.available_modes()] == ["default", "plan"]
    # An unrelated update changes nothing.
    client._handle_notification({"method": "session/update", "params": {
        "update": {"sessionUpdate": "agent_message_chunk",
                   "content": {"type": "text", "text": "hi"}}}})
    assert client.current_mode() == "plan"
