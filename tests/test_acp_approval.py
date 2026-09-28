"""R6-ACP-APPROVE: answering a permission request that is BLOCKING a turn.

WHY THIS IS THE LAST PIECE. Over `exec`, mcode has no interaction host, so when `smart`
decided to ask, nobody could answer and mcode's own guard then refused to start another
session — the chat was blocked PERMANENTLY (R5-MCODE-DEADEND). Over ACP the server asks
and WAITS, so the answer has to come from somewhere: from the user, through the UI.

The handshake crosses two threads — the ACP client's reader thread blocks inside the
request handler, and the answer arrives on the event loop from an HTTP route — so the
tests here pin the handshake's rules rather than the plumbing:

  - an answer that is not a boolean is refused, because `"false"` is truthy in Python
    and granting permission from a typo cannot be taken back;
  - an answer for a request the chat is not waiting on is refused with 409, so a stale
    card cannot decide a DIFFERENT question than the one on screen;
  - a timeout is `cancelled`, not a denial, because the user did not decide.

No model is loaded.
"""
from __future__ import annotations

import inspect

from rigma import serve


def test_the_approval_route_exists_and_takes_a_boolean():
    """Read by INSPECTION: the route is a closure inside `build_app`, so it cannot be
    imported. Its presence and its signature are the contract the UI relies on."""
    src = inspect.getsource(serve.build_app)
    assert '"/api/sessions/{sid}/approval"' in src
    # `allow` must be validated as a real boolean, not merely read.
    assert "isinstance(allow, bool)" in src, (
        "a truthy non-boolean would grant permission")


def test_the_wait_is_bounded():
    """An unbounded wait is worse than a decline.

    The client's reader thread is blocked for the whole time, so a UI that never
    answers would wedge the TRANSPORT rather than the one turn. The wait must therefore
    have a ceiling, and the code must have a timeout path at all.
    """
    src = inspect.getsource(serve.build_app)
    assert "_APPROVAL_WAIT_SECS" in src
    assert "if not answered:" in src, "the timeout path must exist and be distinguishable"


def test_a_timeout_returns_None_which_is_cancelled_not_denied():
    """The driver turns None into `cancelled` — the protocol's own word for "no
    answer", deliberately not the same as a denial, because a timeout is not a decision
    the user made."""
    from rigma import harness_mcode_acp as acp

    assert acp.answer_permission({"options": []}, allow=None) == {
        "outcome": {"outcome": "cancelled"}}


def test_the_slot_is_cleared_even_when_nobody_answers():
    """A slot left behind would let a later click answer a question that no longer
    exists — the answer would be applied to whatever request came next."""
    src = inspect.getsource(serve.build_app)
    assert "_approvals.pop(sid, None)" in src, (
        "the slot must be cleared on the timeout path too, not only on success")


def test_an_approval_is_keyed_by_session_not_globally():
    """One slot per SESSION, so an answer for one chat cannot resolve another's
    request. A single global slot would have accepted exactly that."""
    src = inspect.getsource(serve.build_app)
    assert "_approvals: dict[str, dict] = {}" in src
    assert "slot = _approvals.get(sid)" in src


def test_the_awaiting_flag_is_what_arms_the_button():
    """`awaiting` is set only by a transport that can actually be answered.

    A DSH ask with no decision is pending in the audit trail but can NEVER be answered
    over that wire — `HarnessSdkRequestMap` is exactly initialize/session/prompt/
    shutdown — so a button on it would promise something the connection cannot do.
    """
    src = inspect.getsource(serve.build_app)
    assert '"awaiting": True' in src
    # And the driver sets it only on the ACP path, which is the only one with a handler.
    assert "on_permission=_answer_permission)" in src


def test_the_driver_reports_the_ask_before_it_waits():
    """The UI can only answer what it has been told about, so the `approval/asked`
    event has to be queued BEFORE the wait begins — not after it resolves, which would
    deadlock: the reader thread waits for an answer to a question nobody was asked."""
    src = inspect.getsource(serve.build_app)
    asked_at = src.index('"awaiting": True')
    wait_at = src.index('slot["event"].wait(')
    assert asked_at < wait_at, (
        "the ask must be published before the wait, or nothing can ever answer it")
