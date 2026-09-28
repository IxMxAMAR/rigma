"""R6-ACP-REACH: the code told a stuck user that the fix does not exist.

TWO DEFECTS, both about the product asserting something false.

1. `_interaction_dead_end`'s docstring said the permanent block was "not fixable here"
   because "mcode's ACP path" was unavailable and "ACP needs `mcode login`". BOTH claims
   are false, and both had ALREADY been corrected elsewhere: `c9f9f4a` established that
   ACP needs no login (the earlier "blocked on authentication" was a probe bug — stdin
   was attached to a FILE, so mcode hit EOF before it answered), and `a7caded` shipped
   Rigma's own ACP client. The correction reached the docs and never reached this
   function.

2. The user-facing recovery sentence repeated it — "mcode's own TUI or ACP client, which
   Rigma cannot drive. Start a NEW chat" — so a user whose chat had just died was told to
   abandon it, when the remedy Rigma had built was one selector away.

A stale sentence is not cosmetic here. It is the difference between a user losing their
conversation and switching a dropdown.

No model is loaded.
"""
from __future__ import annotations

import inspect

from rigma import harness_mcode as mcode


def _recovery() -> str:
    """The real sentence, from the real trigger rather than a copy of the text."""
    return mcode._interaction_dead_end("requires an interactive host")


def test_the_message_no_longer_says_rigma_cannot_drive_acp():
    """The claim that was false. Rigma has had an ACP client since a7caded."""
    msg = _recovery().lower()
    assert "cannot drive" not in msg, (
        "Rigma drives ACP; the message must not say otherwise")
    assert "can't drive" not in msg


def test_the_message_no_longer_says_acp_needs_a_login():
    """Retracted in c9f9f4a: the probe attached stdin to a file, so mcode hit EOF before
    answering. An unanswered probe is not a negative result."""
    assert "login" not in _recovery().lower()


def test_the_message_points_at_the_selector_that_fixes_it():
    """The remedy has to be ACTIONABLE. "Start a new chat" throws away the conversation;
    switching the transport does not."""
    msg = _recovery()
    assert "acp" in msg, "the one remedy Rigma built is not named"
    assert "transport" in msg, "the user is not told WHICH control to change"


def test_the_message_still_says_retrying_cannot_work():
    """The trap is that retrying looks reasonable and fails in seconds. Removing the
    warning while adding the remedy would trade one wrong message for another."""
    assert "retrying cannot succeed" in _recovery().lower()


def test_the_message_still_offers_the_permission_half_of_the_fix():
    """`full` avoids the permission-request half, and that was the one true part of the
    old sentence."""
    assert "full" in _recovery()


def test_a_clean_stderr_produces_no_message():
    """The detector must not fire on a turn that ended normally — a recovery sentence on
    a working chat would be its own kind of lie."""
    assert mcode._interaction_dead_end("") == ""
    assert mcode._interaction_dead_end("some other failure") == ""


def test_the_docstring_records_the_correction_and_its_source():
    """Pinned because the failure mode was a correction that reached the docs and not the
    code. The docstring has to carry the commit SHAs, so the next reader can check."""
    doc = inspect.getdoc(mcode._interaction_dead_end) or ""
    assert "c9f9f4a" in doc, "the correction's source is not cited"
    assert "a7caded" in doc


def test_the_docstring_does_not_ASSERT_the_retracted_claim():
    """It may QUOTE the old claim — that is how a correction is recorded — but it must
    not assert it.

    An earlier version of this test searched for the bare phrase and failed on the
    docstring's own quotation of it. That is the right kind of test failure (it caught a
    test that could not tell quoting from asserting) and the wrong assertion, so the
    check is on the ASSERTION rather than the words: the retracted sentence began
    "NOT fixable here:", and a correction quotes it rather than repeating it.
    """
    doc = inspect.getdoc(mcode._interaction_dead_end) or ""
    assert "NOT fixable here:" not in doc, (
        "the retracted claim is still being asserted")
    # And the correction has to be visible as one, or a reader cannot tell which half is
    # current.
    assert "false" in doc.lower(), "the docstring does not say the old claim was wrong"


def test_the_docstring_does_not_promise_the_replay():
    """`acp` is an interactive host, so the switch is a real remedy. Whether mcode
    REPLAYS a pending request on a resumed session is NOT verified — no live ACP session
    has been driven — so the docstring must not claim it."""
    doc = (inspect.getdoc(mcode._interaction_dead_end) or "").lower()
    assert "not verified" in doc, (
        "the unverified part must be labelled, not implied")
