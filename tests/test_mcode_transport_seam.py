"""R6-ACP-TURN: selecting the ACP wire, and what happens when it is not there.

WHY THIS FILE EXISTS. `drive_turn_acp` is only useful if a chat can choose it, and
the choice crosses a boundary where a silent fallback would be the worst outcome: if
the field says `acp` and `exec` runs instead, the user believes a permission prompt
could have been answered when it could not. So the seam's job is to run the wire the
chat asked for or to SAY IT COULD NOT — never to quietly run the other one.

These tests do not load a model and do not start a harness.
"""
from __future__ import annotations


from rigma import harness, sessions


def test_the_transport_field_is_a_session_field_with_a_type():
    """A field the PATCH surface offers must be typed, or a client can store a list
    where the adapter compares a string."""
    assert "mcode_transport" in sessions.MUTABLE_FIELDS
    assert sessions._FIELD_TYPES["mcode_transport"] is str


def test_only_exec_and_acp_are_valid_transports():
    assert sessions.MCODE_TRANSPORTS == ("exec", "acp")


def test_mcode_exposes_the_acp_driver_as_a_capability():
    """The seam looks for this ATTRIBUTE, not the module's name.

    A name check keeps answering True after a rename, and then the ACP path is
    reported as taken while `exec` silently runs.
    """
    mcode = harness.adapter("mcode")
    assert mcode is not None
    assert callable(getattr(mcode, "drive_turn_acp", None)), (
        "the ACP driver must be reachable as `adapter.drive_turn_acp`")


def test_dsh_does_NOT_expose_it():
    """And the capability is not accidentally universal.

    DSH has no ACP driver, so a DSH chat with the field set to `acp` must be refused
    rather than run over its own transport — which is what the seam's guard does, and
    what this pins the precondition for.
    """
    dsh = harness.adapter("dsh")
    assert dsh is not None
    assert getattr(dsh, "drive_turn_acp", None) is None


def test_the_acp_driver_has_the_seam_s_call_signature():
    """The seam calls it with keywords only, exactly as it calls `drive_turn`.

    Checked by INSPECTION rather than by calling it, because calling it would start a
    subprocess and, with a real engine behind it, a model.
    """
    import inspect

    from rigma import harness_mcode_acp

    params = inspect.signature(harness_mcode_acp.drive_turn_acp).parameters
    for name in ("base_url", "model", "system_prompt", "session_id", "cwd",
                 "max_tokens", "context_window", "state", "cancel", "permission"):
        assert name in params, f"the seam passes `{name}` and the driver must accept it"


def test_the_acp_driver_declares_what_it_does_not_do():
    """The driver is a PARALLEL path, not the default, and the adapter's `drives`
    line must say so — a menu that implies the chat turn already uses ACP would be
    describing a turn that is not happening."""
    # `list_harnesses()` is the module's own registry accessor and returns dicts, so
    # the row is selected by name rather than by position — an order-dependent test
    # would break the day a backend is inserted.
    rows = {h["name"]: h for h in harness.list_harnesses()}
    drives = rows[harness.MCODE]["drives"]
    assert "exec" in drives
    assert "ACP" in drives or "acp" in drives
    assert "does not use it yet" in drives, drives
