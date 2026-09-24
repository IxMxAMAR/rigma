"""Area 13 (adversarial security) guard tests for the shared surface.

Every test here calls the REAL guard / decision function on command TEXT or on a
synthetic request object. Nothing dangerous is ever handed to something that
would execute it (safety rule 7), and no test performs network I/O.
"""
from rigma import tools


# --- 13-3: the exec gate, not the destructive-command regex -------------------
#
# The regex is a literal-text advisory: any wrapper (base64, -EncodedCommand, a
# nested interpreter) defeats it. These call `exec_decision` on the TEXT.

# A base64-wrapped payload: the literal text contains no blocked word.
_OBFUSCATED = (
    "powershell -NoProfile -Command \"$c=[Text.Encoding]::Unicode.GetString("
    "[Convert]::FromBase64String('UwB0AG8AcAAuAC4A')); Invoke-Expression $c\"")


def test_the_regex_cannot_see_an_obfuscated_payload():
    # this is the finding, asserted rather than assumed
    assert tools._BLOCKED_CMD.search(_OBFUSCATED) is None


def test_an_obfuscated_command_is_gated_by_confirmation():
    allowed, why = tools.exec_decision(
        _OBFUSCATED, {"allow_code": True, "confirm_exec": False})
    assert allowed is False
    assert "confirmation" in why


def test_execution_is_allowed_once_the_session_confirms():
    allowed, why = tools.exec_decision(
        "echo hi", {"allow_code": True, "confirm_exec": True})
    assert allowed is True and why == ""


def test_the_literal_destructive_form_is_still_refused():
    allowed, why = tools.exec_decision(
        "Stop-Computer -Force", {"allow_code": True, "confirm_exec": True})
    assert allowed is False and "blocked" in why


def test_deletion_is_refused_under_the_no_delete_profile():
    allowed, why = tools.exec_decision(
        "del a.txt", {"allow_code": True, "profile": "no-delete",
                      "confirm_exec": True})
    assert allowed is False and "deletion" in why


def test_confined_refuses_execution():
    allowed, why = tools.exec_decision(
        "echo hi", {"allow_code": True, "profile": "confined",
                    "confirm_exec": True})
    assert allowed is False and "confined" in why


def test_python_is_gated_the_same_way():
    allowed, why = tools.exec_decision(
        "print(1)", {"allow_code": True, "confirm_exec": False},
        python_src="print(1)")
    assert allowed is False and "confirmation" in why


def test_run_shell_does_not_spawn_when_confirmation_is_off(monkeypatch):
    """The wiring, not just the decision: the handler must not reach
    `_run_subprocess` at all. The launcher is stubbed so nothing can run."""
    called = []

    def _boom(*a, **k):
        called.append(a)
        return "ran"

    monkeypatch.setattr(tools, "_run_subprocess", _boom)
    out = tools.run_tool("run_shell", {"command": "echo hi"},
                         {"allow_code": True, "confirm_exec": False})
    assert out.startswith("error") and "confirmation" in out
    assert called == []
