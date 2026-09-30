"""ODR-1: the PRODUCT-DEFAULT home workspace must not be a persistence primitive.

The defect this file exists for: a chat with no workspace is given
`workspace = Path.home()` (serve.py: `s.get("workspace") or str(Path.home())`),
and `write_file` / `edit_file` / a RELATIVE `copy_files` destination then pass
through `_ws_path`, which checks containment and NOTHING else. Every path in the
tests below is INSIDE the home workspace, so containment does not stop it — only
the persistence denylist can. That is why the workspace here is the default
`Path.home()` and not a `tmp_path` folder: the test gap ODR-1 names is that
every existing test passed an explicit narrow workspace.

Hermetic: `Path.home` and the Windows environment variables that name the
persistence folders are monkeypatched onto `tmp_path`, so the real Startup
folder is never touched.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rigma import tools


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake user profile wired the way the product wires the real one."""
    h = tmp_path / "home"
    appdata = h / "AppData" / "Roaming"
    programdata = tmp_path / "ProgramData"
    systemroot = tmp_path / "Windows"
    for d in (appdata, programdata, systemroot / "System32"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("LOCALAPPDATA", str(h / "AppData" / "Local"))
    monkeypatch.setenv("PROGRAMDATA", str(programdata))
    monkeypatch.setenv("SystemRoot", str(systemroot))
    monkeypatch.setenv("WINDIR", str(systemroot))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    return h


def _ctx(**extra):
    """The ctx serve.py builds for a chat with no workspace."""
    return {"workspace": str(Path.home()), "allow_code": True, **extra}


def _write(rel, content="payload"):
    return tools.run_tool("write_file", {"path": rel, "content": content}, _ctx())


def _refused(out):
    assert out.startswith("error"), out
    assert "persistence" in out.lower(), out


def _startup(home):
    return (home / "AppData" / "Roaming" / "Microsoft" / "Windows"
            / "Start Menu" / "Programs" / "Startup")


# --- the control: an ordinary write in the SAME default home workspace --------

@pytest.mark.parametrize("rel", [
    "notes/hello.txt",
    "project/src/main.py",
    ".gitignore",
    ".config/myapp/settings.json",       # not ~/.config/autostart
    "Documents/thesis.docx",
])
def test_an_ordinary_write_in_the_default_home_workspace_still_succeeds(home,
                                                                       rel):
    out = _write(rel)
    assert not out.startswith("error"), out
    assert (home / rel).is_file()


# --- the finding: a persistence write in the same workspace is refused --------

_USER_PERSISTENCE = [
    # user Startup: the classic per-user persistence point
    "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/x.cmd",
    "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/x.bat",
    # Start Menu > Programs (a .lnk runs from the menu too)
    "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/evil.lnk",
    # the rest of %APPDATA%\Microsoft\Windows (Templates / SendTo / ...)
    "AppData/Roaming/Microsoft/Windows/evil.dll",
    # PowerShell profiles, both default locations
    "Documents/WindowsPowerShell/Microsoft.PowerShell_profile.ps1",
    "Documents/PowerShell/Microsoft.PowerShell_profile.ps1",
    # git runs core.fsmonitor / aliases from these
    ".gitconfig",
    "_gitconfig",
    ".config/git/config",
    # POSIX startup files
    ".bashrc",
    ".profile",
    ".zshrc",
    ".config/autostart/x.desktop",
]


@pytest.mark.parametrize("rel", _USER_PERSISTENCE)
def test_a_persistence_write_inside_the_default_home_workspace_is_refused(
        home, rel):
    out = _write(rel)
    _refused(out)
    assert not (home / rel).exists(), f"{rel} was created"


def test_edit_file_cannot_rewrite_a_powershell_profile(home):
    """edit_file is the way a model MODIFIES a persistence file it could not
    create — same refusal, on the same resolved path."""
    prof = home / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1"
    prof.parent.mkdir(parents=True, exist_ok=True)
    prof.write_text("# original\n", encoding="utf-8")
    out = tools.run_tool(
        "edit_file",
        {"path": "Documents/PowerShell/Microsoft.PowerShell_profile.ps1",
         "old": "# original", "new": "Invoke-Evil"},
        _ctx())
    _refused(out)
    assert prof.read_text(encoding="utf-8") == "# original\n"


def test_a_relative_copy_destination_into_startup_is_refused(home):
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool(
        "copy_files",
        {"paths": ["a.txt"],
         "dest": "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"},
        _ctx())
    _refused(out)
    assert not (_startup(home) / "a.txt").exists()


def test_a_relative_move_destination_into_startup_is_refused(home):
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool(
        "move_files",
        {"paths": ["a.txt"],
         "dest": "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"},
        _ctx())
    _refused(out)
    assert (home / "a.txt").is_file()          # source untouched
    assert not (_startup(home) / "a.txt").exists()


def test_a_relative_destination_to_a_shell_rc_is_refused(home):
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": ".config/autostart"},
                         _ctx())
    _refused(out)
    assert not (home / ".config" / "autostart" / "a.txt").exists()


# --- the grant and the allowlist must not reach it ---------------------------

def test_the_absolute_write_grant_does_not_reach_user_startup(home):
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool(
        "copy_files", {"paths": ["a.txt"], "dest": str(_startup(home))},
        _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (_startup(home) / "a.txt").exists()


def test_an_allowlisted_root_does_not_reach_user_startup(home):
    (home / "a.txt").write_text("x", encoding="utf-8")
    appdata = Path(os.environ["APPDATA"])
    out = tools.run_tool(
        "copy_files", {"paths": ["a.txt"], "dest": str(_startup(home))},
        _ctx(write_allowlist=[str(appdata)]))
    _refused(out)
    assert not (_startup(home) / "a.txt").exists()


def test_the_common_startup_folder_is_refused_through_the_grant(home):
    dest = (Path(os.environ["PROGRAMDATA"]) / "Microsoft" / "Windows"
            / "Start Menu" / "Programs" / "StartUp")
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": str(dest)},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (dest / "a.txt").exists()


def test_scheduled_tasks_are_refused_through_the_grant(home):
    dest = Path(os.environ["SystemRoot"]) / "System32" / "Tasks"
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": str(dest)},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (dest / "a.txt").exists()


# --- the verifier's evasions: resolution must see through all of them ---------

@pytest.mark.skipif(os.name != "nt", reason="Windows path spelling")
@pytest.mark.parametrize("rel", [
    # mixed separators
    "appdata\\roaming/MICROSOFT\\windows/start menu/programs/STARTUP/x.cmd",
    # a dot segment and a `..` that resolves back into Startup
    "AppData/Roaming/./Microsoft/Windows/Start Menu/Programs/Startup/../Startup/x.cmd",
    # a `..` from a sibling folder
    "Documents/../AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/x.cmd",
    # trailing dot/space on a component (Windows strips them)
    "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup./x.cmd",
])
def test_case_separators_and_dotdot_still_land_on_startup(home, rel):
    out = _write(rel)
    _refused(out)


@pytest.mark.skipif(os.name != "nt", reason="8.3 names are a Windows property")
def test_an_8dot3_short_name_of_startup_is_refused(home):
    """`MICROS~1` / `STARTM~1` must resolve to the real folders, not become a
    differently-named directory inside the workspace."""
    import ctypes

    startup = _startup(home)
    startup.mkdir(parents=True, exist_ok=True)
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(startup), buf, 1024)
    short = buf.value
    if not n or Path(short) == startup:
        pytest.skip("8.3 aliases are disabled on this volume")
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": short},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (startup / "a.txt").exists()


def _junction(link: Path, target: Path) -> bool:
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True, text=True)
    return r.returncode == 0 and link.exists()


@pytest.mark.skipif(sys.platform != "win32",
                    reason="junctions are a Windows reparse point")
def test_a_junction_inside_the_workspace_does_not_reach_startup(home):
    """`_ws_path`/`_write_path` resolve the path before the denylist runs, so a
    junction that lexically sits inside the workspace resolves to the real
    Startup folder and is refused."""
    startup = _startup(home)
    startup.mkdir(parents=True, exist_ok=True)
    link = home / "link"
    if not _junction(link, startup):
        pytest.skip("cannot create a junction here")
    out = _write("link/x.cmd")
    _refused(out)
    assert not (startup / "x.cmd").exists()


# --- FAIL-1: trailing dot/space on a shape whose ANCESTOR is not a shape ------
#
# Windows strips a trailing dot/space from every component at the filesystem
# API, but `Path.resolve()` canonicalises a component only when it already
# EXISTS. The creation case is a component that does not exist yet, so
# `PowerShell.` stays `PowerShell.` and never matched the shape `PowerShell`.
# The earlier `Startup./x.cmd` case was caught by the `%APPDATA%\Microsoft\
# Windows` ANCESTOR shape, so it never exercised a standalone shape.

@pytest.mark.skipif(os.name != "nt", reason="Windows strips trailing dots/spaces")
@pytest.mark.parametrize("rel", [
    ".gitconfig.",
    ".gitconfig ",                        # trailing SPACE, not dot
    "_gitconfig.",
    ".bashrc.",
    ".profile ",
    ".zshrc.",
    ".config/git/config.",
    ".config/fish/config.fish.",
    ".config/autostart./x.desktop",
    "Documents/PowerShell./Microsoft.PowerShell_profile.ps1",
    "Documents/WindowsPowerShell./Microsoft.PowerShell_profile.ps1",
])
def test_a_trailing_dot_or_space_on_a_standalone_shape_is_refused(home, rel):
    out = _write(rel)
    _refused(out)
    assert not (home / rel).exists(), f"{rel} was created"


@pytest.mark.skipif(os.name != "nt", reason="Windows strips trailing dots/spaces")
def test_a_relative_copy_destination_with_a_trailing_dot_is_refused(home):
    """The dir does not exist yet, so `resolve()` keeps `PowerShell.`."""
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"],
                                        "dest": "Documents/PowerShell."},
                         _ctx())
    _refused(out)
    assert not (home / "Documents" / "PowerShell").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows strips trailing dots/spaces")
def test_scheduled_tasks_with_a_trailing_dot_is_refused_through_the_grant(home):
    systemroot = Path(os.environ["SystemRoot"])
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files",
                         {"paths": ["a.txt"],
                          "dest": str(systemroot / "System32" / "Tasks.")},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (systemroot / "System32" / "Tasks").exists()


# --- FAIL-2: a local UNC / admin-share spelling of the same directory ---------
#
# `Path.resolve()` keeps a UNC path as UNC, so `\\localhost\c$\…\Startup` and
# the `\\?\UNC\` spelling never matched a drive-letter anchor — and with
# `allow_absolute_writes=True` they wrote straight into Startup.

def _unc(p: Path) -> str:
    s = str(p)
    return "\\\\localhost\\" + s[0].lower() + "$" + s[2:]


def _unc_reaches(unc: str) -> bool:
    """Can this box actually open the admin share? Skip if not. (Do NOT compare
    `resolve()` forms: a UNC path resolves to itself, not to the drive form.)"""
    try:
        return Path(unc).is_dir()
    except (OSError, ValueError):
        return False


@pytest.mark.skipif(os.name != "nt", reason="admin shares are a Windows property")
def test_a_unc_admin_share_of_startup_is_refused_through_the_grant(home):
    startup = _startup(home)
    startup.mkdir(parents=True, exist_ok=True)
    unc = _unc(startup)
    if not _unc_reaches(unc):
        pytest.skip("the c$ admin share is not reachable from this session")
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": unc},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (startup / "a.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="admin shares are a Windows property")
def test_the_extended_unc_spelling_of_startup_is_refused_through_the_grant(home):
    startup = _startup(home)
    startup.mkdir(parents=True, exist_ok=True)
    unc = _unc(startup)
    if not _unc_reaches(unc):
        pytest.skip("the c$ admin share is not reachable from this session")
    extended = "\\\\?\\UNC\\" + unc[2:]
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": extended},
                         _ctx(allow_absolute_writes=True))
    _refused(out)
    assert not (startup / "a.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="admin shares are a Windows property")
def test_a_unc_admin_share_is_refused_without_the_grant_too(home):
    startup = _startup(home)
    startup.mkdir(parents=True, exist_ok=True)
    unc = _unc(startup)
    if not _unc_reaches(unc):
        pytest.skip("the c$ admin share is not reachable from this session")
    (home / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": unc}, _ctx())
    assert out.startswith("error"), out            # outside the workspace
    assert not (startup / "a.txt").exists()


# --- ODR-6-residual: the credential/state-dir rule and a >=260-char path ------
#
# `_ws_path`/`_long_path` hand back a `\\?\`-prefixed path once it is >=260
# chars, and `_credential_path_reason` matches the path SHAPE — `\\?\C:\...` is
# not `is_relative_to` the unprefixed state dir. `_unlong` is now applied at
# `_read_path`, `_write_path`, `_resolve_image` and the workspace walker.

def _deep_dir(base: Path, leaf: str = "deep") -> Path:
    """A directory whose path is >=300 chars (so `_long_path` prefixes it)."""
    d = base
    while len(str(d / leaf)) < 300:
        d = d / ("d" * 40)
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.skipif(os.name != "nt", reason="the \\\\?\\ prefix is Windows-only")
def test_a_long_state_dir_path_is_refused_by_read_file(home):
    rigma = Path(os.environ["RIGMA_HOME"])
    target = _deep_dir(rigma) / "secret.txt"
    target.write_text("TOP SECRET", encoding="utf-8")
    assert len(str(target)) >= 260
    out = tools.run_tool("read_file", {"path": str(target)},
                         _ctx(allow_absolute_reads=True))
    assert out.startswith("error"), out
    assert "state" in out.lower(), out
    assert "TOP SECRET" not in out


@pytest.mark.skipif(os.name != "nt", reason="the \\\\?\\ prefix is Windows-only")
def test_a_long_state_dir_destination_is_refused_by_copy_files(home):
    rigma = Path(os.environ["RIGMA_HOME"])
    dest = _deep_dir(rigma)
    (home / "a.txt").write_text("x", encoding="utf-8")
    assert len(str(dest)) >= 260
    out = tools.run_tool("copy_files", {"paths": ["a.txt"], "dest": str(dest)},
                         _ctx(allow_absolute_writes=True))
    assert out.startswith("error"), out
    assert "state" in out.lower(), out
    assert not (dest / "a.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="the \\\\?\\ prefix is Windows-only")
def test_a_long_workspace_does_not_walk_into_the_state_dir(tmp_path, monkeypatch):
    ws = _deep_dir(tmp_path / "longws")
    state = ws / ".rigma"
    (state / "sessions").mkdir(parents=True, exist_ok=True)
    (state / "sessions" / "x.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("RIGMA_HOME", str(state))
    ctx = {"workspace": str(ws), "allow_code": True}
    out = tools.run_tool("find_files", {"pattern": "**/*.json"}, ctx)
    assert "x.json" not in out, out
    assert "state" not in out.lower(), out


@pytest.mark.skipif(os.name != "nt", reason="the \\\\?\\ prefix is Windows-only")
def test_a_long_relative_image_in_the_state_dir_is_refused(tmp_path, monkeypatch):
    ws = _deep_dir(tmp_path / "longws")
    state = ws / ".rigma"
    state.mkdir(parents=True, exist_ok=True)
    (state / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    monkeypatch.setenv("RIGMA_HOME", str(state))
    out = tools.run_tool("view_image", {"path": ".rigma/secret.png"},
                         {"workspace": str(ws), "allow_code": True,
                          "has_vision": True})
    assert out.startswith("error"), out
    assert "state" in out.lower(), out


# --- ODR-1b: the FILE a copy/move creates is checked, not only its folder -----

@pytest.mark.parametrize("name", [".bashrc", ".gitconfig", ".profile"])
@pytest.mark.parametrize("tool", ["copy_files", "move_files"])
def test_a_crafted_rc_file_cannot_be_copied_or_moved_into_the_home(home, tool, name):
    # The destination FOLDER is the home workspace itself, which is allowed; the
    # file it would create is the persistence location.
    staged = home / "x" / name
    staged.parent.mkdir()
    staged.write_text("payload", encoding="utf-8")
    out = tools.run_tool(tool, {"paths": [f"x/{name}"], "dest": "."}, _ctx())
    _refused(out)
    assert not (home / name).exists()
    assert staged.is_file()                    # a refused move leaves the source


def test_a_path_that_cannot_be_resolved_is_refused(monkeypatch):
    def unresolvable(self, *a, **k):
        raise OSError("cannot resolve")
    monkeypatch.setattr(Path, "resolve", unresolvable)
    assert "persistence" in tools._persistence_path_reason(Path("anything.txt"))
