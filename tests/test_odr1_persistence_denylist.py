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
