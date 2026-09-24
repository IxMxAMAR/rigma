"""R3-1: a JUNCTION inside the workspace is walked through, so the walkers read
outside the workspace.

`os.walk(followlinks=False)` and `Path.is_symlink()` both treat a Windows
junction as an ordinary directory — a junction is a MOUNT-POINT reparse point,
not a name-surrogate one, so `is_symlink()` is False for it (measured on this
box). Junctions need no privilege (`mklink /J`), unlike symlinks, so this is the
reachable form of the escape: `find_files` names outside files, `grep` returns
their CONTENT, `watch.Watcher` records them for `undo_last_change`, and
`workspace.pack_folder` packs them into the prompt.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from rigma import tools, watch, workspace

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="junctions are a Windows reparse point")


def _junction(link: Path, target: Path) -> bool:
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True, text=True)
    return r.returncode == 0 and link.exists()


@pytest.fixture
def esc(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    out = tmp_path / "outside"
    out.mkdir()
    (out / "secret.txt").write_text("TOPSECRET-KEY\n", encoding="utf-8")
    (ws / "inside.txt").write_text("nothing\n", encoding="utf-8")
    link = ws / "link"
    if not _junction(link, out):
        pytest.skip("cannot create a junction here")
    return ws, out, link


def test_a_junction_is_not_reported_as_a_symlink(esc):
    """The assumption the walkers were built on, asserted rather than assumed."""
    _ws, _out, link = esc
    assert Path(link).is_symlink() is False


def test_grep_does_not_read_through_a_junction(esc):
    ws, _out, _link = esc
    ctx = {"workspace": str(ws), "profile": "confined"}
    out = tools.run_tool("grep", {"pattern": "TOPSECRET"}, ctx)
    assert "TOPSECRET" not in out, out


def test_find_files_does_not_list_through_a_junction(esc):
    ws, _out, _link = esc
    ctx = {"workspace": str(ws), "profile": "confined"}
    out = tools.run_tool("find_files", {"pattern": "**/*.txt"}, ctx)
    assert "secret.txt" not in out, out


def test_the_watcher_does_not_track_through_a_junction(esc):
    ws, _out, _link = esc
    w = watch.Watcher(ws, record=lambda p, d: True)
    w.poll_once()
    assert not [k for k in w._known if "secret.txt" in k], list(w._known)


def test_undo_cannot_write_outside_through_a_junction(esc, tmp_path,
                                                      monkeypatch):
    """The end-to-end consequence: the watcher records the junction path (which
    is lexically inside the workspace), `_newest_undo_key` accepts it, and the
    restore writes through the junction to the outside file."""
    ws, out, _link = esc
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    target = out / "secret.txt"
    w = watch.Watcher(ws)                 # real undo store, real record
    w.poll_once()
    target.write_text("CHANGED-BY-THE-ARM\n", encoding="utf-8")
    w.poll_once()
    tools.run_tool("undo_last_change", {},
                   {"workspace": str(ws), "allow_code": True, "profile": "all"})
    assert target.read_text(encoding="utf-8") == "CHANGED-BY-THE-ARM\n"


def test_pack_folder_does_not_pack_through_a_junction(esc):
    ws, _out, _link = esc
    packed = workspace.pack_folder(str(ws))
    assert "TOPSECRET" not in packed["content"], packed["content"][:400]
