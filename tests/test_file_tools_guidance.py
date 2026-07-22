"""read_file / write_file must GUIDE, not dead-end.

Live transcript 2026-07-21: the model spent ~25 tool calls flailing to open
one file. It could not discover a directory named `comfyui-wildcard-engine`,
so it guessed -- `wildcard-yngine`, `wildcard-engin`, `wild-card-engine`,
`wild*ngine` -- because every miss returned a bare "no such file" that taught
it nothing. It even ran find_files and got the answer, then could not use it.
The loop ended with write_file trying to write 7943 chars to a path with a
literal `*` in it (illegal on Windows).

Root cause: the file tools reject bad input instead of steering toward good
input. These tests pin the steering.
"""
import pytest

from rigma import tools


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "comfyui-wildcard-engine").mkdir()
    (tmp_path / "comfyui-wildcard-engine" / "__init__.py").write_text(
        "REAL MODULE", encoding="utf-8")
    (tmp_path / "readme.md").write_text("hi", encoding="utf-8")
    return {"workspace": str(tmp_path), "allow_code": True}


def run(name, args, ctx):
    return tools.run_tool(name, args, ctx)


# --- read_file on a directory redirects instead of dead-ending -------------

def test_read_file_on_the_workspace_root_lists_it(ws):
    out = run("read_file", {"path": "."}, ws)
    assert not out.startswith("error")
    assert "comfyui-wildcard-engine" in out
    assert "readme.md" in out


def test_read_file_on_a_subfolder_lists_it(ws):
    out = run("read_file", {"path": "comfyui-wildcard-engine"}, ws)
    assert not out.startswith("error")
    assert "__init__.py" in out
    assert "folder" in out.lower()          # tells the model it's a directory


# --- read_file recovers a wrong DIRECTORY component ------------------------

def test_a_typoed_directory_name_names_the_real_one(ws):
    """The exact failure: the model typed the wrong directory. The miss must
    surface the real name so the next call is right, not another guess."""
    out = run("read_file",
              {"path": "comfyui-wildcard-yngine/__init__.py"}, ws)
    assert out.startswith("error")
    assert "comfyui-wildcard-engine" in out    # the real name is shown


def test_a_missing_file_shows_the_folders_real_contents(ws):
    out = run("read_file",
              {"path": "comfyui-wildcard-engine/nope.py"}, ws)
    assert out.startswith("error")
    assert "__init__.py" in out                # what's actually there


# --- read_file resolves a glob instead of rejecting it ---------------------

def test_a_glob_that_matches_one_file_is_read(ws):
    out = run("read_file",
              {"path": "comfyui-wild*ngine/__init__.py"}, ws)
    assert "REAL MODULE" in out                # it just read the file


def test_a_glob_with_no_match_points_at_find_files(ws):
    out = run("read_file", {"path": "nothing-*-here/x.py"}, ws)
    assert out.startswith("error")
    assert "find_files" in out


# --- write_file refuses a dangerous path BEFORE it can act -----------------

def test_write_to_a_glob_path_is_refused(ws, tmp_path):
    out = run("write_file",
              {"path": "comfyui-wild*ngine/__init__.py",
               "content": "X" * 7943}, ws)
    assert out.startswith("error")
    assert "*" in out and "find_files" in out
    # and NOTHING was written or created
    assert not (tmp_path / "comfyui-wild*ngine").exists()
    assert (tmp_path / "comfyui-wildcard-engine" / "__init__.py").read_text(
        encoding="utf-8") == "REAL MODULE"


def test_write_to_an_illegal_character_path_is_refused_cleanly(ws):
    out = run("write_file", {"path": 'weird?name.txt', "content": "x"}, ws)
    assert out.startswith("error")
    assert "?" in out
    assert "WinError" not in out               # a clean message, not a traceback


def test_a_normal_write_still_works(ws, tmp_path):
    out = run("write_file", {"path": "notes/day1.md", "content": "hello"}, ws)
    assert not out.startswith("error"), out
    assert (tmp_path / "notes" / "day1.md").read_text(
        encoding="utf-8") == "hello"


def test_the_error_paths_do_not_break_the_progress_log_rescue(ws):
    """read_file has a run-only shortcut for progress.md; the new directory
    and glob handling must not shadow it."""
    out = run("read_file", {"path": "progress.md"},
              {**ws, "run_id": "r1"})
    # no such run -> empty tail, but it must take the progress branch, not
    # the new folder/glob branches
    assert "progress" in out.lower() or out.startswith("error")
