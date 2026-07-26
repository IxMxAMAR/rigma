"""Merged from the dev branch 2026-07-22: argument normalisation, tool-name
aliasing, the widened rescue parser, CRLF-preserving edits and the write lock.

The theme is the same one the file tools already follow: a weak local model
that gets a NAME or a SHAPE slightly wrong should be steered, not failed —
but never so eagerly that prose about a call becomes a call.
"""
import threading

import pytest

from rigma import tools


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rhome"))
    return tmp_path


def _ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return {"workspace": str(ws), "allow_code": True}, ws


# --- tool-name aliases --------------------------------------------------------

@pytest.mark.parametrize("alias,real", [
    ("read", "read_file"), ("edit", "edit_file"), ("write", "write_file"),
    ("ls", "list_directory"), ("dir", "list_directory"),
    ("find", "find_files"), ("bash", "run_shell"), ("sh", "run_shell"),
    ("python", "run_python"),
])
def test_other_harnesses_tool_names_resolve(alias, real):
    # difflib can't reach these: "read" vs "read_file" scores 0.62, under its
    # 0.7 cutoff, so before the alias table every one of these burned a turn
    assert tools.resolve_tool_name(alias) == real


def test_aliases_are_not_advertised_as_separate_tools():
    # THE reason aliases live in resolve_tool_name and not in _REGISTRY:
    # specs() iterates the registry, so a registered alias would ship the
    # model a second, identical tool and charge context for it.
    names = [s["function"]["name"] for s in tools.tool_specs(
        allow_code=True, workspace=True, has_rag=True, has_vision=True,
        has_run=True)]
    assert len(names) == len(set(names)), "duplicate tool on the wire"
    for alias in ("read", "edit", "write", "ls", "bash", "python"):
        assert alias not in names


def test_alias_actually_runs_the_real_tool(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "a.txt").write_text("hello\n", encoding="utf-8")
    assert "hello" in tools.run_tool("read", {"path": "a.txt"}, ctx)


# --- argument normalisation ---------------------------------------------------

@pytest.mark.parametrize("tool,given,want", [
    ("read_file", {"file": "a.md"}, "path"),
    ("read_file", {"filepath": "a.md"}, "path"),
    ("read_file", {"file_path": "a.md"}, "path"),
    ("run_shell", {"cmd": "dir"}, "command"),
    ("run_python", {"source": "print(1)"}, "code"),
    ("find_files", {"glob": "**/*.py"}, "pattern"),
    ("edit_file", {"old_string": "a", "new_string": "b", "path": "x"}, "old"),
])
def test_alias_fills_the_declared_parameter(tool, given, want):
    assert tools.normalize_tool_args(tool, given).get(want) is not None


def test_a_declared_parameter_is_never_treated_as_an_alias():
    # grep takes BOTH `pattern` (the regex) and `glob` (which files). Copying
    # glob into pattern would search for the file filter as if it were the
    # regex — a silently wrong result, the worst kind.
    out = tools.normalize_tool_args("grep", {"glob": "*.py"})
    assert "pattern" not in out
    assert out["glob"] == "*.py"


def test_single_unnamed_value_becomes_the_one_required_argument():
    assert tools.normalize_tool_args(
        "read_file", {"input": "notes.md"}) == {"path": "notes.md"}


def test_normalization_leaves_correct_args_alone():
    good = {"path": "a.md", "offset": 3}
    assert tools.normalize_tool_args("read_file", dict(good)) == good


def test_unknown_tool_args_pass_through_untouched():
    given = {"whatever": 1}
    assert tools.normalize_tool_args("not_a_tool", dict(given)) == given


def test_a_lone_line_number_becomes_a_range():
    out = tools.normalize_tool_args("edit_file",
                                    {"path": "a", "new": "b", "line": 7})
    assert out["start_line"] == 7 and out["end_line"] == 7


# --- the rescue parser --------------------------------------------------------

def test_rescue_reads_a_fenced_json_call():
    name, args = tools.rescue_tool_call(
        'here you go:\n```json\n'
        '{"name": "read_file", "arguments": {"path": "a.md"}}\n```')
    assert (name, args) == ("read_file", {"path": "a.md"})


def test_rescue_reads_a_bare_json_object_reply():
    name, args = tools.rescue_tool_call(
        '{"function": {"name": "list_directory", "arguments": {"path": "."}}}')
    assert (name, args) == ("list_directory", {"path": "."})


def test_rescue_reads_the_tool_kwargs_shape():
    name, args = tools.rescue_tool_call(
        '{"tool": "run_python", "kwargs": {"code": "print(1)"}}')
    assert (name, args) == ("run_python", {"code": "print(1)"})


def test_rescue_infers_the_tool_from_an_unambiguous_shape():
    assert tools.rescue_tool_call(
        '{"path": "a.md", "content": "hi"}')[0] == "write_file"
    assert tools.rescue_tool_call(
        '{"path": "a.md", "old": "x", "new": "y"}')[0] == "edit_file"
    assert tools.rescue_tool_call('{"command": "dir"}')[0] == "run_shell"


def test_rescue_repairs_a_stringified_arguments_field():
    name, args = tools.rescue_tool_call(
        '{"name": "read_file", "arguments": "{\\"path\\": \\"a.md\\"}"}')
    assert (name, args) == ("read_file", {"path": "a.md"})


def test_rescue_maps_a_near_miss_name_inside_json():
    assert tools.rescue_tool_call(
        '{"name": "ReadFile", "arguments": {"path": "a.md"}}')[0] == "read_file"


def test_rescue_reads_the_react_shape():
    name, args = tools.rescue_tool_call(
        'Thought: I should look.\nAction: list_directory\n'
        'Action Input: {"path": "."}')
    assert (name, args) == ("list_directory", {"path": "."})


# THE guard. Widening the parser to JSON is only safe while "a reply that IS a
# call" stays distinguishable from "a reply ABOUT a call" — otherwise
# explaining write_file performs a write nobody asked for.
@pytest.mark.parametrize("prose", [
    'You would pass {"path": "a.txt", "content": "hi"} to write it.',
    'The argument object looks like {"command": "dir"} in that case.',
    'Set {"path": "notes.md", "old": "a", "new": "b"} and it replaces the line.',
    'For example: {"name": "write_file", "arguments": {"path": "x",'
    ' "content": "y"}} — but I will not run it.',
    "I will call list_directory now.",
    "the function=thing syntax is documented",
    "", None,
])
def test_rescue_never_fires_on_prose_that_merely_mentions_a_call(prose):
    assert tools.rescue_tool_call(prose) == (None, None)


def test_the_old_rescue_name_still_works():
    assert tools.rescue_xml_tool_call is tools.rescue_tool_call


# --- repair_json_args ---------------------------------------------------------

def test_python_repr_arguments_are_repaired():
    args, note = tools.repair_json_args("{'path': 'a.md', 'numbered': True}")
    assert args == {"path": "a.md", "numbered": True}
    assert "Python" in note


# --- edit_file: line endings and the write lock -------------------------------

def test_edit_preserves_crlf_line_endings(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "crlf.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    out = tools.run_tool("edit_file", {"path": "crlf.txt", "old": "two",
                                       "new": "TWO"}, ctx)
    assert not out.startswith("error"), out
    raw = (ws / "crlf.txt").read_bytes()
    assert b"TWO" in raw
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")


def test_edit_does_not_convert_an_lf_file_to_crlf(tmp_path):
    # write_text() translates "\n" to os.linesep, so on Windows editing one
    # line of an LF file silently rewrote EVERY line ending in it — a one-word
    # change arriving as a whole-file diff.
    ctx, ws = _ctx(tmp_path)
    (ws / "lf.txt").write_bytes(b"one\ntwo\nthree\n")
    tools.run_tool("edit_file", {"path": "lf.txt", "old": "two",
                                 "new": "TWO"}, ctx)
    assert (ws / "lf.txt").read_bytes() == b"one\nTWO\nthree\n"


def test_edit_matches_old_text_quoted_with_crlf(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "lf.txt").write_bytes(b"alpha\nbeta\ngamma\n")
    out = tools.run_tool("edit_file", {
        "path": "lf.txt", "old": "alpha\r\nbeta", "new": "ALPHA\nBETA"}, ctx)
    assert not out.startswith("error"), out
    assert (ws / "lf.txt").read_bytes() == b"ALPHA\nBETA\ngamma\n"


def test_line_range_edit_preserves_crlf(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "crlf.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    out = tools.run_tool("edit_file", {"path": "crlf.txt", "start_line": 2,
                                       "end_line": 2, "new": "TWO"}, ctx)
    assert not out.startswith("error"), out
    assert (ws / "crlf.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_concurrent_edits_do_not_lose_a_write(tmp_path):
    # read -> compare -> write is not atomic. Unlocked, two turns editing the
    # same file both read the ORIGINAL and the second write drops the first
    # edit, with no error anywhere to notice it by.
    ctx, ws = _ctx(tmp_path)
    (ws / "shared.txt").write_text("A\nB\n", encoding="utf-8")
    errors = []

    def _edit(old, new):
        try:
            out = tools.run_tool("edit_file", {"path": "shared.txt",
                                               "old": old, "new": new}, ctx)
            if out.startswith("error"):
                errors.append(out)
        except Exception as e:            # pragma: no cover - surfaced below
            errors.append(repr(e))

    threads = [threading.Thread(target=_edit, args=("A", "AA")),
               threading.Thread(target=_edit, args=("B", "BB"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, errors
    text = (ws / "shared.txt").read_text(encoding="utf-8")
    assert "AA" in text and "BB" in text, f"an edit was lost: {text!r}"


def test_file_lock_is_reentrant():
    # a nested acquire on one thread must not deadlock
    with tools._FILE_LOCK:
        with tools._FILE_LOCK:
            assert True


# --- Windows long paths -------------------------------------------------------

def test_long_path_gets_the_extended_prefix():
    long = "C:\\" + "\\".join(["a" * 40] * 8)      # comfortably over 260
    got = str(tools._long_path(tools.Path(long)))
    if tools.os.name == "nt":
        assert got.startswith("\\\\?\\")
        assert got.endswith(long[3:])              # path itself is unchanged
    else:
        assert got == long


def test_short_paths_are_left_alone():
    p = tools.Path("C:\\ws\\a.txt")
    assert tools._long_path(p) == p


def test_unc_long_path_uses_the_unc_form():
    if tools.os.name != "nt":
        pytest.skip("windows-only path form")
    long_unc = "\\\\srv\\share\\" + "\\".join(["b" * 40] * 8)
    assert str(tools._long_path(tools.Path(long_unc))).startswith("\\\\?\\UNC\\")
