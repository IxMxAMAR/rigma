"""edit_file gets a deterministic path that needs no verbatim quoting.

Live owner report 2026-07-21: "edit file still has problems, it's almost
never used correctly in 1 go". Their own history shows 3 of 4 edit_file calls
failing with "the 'old' string wasn't found EXACTLY".

Root cause: read_file returns raw text with no line numbers, so the only
handle edit_file offers is a verbatim copy of a prose block -- and the model
lightly REWRITES words while quoting (the same autopsy that motivated fuzzy
matching earlier the same day). String matching cannot heal a reworded quote.

So give it a handle that needs no quoting at all: read_file(numbered=true) to
see line numbers, then edit_file(start_line=, end_line=) to replace exactly
those lines. Both are opt-in -- default output is unchanged, because line
numbers leaking into prose the model re-emits would be its own bug.
"""
import pytest

from rigma import tools


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text(
        "alpha\nbravo\ncharlie\ndelta\necho\n", encoding="utf-8")
    return {"workspace": str(tmp_path), "allow_code": True}


def test_read_file_is_unnumbered_by_default(ws):
    out = tools.run_tool("read_file", {"path": "a.txt"}, ws)
    assert out.startswith("alpha")
    assert "1|" not in out


def test_read_file_numbered_on_request(ws):
    out = tools.run_tool("read_file", {"path": "a.txt", "numbered": True}, ws)
    assert "1|alpha" in out.replace(" ", "")
    assert "3|charlie" in out.replace(" ", "")


def test_numbering_respects_offset(ws):
    out = tools.run_tool(
        "read_file", {"path": "a.txt", "numbered": True, "offset": 3}, ws)
    flat = out.replace(" ", "")
    assert "3|charlie" in flat and "1|alpha" not in flat


def test_edit_by_line_range_replaces_exactly_those_lines(ws, tmp_path):
    out = tools.run_tool("edit_file", {
        "path": "a.txt", "start_line": 2, "end_line": 3,
        "new": "BRAVO\nCHARLIE"}, ws)
    assert not out.startswith("error"), out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == (
        "alpha\nBRAVO\nCHARLIE\ndelta\necho\n")


def test_edit_by_single_line(ws, tmp_path):
    tools.run_tool("edit_file", {"path": "a.txt", "start_line": 1,
                                 "end_line": 1, "new": "ALPHA"}, ws)
    assert (tmp_path / "a.txt").read_text(
        encoding="utf-8").startswith("ALPHA\nbravo")


def test_line_range_can_delete_by_writing_nothing(ws, tmp_path):
    tools.run_tool("edit_file", {"path": "a.txt", "start_line": 2,
                                 "end_line": 2, "new": ""}, ws)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == (
        "alpha\n\ncharlie\ndelta\necho\n")


def test_out_of_range_lines_are_refused_with_the_real_count(ws, tmp_path):
    before = (tmp_path / "a.txt").read_text(encoding="utf-8")
    out = tools.run_tool("edit_file", {"path": "a.txt", "start_line": 9,
                                       "end_line": 12, "new": "x"}, ws)
    assert out.startswith("error") and "5 lines" in out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == before


def test_reversed_range_is_refused(ws):
    out = tools.run_tool("edit_file", {"path": "a.txt", "start_line": 4,
                                       "end_line": 2, "new": "x"}, ws)
    assert out.startswith("error")


def test_a_line_edit_is_undoable_like_any_other(ws, tmp_path):
    tools.run_tool("edit_file", {"path": "a.txt", "start_line": 1,
                                 "end_line": 1, "new": "ALPHA"}, ws)
    tools.run_tool("undo_last_change", {}, ws)
    assert (tmp_path / "a.txt").read_text(
        encoding="utf-8").startswith("alpha\nbravo")


def test_pasted_line_numbers_are_stripped_from_old(ws, tmp_path):
    """If the model read with numbered=true it will paste the numbers back
    into `old`. Strip them rather than failing on a mismatch it cannot see."""
    out = tools.run_tool("edit_file", {
        "path": "a.txt", "old": "  2|bravo\n  3|charlie", "new": "B\nC"}, ws)
    assert not out.startswith("error"), out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == (
        "alpha\nB\nC\ndelta\necho\n")


def test_the_failure_message_offers_the_line_range_route(ws):
    out = tools.run_tool("edit_file", {
        "path": "a.txt", "old": "nothing like this exists in the file",
        "new": "x"}, ws)
    assert out.startswith("error")
    assert "start_line" in out


def test_old_and_line_range_together_is_refused(ws):
    out = tools.run_tool("edit_file", {
        "path": "a.txt", "old": "alpha", "start_line": 1, "end_line": 1,
        "new": "x"}, ws)
    assert out.startswith("error")


def test_exact_old_matching_still_works(ws, tmp_path):
    out = tools.run_tool("edit_file", {"path": "a.txt", "old": "charlie",
                                       "new": "CHARLIE"}, ws)
    assert not out.startswith("error"), out
    assert "CHARLIE" in (tmp_path / "a.txt").read_text(encoding="utf-8")
