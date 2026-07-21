"""Phase 1 of the field-parity audit: turn economy + write safety.

edit_file heals whitespace drift and shows the near-miss region; write_file
snapshots before replacing; undo_last_change actually recovers a destroyed
draft; fetch_url pages instead of dead-ending.
"""
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


# --- edit_file: self-healing --------------------------------------------------
def test_edit_flexible_whitespace_match(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "a.py").write_text("def f():\n    return   1\n", encoding="utf-8")
    # the model reproduces the line with drifted spacing — this used to be a
    # hard error and a wasted read_file round-trip
    out = tools.run_tool("edit_file", {
        "path": "a.py", "old": "def f():\n  return 1",
        "new": "def f():\n    return 2"}, ctx)
    assert not out.startswith("error")
    assert "flexibly" in out                      # correction is reported
    assert "return 2" in (ws / "a.py").read_text(encoding="utf-8")


def test_edit_miss_shows_nearest_region(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "b.txt").write_text(
        "alpha\nthe quick brown fox jumps\ngamma\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "b.txt", "old": "the quick brown wolf jumps",
        "new": "x"}, ctx)
    assert out.startswith("error")
    assert "closest matching region" in out
    assert "quick brown fox" in out               # the actual file text
    assert "2:" in out                            # with line numbers


def test_edit_multimatch_reports_line_numbers(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "c.txt").write_text("dup\nmid\ndup\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {"path": "c.txt", "old": "dup",
                                       "new": "x"}, ctx)
    assert out.startswith("error")
    assert "lines 1, 3" in out


def test_edit_exact_match_still_works(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "d.txt").write_text("one two three\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {"path": "d.txt", "old": "two",
                                       "new": "2"}, ctx)
    assert out.startswith("edited")
    assert (ws / "d.txt").read_text(encoding="utf-8") == "one 2 three\n"


# --- write safety: snapshot + undo --------------------------------------------
def test_undo_recovers_replaced_file(tmp_path):
    ctx, ws = _ctx(tmp_path)
    long = "the original long draft " * 100
    tools.run_tool("write_file", {"path": "ch.txt", "content": long}, ctx)
    out = tools.run_tool("write_file",
                         {"path": "ch.txt", "content": "oops short"}, ctx)
    assert "REPLACED" in out and "undo_last_change" in out   # recovery named
    undo = tools.run_tool("undo_last_change", {"path": "ch.txt"}, ctx)
    assert undo.startswith("restored")
    assert (ws / "ch.txt").read_text(encoding="utf-8") == long


def test_undo_without_path_restores_most_recent(tmp_path):
    ctx, ws = _ctx(tmp_path)
    tools.run_tool("write_file", {"path": "x.txt", "content": "keep"}, ctx)
    tools.run_tool("write_file", {"path": "x.txt", "content": "bad"}, ctx)
    out = tools.run_tool("undo_last_change", {}, ctx)
    assert out.startswith("restored")
    assert (ws / "x.txt").read_text(encoding="utf-8") == "keep"


def test_undo_twice_swaps_back(tmp_path):
    ctx, ws = _ctx(tmp_path)
    tools.run_tool("write_file", {"path": "y.txt", "content": "v1"}, ctx)
    tools.run_tool("write_file", {"path": "y.txt", "content": "v2"}, ctx)
    tools.run_tool("undo_last_change", {"path": "y.txt"}, ctx)
    assert (ws / "y.txt").read_text(encoding="utf-8") == "v1"
    tools.run_tool("undo_last_change", {"path": "y.txt"}, ctx)
    assert (ws / "y.txt").read_text(encoding="utf-8") == "v2"


def test_undo_with_nothing_recorded(tmp_path):
    ctx, _ = _ctx(tmp_path)
    out = tools.run_tool("undo_last_change", {}, ctx)
    assert out.startswith("error") and "nothing to undo" in out


def test_edit_file_is_undoable(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "e.txt").write_text("good text here\n", encoding="utf-8")
    tools.run_tool("edit_file", {"path": "e.txt", "old": "good",
                                 "new": "bad"}, ctx)
    assert "bad" in (ws / "e.txt").read_text(encoding="utf-8")
    out = tools.run_tool("undo_last_change", {"path": "e.txt"}, ctx)
    assert out.startswith("restored")
    assert (ws / "e.txt").read_text(encoding="utf-8") == "good text here\n"


def test_undo_gated_behind_code(tmp_path):
    names = {t["function"]["name"] for t in tools.tool_specs()}
    assert "undo_last_change" not in names
    with_code = {t["function"]["name"]
                 for t in tools.tool_specs(allow_code=True)}
    assert "undo_last_change" in with_code


# --- fetch_url paging ---------------------------------------------------------
def test_fetch_url_pages_long_text(monkeypatch):
    long_html = "<html><body>" + "word " * 4000 + "</body></html>"
    monkeypatch.setattr(tools, "_bounded_get",
                        lambda url, **k: (200, long_html))
    out = tools.run_tool("fetch_url", {"url": "http://x.test/page"}, {})
    assert "call fetch_url again with offset=" in out
    # follow the instruction it gave
    import re as _re
    off = int(_re.search(r"offset=(\d+)", out).group(1))
    out2 = tools.run_tool("fetch_url",
                          {"url": "http://x.test/page", "offset": off}, {})
    assert f"chars {off + 1}-" in out2


def test_fetch_url_short_page_has_no_noise(monkeypatch):
    monkeypatch.setattr(tools, "_bounded_get",
                        lambda url, **k: (200, "<p>tiny page</p>"))
    out = tools.run_tool("fetch_url", {"url": "http://x.test/small"}, {})
    assert out.strip() == "tiny page"
