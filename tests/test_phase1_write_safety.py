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


# --- punctuation drift (live 2026-07-21: repeated edit failures on fiction) ---
def test_edit_heals_curly_quote_drift(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "story.txt").write_text(
        "She said, “It’s over — forever.”\nHe left.\n",
        encoding="utf-8")
    # the model retypes the passage with ASCII quotes and hyphen
    out = tools.run_tool("edit_file", {
        "path": "story.txt",
        "old": 'She said, "It\'s over - forever."',
        "new": 'She whispered, "It’s over — forever."'}, ctx)
    assert not out.startswith("error"), out
    text = (ws / "story.txt").read_text(encoding="utf-8")
    assert "whispered" in text
    assert "He left." in text                    # rest untouched


def test_edit_heals_combined_punct_and_whitespace_drift(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "index.txt").write_text(
        "Chapter 5 — “The Fall”\n   status: done\n",
        encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "index.txt",
        "old": 'Chapter 5 - "The Fall"\nstatus: done',
        "new": "Chapter 5 — “The Fall”\n   status: revised"}, ctx)
    assert not out.startswith("error"), out
    assert "revised" in (ws / "index.txt").read_text(encoding="utf-8")


def test_edit_punct_healing_never_matches_wrong_place(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "a.txt").write_text("alpha — one\nalpha — one\n",
                              encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "a.txt", "old": "alpha - one", "new": "x"}, ctx)
    assert out.startswith("error")               # ambiguous stays ambiguous


def test_edit_large_block_flexible_match(tmp_path):
    ctx, ws = _ctx(tmp_path)
    body = "\n".join(f"line {i} of the long spec — detail"
                     for i in range(600))
    (ws / "spec.txt").write_text(body, encoding="utf-8")
    # a ~600-token old with drifted dashes: the old 400-token cap skipped
    # flexible matching entirely for blocks like this
    old = "\n".join(f"line {i} of the long spec - detail"
                    for i in range(100, 200))
    out = tools.run_tool("edit_file", {
        "path": "spec.txt", "old": old, "new": "REPLACED BLOCK"}, ctx)
    assert not out.startswith("error"), out[:200]
    assert "REPLACED BLOCK" in (ws / "spec.txt").read_text(encoding="utf-8")


# --- markdown decoration drift (live 2026-07-21, round 2: index in .md) -------
def test_edit_heals_dropped_bold_marks(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "spec.md").write_text(
        "## STATUS\n\n**Chapters written:** 1–6 + side-bridge\n",
        encoding="utf-8")
    # model quotes the line without the ** and with an ascii dash
    out = tools.run_tool("edit_file", {
        "path": "spec.md",
        "old": "Chapters written: 1-6 + side-bridge",
        "new": "**Chapters written:** 1–7"}, ctx)
    assert not out.startswith("error"), out
    text = (ws / "spec.md").read_text(encoding="utf-8")
    assert "1–7" in text and "side-bridge" not in text
    assert "## STATUS" in text                    # rest untouched


def test_edit_heals_dropped_italics_and_ellipsis(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "s.md").write_text(
        'Chapter 6 ends with: *"Tomorrow, we’ll go to the tank…"*\n',
        encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "s.md",
        "old": 'Chapter 6 ends with: "Tomorrow, we\'ll go to the tank..."',
        "new": 'Chapter 6 ends with: *"The tank waits…"*'}, ctx)
    assert not out.startswith("error"), out
    assert "The tank waits" in (ws / "s.md").read_text(encoding="utf-8")


def test_edit_heals_trailing_hardbreak_spaces(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "h.md").write_text("line one  \nline two\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "h.md", "old": "line one\nline two",
        "new": "line one  \nline 2"}, ctx)
    assert not out.startswith("error"), out
    assert "line 2" in (ws / "h.md").read_text(encoding="utf-8")


def test_markdown_healing_still_refuses_ambiguity(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "amb.md").write_text(
        "**alpha beta**\nmid\n*alpha beta*\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "amb.md", "old": "alpha beta", "new": "x"}, ctx)
    assert out.startswith("error")               # 2 normalised matches


def test_exact_star_edits_in_code_still_exact(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "m.py").write_text("x = a * b\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "m.py", "old": "a * b", "new": "a * b * c"}, ctx)
    assert out.startswith("edited")              # exact path, no healing
    assert (ws / "m.py").read_text(encoding="utf-8") == "x = a * b * c\n"
