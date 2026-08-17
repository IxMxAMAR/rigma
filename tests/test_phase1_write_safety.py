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
    # a genuine miss: below _FUZZY_ACCEPT, so nothing is written and the model
    # is shown what the file really says
    out = tools.run_tool("edit_file", {
        "path": "b.txt", "old": "the quick red dog walks",
        "new": "x"}, ctx)
    assert out.startswith("error")
    assert "closest" in out and "region" in out
    assert "quick brown fox" in out               # the actual file text
    assert "2:" in out                            # with line numbers
    assert "fox" in (ws / "b.txt").read_text(encoding="utf-8")   # untouched


# --- the _FUZZY_ACCEPT bar itself --------------------------------------------
# 2026-07-22 dropped it 0.85 -> 0.75. These pin what that actually changed, so
# the bar can't drift again without a test saying so.

def test_edit_accepts_one_swapped_word_and_says_so(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "b.txt").write_text(
        "alpha\nthe quick brown fox jumps\ngamma\n", encoding="utf-8")
    # one word wrong out of five (~80%) — under the old 0.85 bar this was a
    # hard error; it is exactly the "model reworded while quoting" case the
    # fuzzy path exists for, so now it applies
    out = tools.run_tool("edit_file", {
        "path": "b.txt", "old": "the quick brown wolf jumps",
        "new": "REPLACED"}, ctx)
    assert not out.startswith("error")
    assert "differed slightly" in out          # never silent about guessing
    assert "80%" in out                        # and says how close it was
    assert "undo_last_change" in out           # and how to take it back
    assert "REPLACED" in (ws / "b.txt").read_text(encoding="utf-8")


def test_fuzzy_edit_is_undoable(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "b.txt").write_text(
        "alpha\nthe quick brown fox jumps\ngamma\n", encoding="utf-8")
    tools.run_tool("edit_file", {"path": "b.txt",
                                 "old": "the quick brown wolf jumps",
                                 "new": "REPLACED"}, ctx)
    tools.run_tool("undo_last_change", {"path": "b.txt"}, ctx)
    assert "quick brown fox" in (ws / "b.txt").read_text(encoding="utf-8")


def test_fuzzy_refuses_when_two_regions_are_equally_close(tmp_path):
    ctx, ws = _ctx(tmp_path)
    # the MARGIN, not the threshold, is the safety property: with the accept
    # bar this low, "beat every other candidate" is what stops the wrong
    # paragraph being rewritten
    before = ("the quick brown fox jumps over\n\n"
              "the quick brown cat jumps over\n")
    (ws / "d.txt").write_text(before, encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "d.txt", "old": "the quick brown pig jumps over",
        "new": "CLOBBERED"}, ctx)
    assert out.startswith("error")
    assert (ws / "d.txt").read_text(encoding="utf-8") == before


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


# --- lexical drift (live autopsy 2026-07-21: 93% word overlap, 0 matches) -----
def test_edit_heals_slight_word_drift(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "w.txt").write_text(
        "She walked to the ancient tank at dawn, counting her breaths.\n"
        "The city slept behind her, unaware of the morning's weight.\n"
        "A letter waited in her pocket, unsent for three years now.\n",
        encoding="utf-8")
    # one word swapped ('old' for 'ancient'), one dropped ('now')
    out = tools.run_tool("edit_file", {
        "path": "w.txt",
        "old": "She walked to the old tank at dawn, counting her breaths.\n"
               "The city slept behind her, unaware of the morning's weight.\n"
               "A letter waited in her pocket, unsent for three years.",
        "new": "REPLACED PASSAGE"}, ctx)
    assert not out.startswith("error"), out
    assert "similarity" in out                     # correction is loud
    assert "undo_last_change" in out               # escape hatch is named
    assert (ws / "w.txt").read_text(encoding="utf-8").strip() \
        == "REPLACED PASSAGE"


def test_fuzzy_never_accepts_half_imagined_text(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "x.txt").write_text(
        "Chapter seven begins at the river crossing before sunrise.\n"
        "Ananya counts the boats and finds one missing from the line.\n",
        encoding="utf-8")
    # ~half the words are invented — must refuse (below 50% nothing in the
    # file is a meaningful "closest region"; the honest answer is re-read)
    before = (ws / "x.txt").read_text(encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "x.txt",
        "old": "Chapter seven begins with the storm breaking over the "
               "temple gates while soldiers gather in the courtyard",
        "new": "x"}, ctx)
    assert out.startswith("error")
    assert "read_file" in out                      # points at the recovery
    assert (ws / "x.txt").read_text(encoding="utf-8") == before


def test_fuzzy_refuses_ambiguous_near_twins(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "t.txt").write_text(
        "The guard walked the north wall at midnight tonight.\n"
        "unrelated middle line here\n"
        "The guard walked the south wall at midnight tonight.\n",
        encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "t.txt",
        "old": "The guard walked the east wall at midnight tonight.",
        "new": "x"}, ctx)
    assert out.startswith("error")                 # two candidates too close


# --- short quotes: the case that produced a bare error with no help ----------
# Live 2026-08-18 (owner screenshot): three consecutive edit_file failures on a
# character sheet, each returning the bare "wasn't found EXACTLY" with NO
# region hint. Measured against the real file afterwards: an `old` of 2-3 words
# with ONE word drifted misses every rung — _flexible_find needs 3+ normalised
# chars, _fuzzy_region needs 3+ words AND 0.75 similarity (a 3-word quote with
# one word wrong scores ~0.67), and _nearest_region compares the probe against
# a WHOLE line with SequenceMatcher.ratio(), which is dominated by the length
# difference. 37 of 37 dash-bearing lines produced "NOTHING". A character sheet
# is made of short fields, so this is the common case, not the exotic one.

def test_a_short_drifted_quote_still_gets_a_region_hint(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "sheet.txt").write_text(
        "Name: Seraphine Vale\n"
        "Age: twenty-four winters\n"
        "Focus: the silver thread\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "sheet.txt", "old": "Age: thirty",   # 2 words, one drifted
        "new": "Age: thirty"}, ctx)
    assert out.startswith("error")
    assert "closest" in out and "region" in out, \
        "a short quote must still be told where to look"
    assert "twenty-four" in out                   # the file's real text
    assert "2:" in out                            # with a line number
    # and nothing was written on a guess
    assert "thirty" not in (ws / "sheet.txt").read_text(encoding="utf-8")


def test_a_short_quote_hint_does_not_fire_on_nothing(tmp_path):
    """A hint pointing at an unrelated line is worse than no hint — it sends
    the model to rewrite the wrong place. Genuinely absent text gets the plain
    error and the line-range advice."""
    ctx, ws = _ctx(tmp_path)
    (ws / "sheet.txt").write_text(
        "Name: Seraphine Vale\nAge: twenty-four winters\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "sheet.txt", "old": "Quartermaster requisition form",
        "new": "x"}, ctx)
    assert out.startswith("error")
    assert "closest" not in out
    assert "start_line" in out          # still steered to the route that works


def test_a_single_word_miss_gets_a_hint_too(tmp_path):
    ctx, ws = _ctx(tmp_path)
    (ws / "sheet.txt").write_text(
        "Name: Seraphine Vale\nAge: twenty-four winters\n", encoding="utf-8")
    out = tools.run_tool("edit_file", {
        "path": "sheet.txt", "old": "Seraphina", "new": "Seraphine"}, ctx)
    assert out.startswith("error")
    assert "closest" in out and "Seraphine Vale" in out
