"""The same file must not be sent to the model twice in one turn.

Live 2026-08-21: a model unsure of a spelling issued two read_file calls in the
SAME turn — "Chapter_02_Shubashini.txt" and "Chapter_02_Shubhashini.txt". Only
the second exists; _fuzzy_file resolved the first to it. Both returned the full
23,068-byte file, so ~5,800 tokens — 18% of a 32K window — were spent receiving
the same content twice.

The near-miss note ("you asked for X — used Y") was already there and did not
help: parallel tool calls are emitted before ANY result comes back, so the model
could not have known.

Re-reading must still work when it means something: after a write, at a
different offset, or in a later turn.
"""
from rigma import tools


def _ctx(ws):
    return {"workspace": str(ws), "allow_code": False, "_reads": {}}


def test_the_same_file_twice_in_one_turn_is_not_resent(tmp_path):
    f = tmp_path / "Chapter_02_Shubhashini.txt"
    body = "the witch counted her threads.\n" * 400
    f.write_text(body, encoding="utf-8")
    ctx = _ctx(tmp_path)

    first = tools.run_tool("read_file", {"path": f.name}, ctx)
    assert "counted her threads" in first

    second = tools.run_tool("read_file", {"path": f.name}, ctx)
    assert "counted her threads" not in second, "sent the whole file again"
    assert "already read" in second.lower()
    assert f.name in second, "must name the file it is pointing at"


def test_a_misspelling_that_resolves_to_an_already_read_file_is_caught(tmp_path):
    """The actual case. The typo resolves to the same file, so it is the same
    content — the dedupe has to key on what was RESOLVED, not what was asked."""
    f = tmp_path / "Chapter_02_Shubhashini.txt"
    f.write_text("x" * 5000, encoding="utf-8")
    ctx = _ctx(tmp_path)

    tools.run_tool("read_file", {"path": "Chapter_02_Shubhashini.txt"}, ctx)
    again = tools.run_tool("read_file", {"path": "Chapter_02_Shubashini.txt"}, ctx)
    assert "xxxx" not in again, "the typo bypassed the dedupe"
    assert "already read" in again.lower()


def test_a_file_that_changed_is_read_again(tmp_path):
    """A read after a write must return the NEW content. Deduping on filename
    alone would hand back a stale copy and the model would never see its own
    edit."""
    f = tmp_path / "notes.txt"
    f.write_text("before", encoding="utf-8")
    ctx = _ctx(tmp_path)
    assert "before" in tools.run_tool("read_file", {"path": "notes.txt"}, ctx)

    f.write_text("after the edit", encoding="utf-8")
    again = tools.run_tool("read_file", {"path": "notes.txt"}, ctx)
    assert "after the edit" in again


def test_a_different_page_of_the_same_file_is_read(tmp_path):
    """Paged reads of a large file are different content, not a repeat."""
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line {i}" for i in range(2000)), encoding="utf-8")
    ctx = _ctx(tmp_path)
    tools.run_tool("read_file", {"path": "big.txt", "offset": 0}, ctx)
    page2 = tools.run_tool("read_file", {"path": "big.txt", "offset": 900}, ctx)
    assert "line 9" in page2 or "line 1" in page2
    assert "already read" not in page2.lower()


def test_a_fresh_turn_reads_it_again(tmp_path):
    """The cache lives in the per-turn ctx. A new turn is a new ctx, and the
    model legitimately needs the content back in its window."""
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    assert "hello world" in tools.run_tool("read_file", {"path": "a.txt"},
                                           _ctx(tmp_path))
    assert "hello world" in tools.run_tool("read_file", {"path": "a.txt"},
                                           _ctx(tmp_path))
