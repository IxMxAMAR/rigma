"""Tools must SIGNAL truncation so the model never mistakes a partial view for
the whole thing (regression: a 2330-file folder showed only 200 silently)."""
import os
from contextlib import contextmanager
from pathlib import Path

from rigma import tools


def _ws(tmp):
    return {"workspace": str(tmp), "allow_code": True}


def test_list_directory_summarises_large_folders(tmp_path):
    # a big folder is SUMMARISED (counts by type + examples) rather than dumped:
    # 200 raw filenames was a large, low-value chunk of a slow model's context
    for i in range(250):
        (tmp_path / f"f{i:03}.png").write_bytes(b"x")
    out = tools.run_tool("list_directory", {}, _ws(tmp_path))
    assert "250" in out                      # true total still surfaced
    assert "250× .png" in out                # what's actually in there
    assert "sample_files" in out             # and the cheap way to use it
    assert out.count("f0") < 30              # not a full dump


def test_list_directory_small_folder_has_no_more_marker(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    out = tools.run_tool("list_directory", {}, _ws(tmp_path))
    assert "more" not in out.lower()


def test_find_files_reports_truncation(tmp_path):
    for i in range(250):
        (tmp_path / f"f{i:03}.py").write_text("x")
    out = tools.run_tool("find_files", {"pattern": "*.py"}, _ws(tmp_path))
    assert "250" in out and "200 of 250" in out


def test_list_directory_bounds_the_scan_not_just_the_output(tmp_path,
                                                            monkeypatch):
    """14-1: the 200-name cap must not force a stat-per-entry sort of the whole
    directory. With the scan cap at 50 and 200 files on disk, only ~50 entries
    may be examined."""
    monkeypatch.setattr(tools, "_SCAN_MAX", 50)
    for i in range(200):
        (tmp_path / f"f{i:03d}.txt").write_text("x")

    seen = {"n": 0}
    real = os.scandir

    @contextmanager
    def counting(path):
        with real(path) as it:
            def gen():
                for e in it:
                    seen["n"] += 1
                    yield e
            yield gen()

    monkeypatch.setattr(tools.os, "scandir", counting)
    out = tools.run_tool("list_directory", {}, _ws(tmp_path))
    assert seen["n"] <= 51, f"examined {seen['n']} entries for a 200-file folder"
    assert "50+ entries" in out, out


def test_sample_files_bounds_the_scan(tmp_path, monkeypatch):
    """14-1: sample_files used to resolve() every candidate before sampling 20."""
    monkeypatch.setattr(tools, "_SCAN_MAX", 40)
    for i in range(200):
        (tmp_path / f"img_{i:03d}.png").write_text("x")
    out = tools.run_tool("sample_files", {"path": ".", "pattern": "*.png",
                                          "count": 5}, _ws(tmp_path))
    assert "40+ files match" in out, out
    picked = [ln for ln in out.splitlines() if ln.endswith(".png")]
    assert len(picked) == 5


# --- 14-3: grep/find must bound the INPUT, not the output --------------------

def test_grep_bounds_the_input_not_just_the_output(tmp_path, monkeypatch):
    """A never-matching pattern used to read every file under the workspace."""
    monkeypatch.setattr(tools, "_GREP_MAX_FILES", 3)
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("nothing here\n" * 10)

    opened = []
    real = Path.read_text
    monkeypatch.setattr(
        Path, "read_text",
        lambda self, *a, **k: (opened.append(str(self)), real(self, *a, **k))[1])

    out = tools.run_tool("grep", {"pattern": "zzzznope"}, _ws(tmp_path))
    assert len(opened) == 3, opened
    assert "searched the first 3 files" in out, out


def test_grep_default_glob_reaches_nested_files(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("needle\n")
    (tmp_path / "top.txt").write_text("needle\n")
    out = tools.run_tool("grep", {"pattern": "needle"}, _ws(tmp_path))
    assert "top.txt:1" in out and "sub/deep.txt:1" in out, out


def test_grep_glob_is_root_anchored_without_doublestar(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("needle\n")
    (tmp_path / "top.py").write_text("needle\n")
    out = tools.run_tool("grep", {"pattern": "needle", "glob": "*.py"},
                         _ws(tmp_path))
    assert "top.py:1" in out and "deep.py" not in out, out


def test_grep_skips_generated_trees(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "junk.txt").write_text("needle\n")
    (tmp_path / "real.txt").write_text("needle\n")
    out = tools.run_tool("grep", {"pattern": "needle"}, _ws(tmp_path))
    assert "real.txt" in out and "junk" not in out, out


def test_find_files_bounds_the_walk_by_entries_not_hits(tmp_path, monkeypatch):
    """The old islice counted only file HITS, so a tree that was mostly
    directories walked in full (8003 is_file() calls for 3 files)."""
    monkeypatch.setattr(tools, "_WALK_MAX_ENTRIES", 40)
    for i in range(100):
        d = tmp_path / f"d{i:03d}"
        d.mkdir()
        (d / "x.txt").write_text("x")
    out = tools.run_tool("find_files", {"pattern": "**/*.txt"}, _ws(tmp_path))
    assert "stopped after examining" in out, out


def test_read_file_marks_truncation(tmp_path):
    (tmp_path / "big.txt").write_text("A" * 25000)
    out = tools.run_tool("read_file", {"path": "big.txt"}, _ws(tmp_path))
    assert "truncated" in out.lower()
