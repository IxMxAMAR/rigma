"""Tools must SIGNAL truncation so the model never mistakes a partial view for
the whole thing (regression: a 2330-file folder showed only 200 silently)."""
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


def test_read_file_marks_truncation(tmp_path):
    (tmp_path / "big.txt").write_text("A" * 25000)
    out = tools.run_tool("read_file", {"path": "big.txt"}, _ws(tmp_path))
    assert "truncated" in out.lower()
