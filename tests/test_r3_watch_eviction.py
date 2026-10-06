"""R3-6: the watcher never forgets a file that is gone.

`_known`/`_stats` are written on every pass and never pruned, and `_bytes` is
only ever decremented when the SAME key is re-remembered. A file that is created
and then deleted therefore holds its entry (and its bytes) forever: the 64 MB
budget is spent on garbage, after which `_remember` refuses every NEW file and
the watcher silently stops watching — and the key count is unbounded, since a
0-byte file costs nothing against the budget.
"""
from pathlib import Path

from rigma import watch


def _recorder():
    calls = []

    def record(p, data):
        calls.append((p, data))
        return True
    return calls, record


def test_a_deleted_file_is_forgotten(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_bytes(b"x" * 10)
    calls, record = _recorder()
    w = watch.Watcher(tmp_path, record=record)
    w.poll_once()
    assert w.remembered_files == 5

    for i in range(3):
        (tmp_path / f"f{i}.txt").unlink()
    w.poll_once()

    assert w.remembered_files == 2
    # `Path(k).name`, not `k.rsplit("\\", 1)[-1]`: the keys are full paths in the
    # running platform's spelling (`poll_once` uses `str(p)` from `os.walk`), and
    # a backslash is not a separator on POSIX — there the split returned the WHOLE
    # path, so the assertion compared absolute paths against basenames and failed
    # on Linux while the eviction it pins was correct.
    assert sorted(Path(k).name for k in w._known) == ["f3.txt", "f4.txt"]
    assert w.remembered_bytes == 20
    assert len(w._stats) == 2


def test_a_deleted_file_does_not_starve_the_budget(tmp_path, monkeypatch):
    """The user-visible failure: churn of small files fills the budget with
    dead entries, and the watcher then stops remembering anything new."""
    monkeypatch.setattr(watch, "MAX_TOTAL_BYTES", 16)
    calls, record = _recorder()
    w = watch.Watcher(tmp_path, record=record)
    for i in range(10):
        f = tmp_path / f"churn{i}.bin"
        f.write_bytes(b"y" * 8)
        w.poll_once()
        f.unlink()
        w.poll_once()
    assert w.remembered_bytes == 0

    keep = tmp_path / "keep.txt"
    keep.write_bytes(b"z" * 8)
    w.poll_once()
    assert w.remembered_files == 1, "the budget is still full of deleted files"


def test_a_truncated_walk_does_not_forget_anything(tmp_path, monkeypatch):
    """`_iter_files` stops at MAX_FILES, so a pass that stopped early has not
    seen the rest of the tree — evicting on it would drop live files."""
    monkeypatch.setattr(watch, "MAX_FILES", 2)
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_bytes(b"x")
    calls, record = _recorder()
    w = watch.Watcher(tmp_path, record=record)
    w.poll_once()
    first = set(w._known)
    assert len(first) == 2
    w.poll_once()
    assert set(w._known) == first
