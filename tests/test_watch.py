"""The workspace watcher: what makes `undo_last_change` reach the ARM's edits.

Rigma snapshots a file before ITS OWN write_file/edit_file, which is why undo
works in the native loop. An external agent writes with its own tools in its own
process, so Rigma never sees the write and no snapshot was taken — the tool then
answered "nothing to undo" every single time.

The core is `poll_once()`, which starts no threads and takes no locks, so every
test here is deterministic: no sleeps, no polling intervals, no timing.

Covers the open item in docs/HANDOFF-harness-rework.md.
"""
import pytest

from rigma import tools, watch


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


def _recorder():
    """A stand-in for the undo store: records (path, bytes) in order."""
    calls = []

    def record(p, data):
        calls.append((str(p), data))
        return True

    return calls, record


# -- the core: nothing is recorded until something actually changes -----------

def test_the_first_pass_records_nothing(ws):
    """It only learns what things look like. Recording a "baseline" would put a
    workspace-sized set of entries in the undo index sharing one timestamp, and
    `undo_last_change` with no argument means "the most recent change" — a phrase
    that would stop meaning anything."""
    (ws / "a.txt").write_bytes(b"one")
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)

    assert w.poll_once() == []
    assert calls == []
    assert w.remembered_files == 1


def test_a_change_records_the_bytes_from_before_it(ws):
    """The whole point: a change is only visible after it happened, so the old
    bytes can only come from the previous pass having read the file."""
    f = ws / "a.txt"
    f.write_bytes(b"original")
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)
    w.poll_once()

    f.write_bytes(b"clobbered by the arm")
    assert w.poll_once() == [str(f)]

    assert len(calls) == 1
    assert calls[0][0] == str(f)
    assert calls[0][1] == b"original", "must record the PRE-change bytes"


def test_successive_changes_each_record_their_own_predecessor(ws):
    """Not the original, and not the newest: each change has to be undoable to
    the state immediately before it."""
    f = ws / "a.txt"
    f.write_bytes(b"v1")
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)
    w.poll_once()

    f.write_bytes(b"v2")
    w.poll_once()
    f.write_bytes(b"v3")
    w.poll_once()

    assert [c[1] for c in calls] == [b"v1", b"v2"]


def test_an_unchanged_file_is_not_recorded_again(ws):
    f = ws / "a.txt"
    f.write_bytes(b"same")
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)
    for _ in range(4):
        w.poll_once()
    assert calls == []


def test_a_new_file_is_not_a_change(ws):
    """A file that appears was not "changed" — there is no earlier version to
    restore, and recording its creation as an undo point would restore a
    deletion."""
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)
    w.poll_once()
    (ws / "born.txt").write_bytes(b"new")
    assert w.poll_once() == []
    assert calls == []


def test_a_recorded_change_that_fails_to_store_is_not_reported(ws):
    """`record` returns False when the undo store could not be written. The
    caller must not be told a change was captured when it was not."""
    f = ws / "a.txt"
    f.write_bytes(b"v1")
    w = watch.Watcher(ws, record=lambda p, d: False)
    w.poll_once()
    f.write_bytes(b"v2")
    assert w.poll_once() == []


# -- what it refuses to walk --------------------------------------------------

def test_generated_trees_are_not_watched(ws):
    """`node_modules` and `.venv` can be a hundred thousand files. Spending the
    budget there would crowd out the files somebody actually edits."""
    for d in ("node_modules", ".git", ".venv", "__pycache__", "dist"):
        (ws / d).mkdir()
        (ws / d / "junk.txt").write_bytes(b"x")
    (ws / "kept.txt").write_bytes(b"x")

    w = watch.Watcher(ws)
    w.poll_once()
    assert w.remembered_files == 1


def test_a_file_past_the_size_cap_is_skipped(ws):
    big = ws / "big.bin"
    big.write_bytes(b"x" * (watch.MAX_FILE_BYTES + 1))
    (ws / "small.txt").write_bytes(b"x")

    w = watch.Watcher(ws)
    w.poll_once()
    assert w.remembered_files == 1


def test_a_full_budget_does_not_stop_watching_what_is_already_tracked(ws,
                                                                     monkeypatch):
    """The failure mode this guards: once the budget filled, refusing to
    RE-remember a tracked file would silently stop watching exactly the files
    being worked on — the watcher would look alive and record nothing."""
    monkeypatch.setattr(watch, "MAX_TOTAL_BYTES", 8)
    f = ws / "tracked.txt"
    f.write_bytes(b"12345678")            # exactly the budget
    calls, record = _recorder()
    w = watch.Watcher(ws, record=record)
    w.poll_once()

    # a new file cannot be afforded
    (ws / "new.txt").write_bytes(b"z")
    w.poll_once()
    assert w.remembered_files == 1

    # but the tracked one still is, and its change is still recorded
    f.write_bytes(b"CHANGED!")
    assert w.poll_once() == [str(f)]
    assert calls[0][1] == b"12345678"


# -- the registry -------------------------------------------------------------

def test_watch_refuses_a_workspace_that_is_not_one(tmp_path):
    assert watch.watch("") is None
    assert watch.watch("   ") is None
    assert watch.watch(str(tmp_path / "nope")) is None


def test_watch_is_one_watcher_per_workspace(ws):
    """Two watchers on one directory would double the memory for one job."""
    a = watch.watch(str(ws))
    b = watch.watch(str(ws))
    try:
        assert a is b
    finally:
        watch.stop_all()
    assert watch.watchers() == {}


# -- the end-to-end property: the arm breaks a file, undo fixes it -------------

def test_undo_last_change_restores_a_file_the_arm_changed(home, ws):
    """The integration this feature exists for. Nothing here calls Rigma's own
    write_file — the edit stands in for the arm's own tool, which Rigma never
    sees — and the undo must still work."""
    f = ws / "chapter.md"
    f.write_bytes(b"the version the owner wanted")

    w = watch.Watcher(ws)                  # real undo store, real record
    w.poll_once()                          # learns the baseline

    f.write_bytes(b"the arm replaced it")  # the arm's own write
    assert w.poll_once() == [str(f)]

    out = tools.run_tool("undo_last_change", {}, {"workspace": str(ws),
                                                  "allow_code": True,
                                                  "profile": "all"})
    assert "restored" in out, out
    assert f.read_bytes() == b"the version the owner wanted"


def test_the_undo_can_be_swapped_back(home, ws):
    """The undo tool puts the current bytes into the slot so the undo is
    redoable. The watcher must not then fight it: after the swap the slot holds
    the arm's version, which is exactly what a second call should restore."""
    f = ws / "chapter.md"
    f.write_bytes(b"v1")
    w = watch.Watcher(ws)
    w.poll_once()
    f.write_bytes(b"v2")
    w.poll_once()

    ctx = {"workspace": str(ws), "allow_code": True, "profile": "all"}
    assert "restored" in tools.run_tool("undo_last_change", {}, ctx)
    assert f.read_bytes() == b"v1"

    # the watcher sees the undo as a change too, and that is fine: it records v2,
    # which is the same bytes the swap already put in the slot
    w.poll_once()
    assert "restored" in tools.run_tool("undo_last_change", {}, ctx)
    assert f.read_bytes() == b"v2"


def test_a_change_outside_the_workspace_is_not_undoable(home, ws, tmp_path):
    """One undo index covers the whole install (AUDIT F8), so the containment
    test is what keeps a chat from reverting another project."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    g = other / "x.txt"
    g.write_bytes(b"not this project")
    w = watch.Watcher(ws)
    w.poll_once()

    out = tools.run_tool("undo_last_change", {}, {"workspace": str(ws),
                                                  "allow_code": True,
                                                  "profile": "all"})
    assert "nothing to undo" in out
    assert g.read_bytes() == b"not this project"
