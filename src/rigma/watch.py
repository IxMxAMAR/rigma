"""Watch a workspace so `undo_last_change` can reach what the ARM changed.

Rigma snapshots a file before ITS OWN `write_file`/`edit_file` touches it, which
is why `undo_last_change` works in the native loop. An external agent edits with
its OWN tools, in its own process — Rigma never sees the write, so no snapshot is
taken and the tool answers "nothing to undo" every single time. It looks like a
tool that works and finds nothing, which is the worst shape a broken tool can
have.

This is the missing half: remember what each file looked like, so that when one
changes the PREVIOUS bytes are still available to snapshot.

## Why it remembers content rather than watching for events

A change is only visible after it has happened, and by then the old bytes are
gone. There is no event that carries them. So the only way to be able to undo the
FIRST change to a file is to have read that file before it changed — which means
holding content, not just metadata. The budget below is the price of that, and it
is why the caps exist rather than being left unbounded.

## What it deliberately does not do

It records nothing at start-up. The first pass only fills memory; an index entry
is written when a change is actually detected, so `undo_last_change` can never
pick a "baseline" that no one changed. That matters because the tool's no-argument
form means "the most recent change", and a workspace-sized set of entries sharing
one timestamp would make that phrase meaningless.

It also does not chase anything outside the workspace, follow symlinks out of it,
or read files past the size cap. A watcher that quietly widened its own scope
would be a surveillance feature, not an undo feature.

The core is `poll_once()`, which takes no locks, starts no threads and can be
called directly from a test. `start()` is a thread around it.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# Trees that are large, generated, or not the owner's work. `node_modules` and
# `.venv` alone can be a hundred thousand files; watching them would spend the
# whole budget on content nobody edits by hand and nobody wants restored.
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", "dist", "build", ".next", ".cache", ".idea", ".vscode",
    "site-packages", ".gradle", "target", "vendor",
})

# Per-file cap. A file bigger than this is not something an agent edited in a way
# you would undo by hand, and holding it would crowd out everything else.
MAX_FILE_BYTES = 2 << 20            # 2 MB

# Total remembered content across the whole workspace. When this is reached, new
# files stop being remembered — but files ALREADY remembered keep being tracked
# and re-remembered, because dropping those would silently stop watching exactly
# the files that are being worked on.
MAX_TOTAL_BYTES = 64 << 20          # 64 MB

# A ceiling on how many files one pass will look at, so a pathological tree
# cannot make a poll take minutes.
MAX_FILES = 20000

POLL_SECS = 1.5


def _iter_files(root: Path):
    """Candidate files under `root`, pruned and capped.

    Pruning happens on `dirnames` in place, which is what stops `os.walk` from
    descending — skipping entries after the fact would still have walked them.
    """
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p
            seen += 1
            if seen >= MAX_FILES:
                return


def record_snapshot(p: Path, data: bytes) -> bool:
    """Record `data` as the version `undo_last_change` should restore for `p`.

    Writes the SAME slot and index entry that `tools._snapshot_before_write`
    writes, so `undo_last_change` needs no changes at all: it reads one index and
    does not care which of the two recorded an entry. That also means the undo
    tool's existing swap (restore, then put the current bytes in the slot so the
    undo is redoable) keeps working here for free.

    Returns False rather than raising, for the same reason the write path does:
    a safety net must never break the thing it is protecting.
    """
    import hashlib

    from . import tools as T
    try:
        d = T._undo_dir()
        h = hashlib.sha1(str(p).encode("utf-8", "replace")).hexdigest()[:12]
        snap = d / f"{h}-{p.name}"
        T._atomic_bytes(snap, data)
        # Read-modify-write under the lock: one index covers every session, so
        # an entry another turn recorded between the read and the write would be
        # dropped and that file would become un-undoable.
        with T._FILE_LOCK:
            idx = T._undo_index(d)
            idx[str(p)] = {"snap": snap.name, "ts": time.time(),
                           "size": len(data)}
            T._write_undo_index(d, idx)
        return True
    except Exception:
        return False


class Watcher:
    """Remembers what each file looked like, so a change can be undone.

    `record` is injectable so a test can assert WHICH files were recorded and
    with WHAT bytes, without touching the real undo store.
    """

    def __init__(self, workspace, *, record=None):
        self.root = Path(workspace).resolve()
        self._known: dict[str, bytes] = {}
        self._bytes = 0
        self._record = record if record is not None else record_snapshot
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.recorded: list[str] = []

    # -- the testable core ----------------------------------------------------

    def poll_once(self) -> list[str]:
        """One pass. Returns the paths whose change was recorded."""
        changed: list[str] = []
        for p in _iter_files(self.root):
            key = str(p)
            try:
                data = p.read_bytes()
            except OSError:
                continue
            with self._lock:
                old = self._known.get(key)
                if old is None:
                    # First sight: this is the baseline, and nothing has changed
                    # yet, so there is nothing to record.
                    self._remember(key, data)
                    continue
                if old == data:
                    continue
                # The change has already happened and `old` is the version to go
                # back to — which only exists because the previous pass read it.
                if self._record(p, old):
                    changed.append(key)
                self._remember(key, data)
        return changed

    def _remember(self, key: str, data: bytes) -> None:
        prev = self._known.get(key)
        if prev is not None:
            self._bytes -= len(prev)
        elif self._bytes + len(data) > MAX_TOTAL_BYTES:
            # Over budget, and this is a NEW file: do not start tracking it.
            # Files already tracked are always kept — refusing to re-remember one
            # of those would stop watching the files actually being edited.
            return
        self._known[key] = data
        self._bytes += len(data)

    # -- the thread around it -------------------------------------------------

    @property
    def remembered_bytes(self) -> int:
        return self._bytes

    @property
    def remembered_files(self) -> int:
        return len(self._known)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rigma-watch",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # A failure in here must never kill the thread: a watcher that died
        # silently would leave the arm with a tool that had stopped working, and
        # nothing would say so.
        while not self._stop.wait(POLL_SECS):
            try:
                self.poll_once()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)


_WATCHERS: dict[str, Watcher] = {}
_WATCH_LOCK = threading.Lock()


def watch(workspace: str) -> Watcher | None:
    """Start (or return) the watcher for a workspace. None if unusable.

    One watcher per workspace, shared: the arm's MCP server and Rigma's own turn
    path can both ask for the same directory, and two watchers would double the
    memory for one job.
    """
    ws = str(workspace or "").strip()
    if not ws:
        return None
    try:
        root = Path(ws).resolve()
    except (OSError, ValueError):
        return None
    if not root.is_dir():
        return None
    key = str(root)
    with _WATCH_LOCK:
        w = _WATCHERS.get(key)
        if w is None:
            w = Watcher(root)
            _WATCHERS[key] = w
        w.start()
        return w


def watchers() -> dict[str, Watcher]:
    """A copy of the live registry, for tests and diagnostics."""
    with _WATCH_LOCK:
        return dict(_WATCHERS)


def stop_all() -> None:
    with _WATCH_LOCK:
        live = list(_WATCHERS.values())
        _WATCHERS.clear()
    for w in live:
        w.stop()
