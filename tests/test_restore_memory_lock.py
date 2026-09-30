"""A11c: `/api/restore`'s memory snapshot/rollback ran outside the store lock.

The route snapshotted the store's bytes, wrote settings and methods, and on any
failure put the snapshot back. Only the apply write took `MemoryStore._xlock`,
so a memory committed between the snapshot and the rollback was silently
reverted by the rollback — a lost update, on the file that is the only copy of
months of learned rules.

The whole snapshot -> apply -> rollback region must be consistent with respect
to `_xlock`. A real concurrent writer must not be lost: it is ordered either
before the snapshot (and rolled back with it, which the restore's replace
semantics allow) or after the region (and survives). This test uses a real
thread with its own `MemoryStore`, which is the only honest model of the
concurrency the file lock exists for.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from rigma import app_settings, methods
from rigma.memory import MemoryStore
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


@pytest.fixture
def client(home):
    return TestClient(build_app(upstream_port=1),
                      raise_server_exceptions=False)


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]}]}


def _seed(home):
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    app_settings.save({"idle_unload_minutes": 12})
    saved, errs = methods.save_user(_doc())
    assert saved is not None and not errs, errs
    return store


def test_a_concurrent_memory_write_survives_the_rollback(
        client, home, monkeypatch):
    """A write committed while the restore is failing must not be reverted.

    The restore fails at the method write — the point BETWEEN the memory
    snapshot and the rollback — and a concurrent thread adds a memory in that
    window. Before the fix the rollback put the pre-restore bytes back and the
    add was gone.
    """
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99
    doc["methods"] = [_doc("aaa")]
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]

    started = threading.Event()
    done = threading.Event()

    def writer():
        started.wait(10)
        # A real concurrent writer: its own store, its own thread.
        MemoryStore(store.path).add(kind="technique", text="concurrent row")
        done.set()

    def flaky(full):
        # Inside the apply region, after the memory snapshot was taken. Let the
        # writer go and give it a bounded moment to land. With the region
        # holding `_xlock` it cannot; without it, it lands here and is reverted.
        started.set()
        done.wait(2.0)
        raise OSError("no space left on device")

    monkeypatch.setattr(methods, "save_user", flaky)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        r = client.post("/api/restore", json=doc)
        t.join(15)
    finally:
        started.set()               # never leave the writer waiting
    assert r.status_code == 500, r.text
    assert "methods" in r.json()["error"], r.text
    assert done.is_set(), "the concurrent writer never finished"

    texts = [m["text"] for m in store.all()]
    assert "concurrent row" in texts, (
        "a memory committed during the restore window was silently reverted: "
        f"{texts!r}")
    # the restore itself was rolled back: the backup's row is not there and the
    # pre-restore row still is
    assert "Never type filenames." in texts
    assert "Prefer q8_0." not in texts


def test_the_region_holds_the_store_lock(client, home, monkeypatch):
    """Direct proof the fix is a lock and not a timing coincidence: while the
    apply region is inside the failing method write, a second `MemoryStore`
    cannot take the cross-process lock."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["methods"] = [_doc("aaa")]
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]

    from rigma import memory as _mem
    held = {}
    in_region = threading.Event()

    def flaky(full):
        probe = _mem._FileLock(store.lock_path)
        held["got"] = probe.acquire(timeout=0.3)
        probe.release()
        in_region.set()
        raise OSError("no space left on device")

    monkeypatch.setattr(methods, "save_user", flaky)
    r = client.post("/api/restore", json=doc)
    assert in_region.is_set(), "the method write never ran"
    assert r.status_code == 500, r.text
    assert held["got"] is False, (
        "the snapshot/rollback region did not hold the store lock")


def test_a_clean_restore_still_replaces_the_store(client, home):
    """The lock must not turn a good restore into a no-op."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200, r.text
    assert r.json()["memory"] == {"before": 1, "after": 1}
    assert [m["text"] for m in store.all()] == ["Prefer q8_0."]


def test_an_untouched_store_is_not_rolled_back_by_an_earlier_failure(
        home, monkeypatch):
    """The verifier's shape, made fast: a memory is committed during the region
    by the SAME thread on the SAME store (so the store's re-entrant lock lets
    it through), and the region then fails at an EARLIER section. The store was
    never written by the restore, so the rollback must leave it alone."""
    from rigma import serve

    store = _seed(home)
    targets = [("settings", app_settings.settings_path()),
               ("method aaa", methods._method_file("aaa")),
               ("memory", store.path)]

    def flaky(full):
        store.add(kind="technique", text="committed mid-restore")
        raise OSError("no space left on device")

    monkeypatch.setattr(methods, "save_user", flaky)
    with pytest.raises(serve._RestoreFailed) as ei:
        serve._apply_restore(
            store, targets, {"idle_unload_minutes": 99},
            [{"id": "aaa"}],
            [{"id": "mem-1", "kind": "technique", "text": "Prefer q8_0."}])
    assert "methods" in str(ei.value), str(ei.value)
    texts = [m["text"] for m in store.all()]
    assert "committed mid-restore" in texts, texts
    assert "Prefer q8_0." not in texts
    assert "Never type filenames." in texts


# ---------------------------------------------------------------------------
# A11d: the in-process lock must actually serialise two RESTORES.
#
# A11c put the whole snapshot -> apply -> rollback region under the store's
# `_xlock`, but `serve._memory_store()` built a NEW `MemoryStore` per request, so
# the RLock was a different lock for each restore. Only the 10-second
# cross-process file lock serialised them — and `_FileLock.acquire` gives up
# after that and lets the caller proceed unsynchronised. `memory.store_for`
# makes the store (and therefore the RLock) process-wide.
# ---------------------------------------------------------------------------


def test_two_concurrent_restores_serialise_on_the_shared_store_lock(
        client, home, monkeypatch):
    """With the cross-process file lock made unavailable, the ONLY thing that
    can keep two apply regions apart is the in-process store lock. Before A11d
    both regions ran at once; after it, the second waits for the first."""
    from rigma import memory as _mem
    _seed(home)
    doc = client.get("/api/backup").json()

    # The cross-process fallback is unavailable for this test.
    monkeypatch.setattr(_mem._FileLock, "acquire", lambda self, **k: False)

    guard = threading.Lock()
    active = {"n": 0}
    overlap = threading.Event()

    def slow_save_user(full):
        # A real overlap detector: it fires only when two apply regions are
        # inside the method write at the same instant.
        with guard:
            active["n"] += 1
            if active["n"] > 1:
                overlap.set()
        time.sleep(0.3)
        with guard:
            active["n"] -= 1
        return (full, [])

    monkeypatch.setattr(methods, "save_user", slow_save_user)

    results = []

    def restore(tag):
        c = TestClient(build_app(upstream_port=1),
                       raise_server_exceptions=False)
        body = {**doc, "methods": [_doc(tag)]}
        results.append(c.post("/api/restore", json=body).status_code)

    threads = [threading.Thread(target=restore, args=(f"m{i}",))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    assert not overlap.is_set(), (
        "two concurrent restores entered their apply regions at once — the "
        "in-process store lock is not shared across requests")
    assert sorted(results) == [200, 200], results
