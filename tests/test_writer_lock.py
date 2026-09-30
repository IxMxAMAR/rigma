"""ODR-8: one lock for the method/settings writers, held by the restore.

The restore region held only the memory store's `_xlock`. `methods.save_user` /
`delete_user` and `app_settings.save` took no lock, so a save inside the
restore's millisecond window could be deleted after its caller was told
"saved", or overwritten by the restore's rollback.
"""
import threading

import pytest

from rigma import app_settings, methods, serve, writelock
from rigma.memory import MemoryStore


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]}]}


def test_save_user_is_serialised_against_a_held_lock(home):
    """While the lock is held (as the restore holds it), a save must not
    interleave — it waits, then writes normally."""
    started = threading.Event()
    finished = threading.Event()

    def writer():
        started.set()
        methods.save_user(_doc("mine"))
        finished.set()

    with writelock.WRITER_LOCK:
        t = threading.Thread(target=writer)
        t.start()
        assert started.wait(2.0)
        assert not finished.wait(0.3), (
            "save_user wrote while the writer lock was held")
    t.join(5)
    assert finished.is_set(), "save_user never completed after the lock released"
    assert methods.get("mine") is not None


def test_delete_user_and_settings_save_take_the_lock(home):
    saved, errs = methods.save_user(_doc("mine"))
    assert saved is not None and not errs, errs
    deleted = threading.Event()
    settings_done = threading.Event()

    def writer():
        methods.delete_user("mine")
        deleted.set()
        app_settings.save({"idle_unload_minutes": 7})
        settings_done.set()

    with writelock.WRITER_LOCK:
        t = threading.Thread(target=writer)
        t.start()
        assert not deleted.wait(0.3), "delete_user ran while the lock was held"
        assert not settings_done.wait(0.3), (
            "app_settings.save ran while the lock was held")
    t.join(5)
    assert deleted.is_set() and settings_done.is_set()
    assert methods.get("mine") is None
    assert app_settings.load()["idle_unload_minutes"] == 7


def test_the_restore_holds_the_writer_lock(home, monkeypatch):
    """Direct proof the restore region is inside the lock, not just the writers:
    while it is in the failing method write, a probe thread cannot take it."""
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="x")
    saved, errs = methods.save_user(_doc("mine"))
    assert saved is not None and not errs, errs

    held = {}
    in_region = threading.Event()

    def probe_and_fail(full):
        def probe():
            got = writelock.WRITER_LOCK.acquire(timeout=0.3)
            held["got"] = got
            if got:
                writelock.WRITER_LOCK.release()
        p = threading.Thread(target=probe)
        p.start()
        p.join(3.0)
        in_region.set()
        raise OSError("no space left on device")

    monkeypatch.setattr(methods, "save_user", probe_and_fail)
    with pytest.raises(serve._RestoreFailed):
        serve._apply_restore(
            store,
            [("settings", app_settings.settings_path()),
             ("memory", store.path)],
            {},
            [{"id": "aaa"}],
            [{"id": "m1", "kind": "technique", "text": "r"}])
    assert in_region.is_set(), "the method write never ran"
    assert held.get("got") is False, (
        "the restore region did not hold the writer lock")
