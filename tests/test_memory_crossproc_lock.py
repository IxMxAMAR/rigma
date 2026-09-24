"""10-1r: the memory store's read-modify-write must be atomic ACROSS processes.

The 10-1 fix added a per-store threading.RLock, which stops two threads in one
process losing each other's writes. Two Rigma processes (the CLI and the server,
or two server instances) still opened the same JSONL, read it, and wrote their
own snapshot back — last writer wins, learned rules and score updates vanish.

The fix is an OS advisory lock on a sidecar file, held across the whole
read-modify-write, on top of the existing in-process RLock. No subprocess is
spawned here (piped stdio is refused by this sandbox and the brief forbids it):
two independent file descriptors in one process are enough to prove the
primitive, because both msvcrt.locking and fcntl.flock conflict across handles
rather than across processes.
"""
import threading

from rigma import memory


def _lock_for(store):
    return memory._FileLock(store.lock_path)


def test_second_handle_is_refused_while_the_first_holds_the_lock(tmp_path):
    store = memory.MemoryStore(tmp_path / "memories.jsonl")
    first, second = _lock_for(store), _lock_for(store)
    assert first.acquire(timeout=0.2) is True
    try:
        # a second, independent descriptor must NOT be able to take it
        assert second.acquire(timeout=0.2) is False
    finally:
        first.release()
    # and once the first lets go, the second can take it
    assert second.acquire(timeout=0.2) is True
    second.release()


def test_release_makes_the_lock_available_to_a_new_handle(tmp_path):
    store = memory.MemoryStore(tmp_path / "memories.jsonl")
    lock = _lock_for(store)
    assert lock.acquire(timeout=0.2) is True
    lock.release()
    other = _lock_for(store)
    assert other.acquire(timeout=0.2) is True
    other.release()


def test_two_store_instances_do_not_lose_each_others_writes(tmp_path):
    """Two MemoryStore objects are two processes, minus the subprocess.

    Each has its OWN RLock, so only the file lock serialises them. Without it
    the whole-file read-modify-write drops most of the additions.
    """
    path = tmp_path / "memories.jsonl"
    stores = [memory.MemoryStore(path), memory.MemoryStore(path)]

    def writer(store, tag):
        for i in range(6):
            store.add(kind="technique", text=f"rule {tag} number {i}")

    threads = [threading.Thread(target=writer, args=(s, n))
               for n, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    texts = {r["text"] for r in memory.MemoryStore(path).all()}
    assert texts == {f"rule {n} number {i}"
                     for n in range(2) for i in range(6)}, texts
