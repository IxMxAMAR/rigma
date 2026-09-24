"""IMP-6: the memory store can be inspected, corrected and pruned.

Rigma silently learns rules from runs and injects them into later prompts. A
feature the user cannot see, fix or delete is one they cannot trust, so the
trust surface is the point — not a convenience.
"""
import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def store(home):
    from rigma.memory import MemoryStore
    return MemoryStore(home / "memory" / "memories.jsonl")


@pytest.fixture
def client(home):
    return TestClient(build_app(upstream_port=1))


def _by_id(store, mid):
    return next(r for r in store.all() if r["id"] == mid)


def test_memory_list_edit_and_prune(client, store):
    a = store.add(kind="pitfall", text="Never type filenames.")
    b = store.add(kind="technique", text="Prefer q8_0 for the draft model.")
    assert {r["id"] for r in client.get("/api/memory").json()} == {a["id"],
                                                                   b["id"]}

    # PATCH corrects the text in place; the id (the row's identity) is stable
    r = client.patch(f"/api/memory/{a['id']}",
                     json={"text": "Never retype a filename; sample it."})
    assert r.status_code == 200
    assert r.json()["id"] == a["id"]
    assert r.json()["text"] == "Never retype a filename; sample it."
    assert "vec" not in r.json()          # 768 floats of noise, never sent
    assert _by_id(store, a["id"])["text"] == "Never retype a filename; sample it."

    # the kind filter is server-side, so a tab does not need the kind list
    only = client.get("/api/memory", params={"kind": "technique"}).json()
    assert [x["id"] for x in only] == [b["id"]]

    # a prune without ids must NOT empty the store
    r = client.delete("/api/memory")
    assert r.status_code == 400 and "ids" in r.json()["error"]
    assert len(client.get("/api/memory").json()) == 2

    r = client.request("DELETE", "/api/memory", json={"ids": [b["id"]]})
    assert r.status_code == 200 and r.json()["removed"] == 1
    assert [x["id"] for x in client.get("/api/memory").json()] == [a["id"]]

    # one forget, then a 404 — "already gone" is not a silent success
    assert client.delete(f"/api/memory/{a['id']}").status_code == 200
    assert client.get("/api/memory").json() == []
    assert client.delete(f"/api/memory/{a['id']}").status_code == 404


def test_memory_edit_validates_the_correction(client, store):
    m = store.add(kind="pitfall", text="Never type filenames.")
    r = client.patch(f"/api/memory/{m['id']}", json={"status": "gone"})
    assert r.status_code == 400 and "status" in r.json()["error"]
    r = client.patch(f"/api/memory/{m['id']}", json={"kind": "Not A Kind"})
    assert r.status_code == 400 and "kind" in r.json()["error"]
    # an edit is another way to put a failure transcript into a rule
    r = client.patch(f"/api/memory/{m['id']}",
                     json={"text": r"C:\out\Comfy_UI_428.png failed twice"})
    assert r.status_code == 400 and "raw trace" in r.json()["error"]
    # nothing was written by any refused edit
    assert _by_id(store, m["id"])["text"] == "Never type filenames."
    assert _by_id(store, m["id"])["status"] == "draft"

    assert client.patch(f"/api/memory/{m['id']}", json={}).status_code == 400
    assert client.patch("/api/memory/nope",
                        json={"text": "x"}).status_code == 404


def test_memory_edit_keeps_the_score_and_races_nothing(store):
    """The edit and a scoring pass are both read-modify-writes over the whole
    file. An edit that read outside the lock would either clobber a score that
    landed between the read and the write, or lose a concurrent add."""
    import threading

    from rigma import memory as _mem
    m = store.add(kind="pitfall", text="Never type filenames.")
    _mem.score_memories(store, [m["id"]], +1)

    def _adder():
        for i in range(20):
            store.add(kind="technique", text=f"technique number {i}")

    t = threading.Thread(target=_adder)
    t.start()
    store.update(m["id"], text="Sample, never retype.")
    t.join()

    rows = store.all()
    assert len(rows) == 21                     # no add was lost
    row = _by_id(store, m["id"])
    assert row["text"] == "Sample, never retype."
    assert row["outcome_score"] == 1           # the score was not clobbered


def test_store_update_refuses_bad_values(store):
    m = store.add(kind="pitfall", text="Never type filenames.")
    with pytest.raises(ValueError):
        store.update(m["id"], text="   ")
    with pytest.raises(ValueError):
        store.update(m["id"], outcome_score="lots")
    assert store.update("nope", text="x") is None
    assert store.delete("nope") is False
    assert store.delete_many([]) == 0
