"""R3: every write that can grow a kind's group must re-apply that kind's cap.

AUDIT 10-10 bounded every memory kind with the same least-proven eviction key,
applied in `add()`. `update()` was the other way into a group: the memory trust
surface's PATCH accepts `kind` (that is how a mis-filed rule is corrected), and
moving a row into a kind already at the cap left the store one over. The cap was
then re-applied by the NEXT add, which evicted two rows to get back under it —
one more rule dropped than the cap promises, and the extra victim chosen by
`_eviction_key` rather than by the add that triggered it. A silently dropped
just-learned rule is the failure mode the cap is supposed to make predictable.
"""
from rigma import memory


def _store_at_the_cap(tmp_path, n=None):
    store = memory.MemoryStore(tmp_path / "m.jsonl")
    for i in range(n or memory.MAX_PITFALLS):
        store.add(kind="pitfall", text=f"Rule number {i}.")
    return store


def test_a_kind_edit_cannot_push_a_group_over_the_cap(tmp_path):
    store = _store_at_the_cap(tmp_path)
    t = store.add(kind="technique", text="A technique that was mis-filed.")
    store.update(t["id"], kind="pitfall")
    pits = [r for r in store.all() if r["kind"] == "pitfall"]
    assert len(pits) == memory.MAX_PITFALLS, (
        f"{len(pits)} pitfalls after moving one in — the cap is "
        f"{memory.MAX_PITFALLS}")


def test_the_next_add_does_not_have_to_clean_up_an_over_cap_store(tmp_path):
    """The consequence: with the group left one over, the next add's cap pass
    has to evict the extra row as well. The store must come out of the edit at
    the cap, so the next add evicts exactly the one it is supposed to."""
    store = _store_at_the_cap(tmp_path)
    t = store.add(kind="technique", text="A technique that was mis-filed.")
    store.update(t["id"], kind="pitfall")
    before = len(store.all())
    store.add(kind="pitfall", text="A genuinely new rule.")
    after = len(store.all())
    assert before == memory.MAX_PITFALLS, (
        f"the edit left {before} pitfalls, over the cap of "
        f"{memory.MAX_PITFALLS}")
    assert after == memory.MAX_PITFALLS
    assert "A genuinely new rule." in {r["text"] for r in store.all()}, (
        "the just-learned rule was the eviction victim")


def test_an_edit_that_does_not_change_the_kind_evicts_nothing(tmp_path):
    store = _store_at_the_cap(tmp_path)
    ids = [r["id"] for r in store.all()]
    store.update(ids[0], text="Rule number 0, corrected.")
    assert len(store.all()) == memory.MAX_PITFALLS
    assert "Rule number 0, corrected." in {r["text"] for r in store.all()}


def test_a_kind_edit_into_a_group_with_room_keeps_everything(tmp_path):
    store = memory.MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Rule one.")
    t = store.add(kind="technique", text="A technique.")
    store.update(t["id"], kind="pitfall")
    texts = {r["text"] for r in store.all()}
    assert texts == {"Rule one.", "A technique."}
