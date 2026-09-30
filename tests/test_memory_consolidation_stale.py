"""10-R3-16: the conflict gate must not act on a snapshot taken before its await.

`add_consolidated` nominates a candidate from the rows it read BEFORE awaiting
the model's verdict. A second Rigma process can rewrite that row while the gate
is answering, and the id still resolves — so the CONFLICT/DUPLICATE branch used
to act on text the verdict was never about. These tests drive exactly that
interleave: the gate callback rewrites the nominated row before returning.
"""
import asyncio

import pytest

from rigma import memory
from rigma.memory import MemoryStore, add_consolidated


def _run(coro):
    return asyncio.run(coro)


def _reply(text):
    async def _c(prompt):
        return text
    return _c


@pytest.fixture(autouse=True)
def _dense(monkeypatch):
    """Every rule nominates every other rule, so the gate always runs."""
    monkeypatch.setenv("RIGMA_MEMORY_EMBED", "0")
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    yield


def _rewrite_then(store, row_id, new_text, verdict):
    """A gate that stands in for a second process writing mid-answer."""
    async def complete(prompt):
        store.update(row_id, text=new_text)
        return verdict
    return complete


def test_a_conflict_verdict_does_not_retire_a_rewritten_rule(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    old = store.add(kind="pitfall", text="Prefer the f16 KV cache.")
    _run(add_consolidated(
        store, "pitfall", "Prefer the q8_0 KV cache.",
        _rewrite_then(store, old["id"], "Prefer the bf16 KV cache.",
                      "CONFLICT")))
    rows = {r["text"]: r for r in store.all()}
    # the verdict was about "f16"; the row now says "bf16" and must survive it
    assert rows["Prefer the bf16 KV cache."]["status"] == "draft"
    assert "Prefer the q8_0 KV cache." in rows


def test_a_duplicate_verdict_does_not_credit_a_rewritten_rule(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    old = store.add(kind="pitfall", text="Never type filenames.")
    _run(add_consolidated(
        store, "pitfall", "Do not manually type file names.",
        _rewrite_then(store, old["id"], "Never retype a path from memory.",
                      "DUPLICATE")))
    rows = store.all()
    assert len(rows) == 2, rows
    changed = next(r for r in rows
                   if r["text"] == "Never retype a path from memory.")
    assert changed["seen_count"] == 1


def test_a_deleted_nomination_appends(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    old = store.add(kind="pitfall", text="Prefer the f16 KV cache.")

    async def complete(prompt):
        store.delete(old["id"])
        return "CONFLICT"

    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          complete))
    texts = {r["text"] for r in store.all()}
    assert "Prefer the q8_0 KV cache." in texts


def test_a_conflict_verdict_about_an_unchanged_rule_still_supersedes(tmp_path):
    """The control: re-validation must not disable consolidation itself."""
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Prefer the f16 KV cache.", None))
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          _reply("CONFLICT")))
    rows = {r["text"]: r for r in store.all()}
    assert rows["Prefer the f16 KV cache."]["status"] == "retired"
    assert rows["Prefer the q8_0 KV cache."]["status"] == "draft"
