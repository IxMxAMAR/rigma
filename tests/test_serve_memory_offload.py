"""14-10: memory recall and the embedder load must not run on the event loop.

`memory.retrieve` runs a synchronous embedding forward pass, and its first call
constructs the fastembed model from disk. fastembed is optional (and absent
from the review venv), so these use stubs and never require it.
"""
import asyncio

from rigma import memory, serve


def _ran_on_loop() -> bool:
    """True when called from the asyncio event-loop thread."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_memory_recall_runs_off_the_event_loop(monkeypatch):
    seen = {}

    def fake_retrieve(rows, query, workspace="", k=3):
        seen["on_loop"] = _ran_on_loop()
        return [{"text": "hit"}]

    monkeypatch.setattr(memory, "retrieve", fake_retrieve)
    out = asyncio.run(serve._recall_memories([], "q", "ws"))
    assert out == [{"text": "hit"}]
    assert seen["on_loop"] is False, "recall ran on the event loop"


def test_embedder_warm_up_runs_off_the_event_loop(monkeypatch):
    seen = {}

    def fake_get():
        seen["on_loop"] = _ran_on_loop()
        return None

    monkeypatch.setattr(memory, "get_embedder", fake_get)
    asyncio.run(serve._warm_memory_embedder())
    assert seen["on_loop"] is False, "the model load ran on the event loop"
