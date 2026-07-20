"""Phase 2+3 memory: hybrid retrieval, conflict-gated consolidation, outcome
scoring and graduation.

Lexical (BM25) paths are always tested. Dense paths run only when a fastembed
model is already cached — retrieval must WORK without one (memory is never
load-bearing), so the lexical assertions are the contract and the dense ones
are the bonus.
"""
import asyncio
import os

import pytest

from rigma import memory
from rigma.memory import (MemoryStore, add_consolidated, retrieve,
                          score_memories)


def _run(coro):
    return asyncio.run(coro)


def _reply(text):
    async def _c(prompt):
        return text
    return _c


@pytest.fixture(autouse=True)
def _no_dense(monkeypatch):
    """Deterministic by default: lexical-only unless a test opts in."""
    monkeypatch.setenv("RIGMA_MEMORY_EMBED", "0")
    memory._embedder, memory._embedder_tried = None, False
    yield
    memory._embedder, memory._embedder_tried = None, False


def _row(text, kind="pitfall", **kw):
    r = {"id": text[:8], "kind": kind, "text": text, "status": "draft",
         "seen_count": 1, "outcome_score": 0, "vec": None}
    r.update(kw)
    return r


# ---- retrieval ----

def test_metadata_filter_runs_before_scoring():
    # a project note from ANOTHER workspace must be invisible, not merely
    # low-ranked: post-filtering could fill the whole top-3 with wrong-scope
    # hits and then discard them — a silent recall failure
    rows = [_row("batches of 25 into separate files", kind="project",
                 workspace="D:\\Prompts"),
            _row("mine ore with the laser first", kind="project",
                 workspace="D:\\Roblox")]
    hits = retrieve(rows, "write the prompt batches into files",
                    workspace="D:\\Prompts")
    texts = [h["text"] for h in hits]
    assert "batches of 25 into separate files" in texts
    assert "mine ore with the laser first" not in texts


def test_keyword_match_finds_the_rare_entity():
    # the rare entities ARE the memory: a step naming view_sample must pull
    # the view_sample rule even with zero embeddings available
    rows = [_row("Never type filenames; use view_sample."),
            _row("Deliverables go in files via write_file."),
            _row("Prefer q8_0 KV cache on this card.")]
    hits = retrieve(rows, "view the sampled images with view_sample")
    assert hits and hits[0]["text"].startswith("Never type filenames")


def test_retired_memories_are_never_retrieved():
    rows = [_row("Prefer f16 KV cache.", status="retired"),
            _row("Prefer q8_0 KV cache.")]
    hits = retrieve(rows, "which KV cache should be used")
    assert all(h["status"] != "retired" for h in hits)


def test_empty_query_retrieves_nothing():
    assert retrieve([_row("anything")], "") == []


# ---- conflict-gated consolidation (lexical fallback: append, never merge) ----

def test_exact_duplicate_reinforces_through_the_pipeline(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Never type filenames.", None))
    _run(add_consolidated(store, "pitfall", "Never type filenames.", None))
    rows = store.all()
    assert len(rows) == 1 and rows[0]["seen_count"] == 2


def test_without_a_gate_similar_texts_append_instead_of_merging(tmp_path):
    # similarity NOMINATES, the gate DECIDES. With no engine there is no gate,
    # and the safe direction is append — merging on similarity alone is how
    # "prefer q8_0" and "prefer f16" collapse into one row
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.", None))
    _run(add_consolidated(store, "pitfall", "Prefer the f16 KV cache.", None))
    assert len(store.all()) == 2


def test_conflict_verdict_supersedes_the_old_rule(tmp_path, monkeypatch):
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    # fake dense: make the two rules nominate each other without fastembed
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Prefer the f16 KV cache.", None))
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          _reply("CONFLICT")))
    rows = {r["text"]: r for r in store.all()}
    assert rows["Prefer the f16 KV cache."]["status"] == "retired"
    assert rows["Prefer the q8_0 KV cache."]["status"] == "draft"


def test_duplicate_verdict_reinforces_not_appends(tmp_path, monkeypatch):
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Never type filenames.", None))
    _run(add_consolidated(store, "pitfall", "Do not manually type file names.",
                          _reply("DUPLICATE")))
    rows = store.all()
    assert len(rows) == 1 and rows[0]["seen_count"] == 2


def test_gate_failure_appends_safely(tmp_path, monkeypatch):
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])

    async def boom(prompt):
        raise RuntimeError("engine down")

    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Never type filenames.", None))
    _run(add_consolidated(store, "pitfall", "Do not type file names.", boom))
    assert len(store.all()) == 2, "no verdict -> append, never merge"


def test_guard_still_applies_through_consolidation(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    with pytest.raises(ValueError):
        _run(add_consolidated(store, "pitfall",
                              r"Do not open D:\x\a.png", None))


# ---- outcome scoring + graduation ----

def test_success_credits_and_failure_discredits(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.")
    score_memories(store, [m["id"]], +1)
    score_memories(store, [m["id"]], -2)
    assert store.all()[0]["outcome_score"] == -1


def test_draft_graduates_only_in_a_different_run(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.",
                  born_run="run-A")
    score_memories(store, [m["id"]], +1, run_id="run-A")
    assert store.all()[0]["status"] == "draft", \
        "helping the run that wrote it proves nothing about generalisation"
    score_memories(store, [m["id"]], +1, run_id="run-B")
    assert store.all()[0]["status"] == "verified"


def test_failure_never_graduates(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.",
                  born_run="run-A")
    score_memories(store, [m["id"]], -2, run_id="run-B")
    assert store.all()[0]["status"] == "draft"


def test_scoring_unknown_ids_is_harmless(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    score_memories(store, ["nope"], +1)          # must not raise or write
    assert store.all() == []


# ---- trace replay: R@3 on the real 2026-07-19 failures ----

_PLAYBOOK = [
    "Never type filenames; pass files by reference via view_sample.",
    "Use sample_files for big folders, never run_shell dir.",
    "Deliverables go in files via write_file; a reply reaches nobody.",
    # distractors — plausible rules that must NOT crowd out the right one
    "Verify every plan item before calling task_complete.",
    "Keep batches to 25 prompts per file.",
    "Read the error message before retrying a failed call.",
    "Ask for guidance when the mission is ambiguous.",
    "Prefer q8_0 KV cache when VRAM is tight.",
]

_REPLAY = [
    ("view the sampled images and describe each one", "view_sample"),
    ("list example files from the big output folder", "sample_files"),
    ("produce the final prompt file as a deliverable", "write_file"),
]


@pytest.mark.parametrize("step,expect", _REPLAY)
def test_replay_r_at_3_lexical(step, expect):
    # the offline half of the evaluation harness: does querying with the real
    # step text surface the rule that would have prevented the real failure?
    # Zero inference — this can run on every commit.
    rows = [_row(t) for t in _PLAYBOOK]
    hits = retrieve(rows, step, k=3)
    assert any(expect in h["text"] for h in hits), \
        f"R@3 miss for {step!r}: {[h['text'] for h in hits]}"


@pytest.mark.skipif(
    not (os.environ.get("TEMP")
         and os.path.isdir(os.path.join(os.environ["TEMP"], "fastembed_cache",
                                        "models--nomic-ai--nomic-embed-text-v1.5"))),
    reason="nomic not cached")
@pytest.mark.parametrize("step,expect", _REPLAY)
def test_replay_r_at_3_hybrid(step, expect, monkeypatch):
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    memory._embedder, memory._embedder_tried = None, False
    rows = [_row(t, vec=memory.embed_one(t, "doc")) for t in _PLAYBOOK]
    hits = retrieve(rows, step, k=3)
    assert any(expect in h["text"] for h in hits), \
        f"hybrid R@3 miss for {step!r}: {[h['text'] for h in hits]}"
