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


# ---- live run #3 findings ----

def test_anchor_strips_hallucinated_absolute_artifacts():
    # the compiler wrote C:\workspace\naming.md — a folder that does not
    # exist — because its example showed absolute paths and a weak model
    # imitates the example over the rule. Wrong path -> server could never
    # verify -> plan never advanced -> memories never credited.
    from rigma.mission import anchor_spec
    spec = {"steps": [{"id": 1, "artifact": r"C:\workspace\naming.md"}],
            "deliverables": [{"path": r"C:\workspace\count.md"}]}
    out = anchor_spec(spec)
    assert out["steps"][0]["artifact"] == "naming.md"
    assert out["deliverables"][0]["path"] == "count.md"


def test_anchor_keeps_real_absolute_paths(tmp_path):
    # a user who NAMES a real folder means it
    from rigma.mission import anchor_spec
    real = str(tmp_path / "out.md")
    spec = {"steps": [{"id": 1, "artifact": real}], "deliverables": []}
    assert anchor_spec(spec)["steps"][0]["artifact"] == real


def test_model_driven_completion_also_scores(tmp_path, monkeypatch):
    # scoring rode only the server-advance path; live run #3 completed its
    # steps through manage_plan and the injected memories sat at -4 while the
    # run succeeded around them
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_MEMORY", raising=False)
    from rigma import runs as _r
    from rigma import tools as _t
    store = MemoryStore(tmp_path / "memory" / "memories.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.",
                  born_run="other-run")
    run = _r.create("m", "s")
    _r.plan_add(run["id"], "do the thing")
    run["step_injected"] = {"1": [m["id"]]}
    _r.save(run)
    out = _t.run_tool("manage_plan", {"action": "complete", "id": 1},
                      {"run_id": run["id"], "workspace": str(tmp_path)})
    assert "marked done" in out
    row = store.all()[0]
    assert row["outcome_score"] == 1
    assert row["status"] == "verified", \
        "+1 from a different run than born_run must graduate the draft"


# ---- final-review fixes (2026-07-21) ----

def test_unrelated_dense_matches_fall_below_the_floor(monkeypatch):
    # anisotropy: unrelated sentences score ~0.40-0.45 raw cosine, so an
    # uncorrected 0.5*cos put EVERY memory over the floor once dense was
    # active — the filter passed garbage which outcome scoring then falsely
    # punished. The baseline subtraction makes "unrelated" read as zero.
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    import math
    def fake_embed(text, purpose="doc"):
        # unrelated pair at the measured ambient cosine (~0.43)
        return [1.0, 0.0] if purpose == "query" else None
    monkeypatch.setattr(memory, "embed_one", fake_embed)
    amb = 0.43
    row = _row("Prefer q8_0 KV cache.", vec=[amb, math.sqrt(1 - amb * amb)])
    hits = retrieve([row], "write a poem about autumn leaves")
    assert hits == [], "ambient-cosine noise must not be retrieved"


def test_related_dense_matches_still_pass(monkeypatch):
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    import math
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    row = _row("Prefer q8_0 KV cache.", vec=[0.62, math.sqrt(1 - 0.62**2)])
    # cos = 0.62 -> rescaled (0.62-0.40)/0.60 = 0.367 -> 0.5*0.367 = 0.18 > 0.12
    hits = retrieve([row], "no lexical overlap whatsoever here")
    assert hits, "a genuinely related memory must survive the correction"


def test_padded_conflict_verdicts_are_understood(tmp_path, monkeypatch):
    # 'Answer: CONFLICT' failed startswith and silently defaulted to DISTINCT,
    # bypassing consolidation entirely
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Prefer the f16 KV cache.", None))
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          _reply("Answer: CONFLICT")))
    rows = {r["text"]: r for r in store.all()}
    assert rows["Prefer the f16 KV cache."]["status"] == "retired"


def test_a_verdict_naming_both_words_appends(tmp_path, monkeypatch):
    # ambiguity must fall to the recoverable side: append, never merge/retire
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    store = MemoryStore(tmp_path / "m.jsonl")
    _run(add_consolidated(store, "pitfall", "Prefer the f16 KV cache.", None))
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          _reply("Could be CONFLICT or DUPLICATE honestly")))
    assert len([r for r in store.all() if r["status"] != "retired"]) == 2


def test_conflict_demotes_verified_rules_instead_of_killing_them(tmp_path,
                                                                 monkeypatch):
    # a wrong CONFLICT verdict against a VERIFIED rule would trade proven
    # capability for an untested draft — the worst possible error. Verified
    # rules are demoted (must re-earn status), only drafts are retired.
    monkeypatch.delenv("RIGMA_MEMORY_EMBED", raising=False)
    monkeypatch.setattr(memory, "embed_one",
                        lambda text, purpose="doc": [1.0, 0.0])
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Prefer the f16 KV cache.")
    rows = store.all()
    rows[0]["status"] = "verified"
    store._write_all(rows)
    _run(add_consolidated(store, "pitfall", "Prefer the q8_0 KV cache.",
                          _reply("CONFLICT")))
    rows = {r["text"]: r for r in store.all()}
    assert rows["Prefer the f16 KV cache."]["status"] == "draft", \
        "demoted, not retired"


def test_clean_rule_skips_chatty_preambles():
    from rigma.memory import clean_rule
    assert clean_rule("Here is the rule:\nNever type filenames manually.") \
        == "Never type filenames manually."
    assert clean_rule("Sure! The rule is:\n\nUse sample_files for big folders.") \
        == "Use sample_files for big folders."
    assert clean_rule("Never type filenames.") == "Never type filenames."
    assert clean_rule('"Answer briefly."') == "Answer briefly."
