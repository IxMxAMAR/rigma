"""Agent memory, phase 1: event mining, the anchoring guard, and the store.

The miner is deliberately dumb — it detects EVENTS and never writes the rule.
A deterministic extractor over {tool, args, ok} can only emit literal strings,
so left to itself it would learn "ComfyUI_00428_.png failed" (a fact about one
file that will never recur) instead of "never retype filenames". Abstraction is
the distiller's job.

Fixtures below are the real 2026-07-19 failures.
"""
import json

import pytest

from rigma.memory import (MemoryStore, looks_like_raw_trace, mine_events,
                          render_pitfall_block)


def _a(tool, args, ok=True):
    return {"ts": 0, "tool": tool, "args": json.dumps(args), "ok": ok}


# ---- the miner detects events, and only events ----

def test_repeated_identical_failure_is_a_loop_event():
    # the filename retype loop: three attempts, three different wrong names
    actions = [
        _a("view_images", {"paths": ["D:\\Good Stuff\\Comfy_UI_428.png"]}, ok=False),
        _a("view_images", {"paths": ["D:\\Good Stuff\\Comfy_UI_428.png"]}, ok=False),
    ]
    events = mine_events(actions)
    assert any(e["kind"] == "loop" and e["tool"] == "view_images" for e in events)


def test_failure_then_recovery_is_a_recovery_event():
    # what actually fixed it: stop retyping, use the reference tool
    actions = [
        _a("view_images", {"paths": ["D:\\x\\Comfy_UI_428.png"]}, ok=False),
        _a("view_sample", {}, ok=True),
    ]
    events = mine_events(actions)
    rec = [e for e in events if e["kind"] == "recovery"]
    assert rec, events
    assert rec[0]["failed_tool"] == "view_images"
    assert rec[0]["worked_tool"] == "view_sample"


def test_a_clean_run_produces_no_events():
    actions = [_a("read_file", {"path": "a.md"}), _a("write_file", {"path": "b.md"})]
    assert mine_events(actions) == []


def test_recovery_by_the_same_tool_is_not_an_event():
    # retrying the same tool with different args and succeeding is ordinary
    # work, not a lesson worth storing
    actions = [_a("read_file", {"path": "a"}, ok=False),
               _a("read_file", {"path": "b"}, ok=True)]
    assert not [e for e in mine_events(actions) if e["kind"] == "recovery"]


def test_a_failure_rescued_much_later_is_not_attributed():
    # 14 actions apart is coincidence, not causation
    actions = ([_a("view_images", {"p": 1}, ok=False)]
               + [_a("find_files", {"n": i}, ok=False) for i in range(14)]
               + [_a("view_sample", {}, ok=True)])
    rec = [e for e in mine_events(actions) if e["kind"] == "recovery"]
    # the find_files failures right before the success ARE genuine recoveries;
    # the view_images failure 15 actions earlier is not, and is the claim here
    assert not any(e["failed_tool"] == "view_images" for e in rec), rec


def test_an_unrelated_success_is_not_a_recovery():
    # the over-detection this nearly shipped with: ANY different tool
    # succeeding near a failure looked like a fix. In a real run that welds
    # unrelated work onto an old failure and invents a lesson from nothing.
    actions = [_a("view_images", {"p": 1}, ok=False),
               _a("read_file", {"path": "notes"}, ok=True),
               _a("view_sample", {}, ok=True)]
    rec = [e for e in mine_events(actions) if e["kind"] == "recovery"]
    assert len(rec) == 1
    assert rec[0]["worked_tool"] == "read_file", \
        "the FIRST success decides; later ones are separate work"


def test_events_carry_the_raw_args_for_the_distiller():
    # the miner's whole job is to bound what the distiller may talk about, so
    # the event itself must keep the evidence
    actions = [_a("view_images", {"paths": ["D:\\x\\Comfy_UI_428.png"]}, ok=False),
               _a("view_sample", {}, ok=True)]
    ev = [e for e in mine_events(actions) if e["kind"] == "recovery"][0]
    assert "Comfy_UI_428" in json.dumps(ev)


# ---- the anchoring guard ----

@pytest.mark.parametrize("bad", [
    "tried D:\\Good Stuff\\Comfy_UI_428.png and it failed",
    "view_images(paths=['C:/x/ComfyUI_00428_.png']) returned no such file",
    "ComfyUI_00428_.png could not be opened",
])
def test_guard_rejects_raw_traces(bad):
    # a model that reads a transcript of a failing agent SIMULATES a failing
    # agent. Raw traces prime repetition; only the distilled rule helps.
    assert looks_like_raw_trace(bad), bad


@pytest.mark.parametrize("good", [
    "Never type filenames. Pass files by reference via view_sample.",
    "Use sample_files for folders over ~200 entries, never run_shell dir.",
    "Deliverables go in files via write_file; a reply reaches nobody.",
])
def test_guard_allows_distilled_imperatives(good):
    assert not looks_like_raw_trace(good), good


def test_store_refuses_a_pitfall_containing_a_raw_path(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    with pytest.raises(ValueError):
        store.add(kind="pitfall", text="D:\\x\\ComfyUI_00428_.png failed twice")
    assert store.all() == []


def test_project_memories_may_name_files(tmp_path):
    # naming an artifact IS the content of a project memory; the guard is for
    # behavioural rules, where a literal path is always noise
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="project", text="Core Directive lives in Core_Directive.md")
    assert len(store.all()) == 1


# ---- the store ----

def test_memories_start_as_drafts(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames; use view_sample.")
    assert m["status"] == "draft"


def test_store_survives_a_reload(tmp_path):
    p = tmp_path / "m.jsonl"
    MemoryStore(p).add(kind="pitfall", text="Never type filenames.")
    assert len(MemoryStore(p).all()) == 1


def test_exact_duplicates_reinforce_instead_of_appending(tmp_path):
    # phase 1 has no embeddings; exact-text dedup is the honest floor and keeps
    # a 20-hour run from writing the same lesson 40 times
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    store.add(kind="pitfall", text="Never type filenames.")
    assert len(store.all()) == 1
    assert store.all()[0]["seen_count"] == 2


def test_a_corrupt_line_does_not_kill_the_store(tmp_path):
    # memory is NEVER load-bearing: a run must not fail because memory failed
    p = tmp_path / "m.jsonl"
    MemoryStore(p).add(kind="pitfall", text="Never type filenames.")
    with open(p, "a", encoding="utf-8") as f:
        f.write("{ this is not json\n")
    assert len(MemoryStore(p).all()) == 1


# ---- run-start injection ----

def test_pitfall_block_is_terse_imperatives(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Never type filenames; use view_sample.")
    block = render_pitfall_block(store.all())
    assert "Never type filenames" in block
    assert len(block.splitlines()) <= 8, "anything discursive gets narrated back"


def test_empty_store_renders_nothing(tmp_path):
    # no memories must mean no block at all, not an empty header taking up
    # context and inviting the model to comment on it
    assert render_pitfall_block([]) == ""


def test_drafts_are_shown_but_not_hedged(tmp_path):
    # drafts ARE pinned (nothing can graduate until phase 3), but WITHOUT the
    # "UNVERIFIED:" prefix an earlier revision used: under a header saying
    # "these are rules, not suggestions" the hedge is a contradiction a weak
    # model resolves badly. The store still tracks status for phase 3; the
    # bounded cap is the quarantine until then.
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.")
    assert m["status"] == "draft"                  # data model unchanged
    block = render_pitfall_block(store.all())
    assert "Never type filenames" in block
    assert "UNVERIFIED" not in block


def test_verified_memories_are_not_hedged(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.")
    m["status"] = "verified"
    assert "UNVERIFIED" not in render_pitfall_block([m])


def test_verified_outrank_drafts(tmp_path):
    drafted = {"kind": "pitfall", "text": "hunch", "status": "draft",
               "seen_count": 9, "outcome_score": 0}
    proven = {"kind": "pitfall", "text": "proven rule", "status": "verified",
              "seen_count": 1, "outcome_score": 5}
    block = render_pitfall_block([drafted, proven])
    assert block.index("proven rule") < block.index("hunch")


# ---- distillation ----

import asyncio   # noqa: E402

from rigma.memory import distil, harvest_run   # noqa: E402


def _run(coro):
    """distil/harvest_run are async because the engine call is. asyncio.run
    keeps these tests plugin-free."""
    return asyncio.run(coro)


def _reply(text):
    """An async completer that always answers `text`."""
    async def _c(prompt):
        return text
    return _c


def test_distil_asks_for_a_generalised_rule():
    seen = {}

    async def fake(prompt):
        seen["prompt"] = prompt
        return "Never type filenames; pass files by reference with view_sample."

    ev = mine_events([_a("view_images", {"p": r"D:\x\a_00428_.png"}, ok=False),
                      _a("view_sample", {}, ok=True)])[0]
    rule = _run(distil(ev, fake))
    assert rule.startswith("Never type filenames")
    assert "NEVER mention specific filenames" in seen["prompt"]
    assert "a_00428_" in seen["prompt"], "the distiller needs the evidence"


def test_distiller_failure_is_silent():
    # memory is never load-bearing
    async def boom(prompt):
        raise RuntimeError("engine down")

    ev = {"kind": "loop", "tool": "x", "args": "{}", "count": 2}
    assert _run(distil(ev, boom)) == ""


def test_harvest_drops_rules_that_leak_a_path(tmp_path):
    # the distiller runs on a quantised local model and WILL sometimes echo the
    # path back. The guard is the backstop, and a dropped rule beats a stored
    # trace that teaches the failure.
    store = MemoryStore(tmp_path / "m.jsonl")
    actions = [_a("view_images", {"p": r"D:\x\a.png"}, ok=False),
               _a("view_sample", {}, ok=True)]
    written = _run(harvest_run(actions, store,
                               _reply(r"Do not open D:\x\a.png")))
    assert written == []
    assert store.all() == []


def test_harvest_stores_a_clean_rule(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    actions = [_a("view_images", {"p": r"D:\x\a.png"}, ok=False),
               _a("view_sample", {}, ok=True)]
    written = _run(harvest_run(actions, store,
                               _reply("Never retype filenames.")))
    assert len(written) == 1
    assert store.all()[0]["text"] == "Never retype filenames."


def test_harvest_is_bounded(tmp_path):
    # a badly broken run emits dozens of events; one rule each would swamp the
    # store with near-duplicates
    store = MemoryStore(tmp_path / "m.jsonl")
    actions = []
    for i in range(20):
        actions += [_a("view_images", {"p": i}, ok=False), _a("view_sample", {})]
    n = [0]

    async def fake(p):
        n[0] += 1
        return f"Rule number {n[0]}."

    _run(harvest_run(actions, store, fake, max_rules=3))
    assert len(store.all()) <= 3


def test_harvest_never_raises_on_a_broken_trace(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    assert _run(harvest_run([{"garbage": True}], store, _reply("x"))) == []


# ---- review fixes (2026-07-20 adversarial review) ----

def test_store_writes_are_atomic(tmp_path):
    # write_text truncates in place: a crash mid-write 19 hours into a run
    # would zero the store. The fix writes a sibling tmp file and os.replace()s
    # it — the store is always either the old rows or the new rows.
    p = tmp_path / "m.jsonl"
    store = MemoryStore(p)
    store.add(kind="pitfall", text="Never type filenames.")
    store.add(kind="pitfall", text="Deliverables go in files.")
    assert not p.with_suffix(".tmp").exists(), "tmp must not linger"
    assert len(MemoryStore(p).all()) == 2


def test_guard_rejection_is_logged_not_silent(tmp_path, caplog):
    # a swallowed NameError already cost a whole session once. Failures stay
    # non-fatal but must be OBSERVABLE.
    import logging
    store = MemoryStore(tmp_path / "m.jsonl")
    actions = [_a("view_images", {"p": r"D:\x\a.png"}, ok=False),
               _a("view_sample", {}, ok=True)]
    with caplog.at_level(logging.INFO, logger="rigma.memory"):
        _run(harvest_run(actions, store, _reply(r"Do not open D:\x\a.png")))
    assert any("guard rejected" in r.message for r in caplog.records)


def test_pitfall_store_is_bounded(tmp_path):
    # the distiller at nonzero temperature paraphrases the same lesson
    # differently every run — exact dedup cannot catch that, so the cap is
    # what stops 50 runs producing 150 near-duplicate rules
    from rigma.memory import MAX_PITFALLS
    store = MemoryStore(tmp_path / "m.jsonl")
    for i in range(MAX_PITFALLS + 10):
        store.add(kind="pitfall", text=f"Distinct paraphrase number {i}.")
    pits = [m for m in store.all() if m["kind"] == "pitfall"]
    assert len(pits) <= MAX_PITFALLS


def test_eviction_spares_the_proven_rule(tmp_path):
    from rigma.memory import MAX_PITFALLS
    store = MemoryStore(tmp_path / "m.jsonl")
    for _ in range(3):   # reinforce one rule so it is provably not junk
        store.add(kind="pitfall", text="Never type filenames.")
    for i in range(MAX_PITFALLS + 5):
        store.add(kind="pitfall", text=f"One-off hunch {i}.")
    texts = [m["text"] for m in store.all()]
    assert "Never type filenames." in texts


def test_pinned_rules_carry_no_epistemic_hedge(tmp_path):
    # "these are rules, not suggestions" + "UNVERIFIED:" is a contradiction a
    # weak model resolves badly — ignore the hedge, narrate it, or TEST the
    # unverified rule. What gets pinned gets committed to.
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    assert "UNVERIFIED" not in render_pitfall_block(store.all())
