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


def test_drafts_are_labelled_unverified(tmp_path):
    # nothing graduates in phase 1 (promotion needs phase 3's outcome
    # tracking), so drafts ARE shown — but always hedged. A labelled hint is
    # not a foundation, which is what the quarantine protects against.
    store = MemoryStore(tmp_path / "m.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    assert "UNVERIFIED" in render_pitfall_block(store.all())


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
