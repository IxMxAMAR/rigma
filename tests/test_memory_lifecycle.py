# ruff: noqa: F811 — pytest fixtures used as params shadow their imports by design
"""The loop that justifies the whole feature: a failure in one run becomes a
rule the NEXT run starts with.

Unit tests cover the miner, the guard and the store in isolation. This asserts
they are actually wired to the run lifecycle — that harvesting happens at run
end and injection at run start. Both are easy to get right in isolation and
leave unconnected.
"""
import time


from rigma import memory, runs

# `home` matters as much as the rest: it is autouse INSIDE its own module only,
# so importing the others without it would point RIGMA_HOME at the real
# ~/.rigma and let these tests write to the owner's actual memory store.
from test_autonomous_run import (_Engine, _client, _wait, engine,  # noqa: F401
                                 home)


def _store():
    from rigma.runtime import rigma_home
    return memory.MemoryStore(rigma_home() / "memory" / "memories.jsonl")


def test_a_run_end_harvest_writes_a_rule(engine, monkeypatch):
    # the fake engine answers every non-streaming call with compile_reply, and
    # the distiller is a non-streaming call — so that IS the distilled rule
    _Engine.compile_reply = "Never type filenames; use view_sample."
    # a failure followed by a DIFFERENT tool succeeding — that pair is the
    # recovery event. One lone failure is neither a loop nor a recovery and
    # correctly teaches nothing.
    _Engine.script = [("view_images", {"paths": ["nope.png"]}),
                      ("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "look", "budget_hours": 1}).json()["id"]
    _wait(c, rid, timeout=25)
    # harvesting runs in the loop's finally, just after the status goes
    # terminal, so _wait can return a beat before the write lands
    for _ in range(50):
        texts = [m["text"] for m in _store().all()]
        if texts:
            break
        time.sleep(0.2)
    assert any("view_sample" in t for t in texts), texts


def test_the_next_run_starts_knowing_it(engine, monkeypatch):
    from rigma import sessions
    _store().add(kind="pitfall", text="Never type filenames; use view_sample.")
    _Engine.compile_reply = "not a spec"
    _Engine.script = [("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    prompt = sessions.load(r["session_id"]).get("system_prompt", "")
    assert "Never type filenames" in prompt
    assert "WHAT YOU LEARNED BEFORE" in prompt


def test_memory_can_be_switched_off(engine, monkeypatch):
    from rigma import sessions
    monkeypatch.setenv("RIGMA_MEMORY", "0")
    _store().add(kind="pitfall", text="Never type filenames; use view_sample.")
    _Engine.compile_reply = "not a spec"
    _Engine.script = [("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    prompt = sessions.load(r["session_id"]).get("system_prompt", "")
    assert "Never type filenames" not in prompt


def test_an_unwritable_store_does_not_break_a_run(engine, monkeypatch):
    # memory must never be load-bearing. Earlier the RAG sidecar 500'd because
    # a package was missing; if memory were on the critical path that would
    # have killed every run.
    def boom(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(memory.MemoryStore, "all", boom)
    _Engine.compile_reply = "not a spec"
    _Engine.script = [("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    assert r["status"] in runs.TERMINAL, r["status"]


def test_a_reboot_does_not_strand_the_active_run(engine):
    # local review, critical: shutdown only REQUESTS cancellation and there
    # was no startup reconciliation, so a reboot mid-run left active.json at
    # status=running forever — and start_run 409s while a run is active, so
    # one Windows update permanently disabled autonomous mode.
    _Engine.compile_reply = "not a spec"
    _Engine.script = [("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    _wait(c, rid, timeout=25)      # let the first loop actually END —
    # in production a reboot kills it; in this shared-process test it would
    # otherwise keep driving the run and overwrite the state we forge next
    r = runs.load(rid)
    r["status"] = "running"        # forge what a crash leaves behind
    runs.save(r)
    import json as _json
    runs._active_path().write_text(_json.dumps({"id": rid}), encoding="utf-8")
    # a NEW app instance boots (the TestClient context manager runs startup)
    c2 = _client(engine)
    r2 = c2.post("/api/runs", json={"mission": "y", "budget_hours": 1})
    assert r2.status_code == 200, r2.json()
    assert runs.load(rid)["status"] == "stopped"


def test_delegate_runs_in_a_fresh_context_and_returns_one_answer(engine):
    # the context firewall: the helper's exploration must NOT enter the main
    # session — only its condensed answer does. The fake engine answers every
    # non-streaming call with compile_reply, which here plays the helper's
    # final answer.
    from rigma import sessions
    _Engine.compile_reply = "The files use ComfyUI_%05d_.png naming."
    _Engine.script = [("delegate", {"question": "what naming scheme?"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "investigate naming",
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    blob = "\n".join(str(m.get("content", "")) for m in
                     sessions.load(r["session_id"])["messages"])
    assert "TOOL RESULT delegate: The files use ComfyUI_" in blob
    # the helper conversation itself (system prompt etc.) must never leak in
    assert "research helper" not in blob


def test_per_step_recall_injects_notes_once_per_step(engine):
    import json as _json
    from rigma import sessions
    _store().add(kind="pitfall",
                 text="Never type filenames; use view_sample.")
    _Engine.compile_reply = _json.dumps({
        "objective": "look at images", "deliverables": [], "constraints": [],
        "steps": [{"id": 1, "description": "view the sampled images",
                   "artifact": "", "verification": {"type": "none"}}]})
    _Engine.script = [("current_datetime", {}), ("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "look at images",
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    driving = [str(m.get("content", "")) for m in
               sessions.load(r["session_id"])["messages"]
               if m.get("role") == "user"]
    notes = [d for d in driving if "### NOTES" in d]
    assert notes, "step-matched memory should be recalled into the driving line"
    assert "view_sample" in notes[0]
    # once per STEP, not per turn — per-turn injection caused the narration loop
    assert len(notes) == 1, f"injected {len(notes)} times for one step"
