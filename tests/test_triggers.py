"""Trigger rules fire, and cannot run away (spec §4.3).

The loop guard is the load-bearing part. A rule whose action causes the very
event it watches is an infinite loop that costs the owner tokens and, worse,
writes to their files unattended -- so all three guards get their own test.
"""
from rigma import triggers


def _method(**over):
    m = {"id": "book", "macros": [{"id": "recap", "label": "Recap",
                                   "steps": []}],
         "rules": [{"id": "r1", "kind": "trigger",
                    "on": {"event": "tool_ran", "tool": "write_file",
                           "path_glob": "chapters/*"},
                    "do": {"mode": "nudge", "text": "update the bible"}}]}
    m.update(over)
    return m


def _ev(**over):
    e = {"kind": "tool_ran", "tool": "write_file",
         "path": "chapters/ch1.md", "turn": 1, "by_trigger": False,
         "user_spoke": True}
    e.update(over)
    return e


def _state(**over):
    s = triggers.new_state()
    s.update(over)
    return s


def test_a_matching_tool_ran_fires_a_nudge():
    actions, _ = triggers.evaluate(_method(), _ev(), _state())
    assert len(actions) == 1
    assert actions[0]["mode"] == "nudge"
    assert actions[0]["text"] == "update the bible"


def test_a_different_tool_does_not_fire():
    actions, _ = triggers.evaluate(_method(), _ev(tool="read_file"),
                                   _state())
    assert actions == []


def test_the_path_glob_is_respected():
    actions, _ = triggers.evaluate(_method(), _ev(path="notes/x.md"),
                                   _state())
    assert actions == []


def test_a_glob_free_trigger_matches_any_path():
    m = _method(rules=[{"id": "r", "kind": "trigger",
                        "on": {"event": "tool_ran", "tool": "write_file"},
                        "do": {"mode": "nudge", "text": "hi"}}])
    actions, _ = triggers.evaluate(m, _ev(path="anywhere.txt"), _state())
    assert len(actions) == 1


def test_turn_ended_fires_on_every_turn():
    m = _method(rules=[{"id": "r", "kind": "trigger",
                        "on": {"event": "turn_ended"},
                        "do": {"mode": "nudge", "text": "tick"}}])
    actions, _ = triggers.evaluate(m, _ev(kind="turn_ended"), _state())
    assert len(actions) == 1


def test_every_n_turns_fires_only_on_multiples():
    m = _method(rules=[{"id": "r", "kind": "trigger",
                        "on": {"event": "every_n_turns", "n": 3},
                        "do": {"mode": "nudge", "text": "checkpoint"}}])
    fired = [bool(triggers.evaluate(m, _ev(kind="turn_ended", turn=t),
                                    _state())[0])
             for t in (1, 2, 3, 4, 5, 6)]
    assert fired == [False, False, True, False, False, True]


def test_run_mode_yields_a_macro_action():
    m = _method(rules=[{"id": "r", "kind": "trigger",
                        "on": {"event": "turn_ended"},
                        "do": {"mode": "run", "macro": "recap"}}])
    actions, _ = triggers.evaluate(m, _ev(kind="turn_ended"), _state())
    assert actions[0]["mode"] == "run" and actions[0]["macro"] == "recap"


def test_standing_rules_are_never_triggers():
    m = _method(rules=[{"id": "s", "kind": "standing", "text": "be brief"}])
    actions, _ = triggers.evaluate(m, _ev(kind="turn_ended"), _state())
    assert actions == []


# --- loop guard 1: a trigger never fires from its own action ---------------

def test_a_trigger_does_not_fire_on_its_own_action():
    actions, _ = triggers.evaluate(_method(), _ev(by_trigger=True), _state())
    assert actions == []


# --- loop guard 2: at most 3 fires per turn --------------------------------

def test_fires_are_capped_at_three_per_turn():
    st = _state()
    got = []
    for _ in range(6):
        actions, st = triggers.evaluate(_method(), _ev(), st)
        got.extend(actions)
    assert len(got) == triggers.MAX_FIRES_PER_TURN == 3


def test_the_per_turn_cap_resets_on_a_new_turn():
    st = _state()
    for _ in range(5):
        _, st = triggers.evaluate(_method(), _ev(turn=1), st)
    actions, st = triggers.evaluate(_method(), _ev(turn=2), st)
    assert len(actions) == 1


# --- loop guard 3: mute a trigger that fires with no user input ------------

def test_a_trigger_firing_without_the_user_gets_muted():
    st = _state()
    muted_at = None
    for turn in range(1, 12):
        actions, st = triggers.evaluate(
            _method(), _ev(turn=turn, user_spoke=False), st)
        if not actions and muted_at is None:
            muted_at = turn
            break
    assert muted_at is not None, "it never muted — that is a runaway loop"
    assert st["muted"] == ["r1"]


def test_muting_is_reported_so_the_user_is_told():
    st = _state()
    notices = []
    for turn in range(1, 12):
        actions, st = triggers.evaluate(
            _method(), _ev(turn=turn, user_spoke=False), st)
        notices.extend(a["notice"] for a in actions if a.get("notice"))
        if st["muted"]:
            break
    assert any("muted" in n.lower() for n in notices), notices


def test_a_user_message_unmutes():
    st = _state(muted=["r1"], consecutive={"r1": 9})
    actions, st = triggers.evaluate(_method(), _ev(user_spoke=True), st)
    assert st["muted"] == []
    assert len(actions) == 1


def test_state_is_json_safe():
    import json
    _, st = triggers.evaluate(_method(), _ev(), _state())
    assert json.loads(json.dumps(st)) == st


def test_evaluate_never_mutates_the_state_it_was_given():
    st = _state()
    before = dict(st)
    triggers.evaluate(_method(), _ev(), st)
    assert st == before


def test_a_method_without_rules_is_cheap_and_silent():
    actions, st = triggers.evaluate({"id": "x"}, _ev(), _state())
    assert actions == [] and st["muted"] == []


# --- the serve.py hook, end to end ----------------------------------------

def test_a_write_to_a_chapter_queues_the_bible_nudge(tmp_path, monkeypatch):
    """The book method ships exactly this trigger. Drive serve.py's own
    hook rather than trusting that the wiring matches the unit tests."""
    import os
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from fastapi.testclient import TestClient  # noqa: F401
    from rigma import sessions
    from rigma import state as st
    from rigma.serve import build_app

    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    app = build_app(upstream_port=1)
    fire = app.state.fire_triggers            # exported for exactly this

    s = sessions.create("Chapter 1")
    s["method"] = "book"
    trace = [{"name": "write_file", "args": {"path": "chapters/ch1.md"},
              "result": "wrote 10 chars"}]
    notices = fire(s, trace, user_spoke=True)
    assert notices == []
    assert any("STORY SO FAR" in n for n in s["pending_nudges"]), \
        s["pending_nudges"]


def test_a_write_somewhere_else_queues_nothing(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from fastapi.testclient import TestClient  # noqa: F401
    from rigma import sessions
    from rigma import state as st
    from rigma.serve import build_app

    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    fire = build_app(upstream_port=1).state.fire_triggers
    s = sessions.create("Chapter 1")
    s["method"] = "book"
    fire(s, [{"name": "write_file", "args": {"path": "notes/todo.md"},
              "result": "ok"}], user_spoke=True)
    assert s["pending_nudges"] == []


def test_a_method_with_no_triggers_stays_untouched(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import sessions
    from rigma import state as st
    from rigma.serve import build_app

    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    fire = build_app(upstream_port=1).state.fire_triggers
    s = sessions.create("t")
    s["method"] = "coding"
    assert fire(s, [{"name": "write_file", "args": {"path": "a.py"}}],
                user_spoke=True) == []
    assert s.get("pending_nudges", []) == []
