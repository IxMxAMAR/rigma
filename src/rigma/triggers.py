"""Trigger rules: the automatic half of a Method's rules.

Pure by design -- evaluate() takes a method, an event and a state, and
returns actions plus the next state. No IO, no session, no server, so the
loop guard can be tested exhaustively without any of them.

THE LOOP GUARD IS THE POINT. A rule whose action causes the event it watches
is an infinite loop that spends the owner's tokens and, worse, writes to
their files unattended. Three independent guards, each with its own test in
tests/test_triggers.py:

  1. an event caused BY a trigger action never fires a trigger
  2. at most MAX_FIRES_PER_TURN fires in any one turn
  3. a trigger that fires MUTE_AFTER turns running with no user input is
     muted until the user speaks again -- and the muting is announced

Spec §4.3: docs/superpowers/specs/2026-07-21-custom-methods-design.md
"""
from __future__ import annotations

import fnmatch

MAX_FIRES_PER_TURN = 3
MUTE_AFTER = 5

EVENTS = ("tool_ran", "turn_ended", "every_n_turns", "method_applied")


def new_state() -> dict:
    """Per-session trigger bookkeeping. JSON-safe: it rides on the session."""
    return {"turn": 0, "fires_this_turn": 0, "consecutive": {}, "muted": []}


def _matches(on: dict, ev: dict) -> bool:
    want = on.get("event")
    if want == "tool_ran":
        if ev.get("kind") != "tool_ran":
            return False
        if on.get("tool") and on["tool"] != ev.get("tool"):
            return False
        glob = on.get("path_glob")
        if glob and not fnmatch.fnmatch(str(ev.get("path") or ""), glob):
            return False
        return True
    if want == "turn_ended":
        return ev.get("kind") == "turn_ended"
    if want == "every_n_turns":
        n = int(on.get("n", 0) or 0)
        return (ev.get("kind") == "turn_ended" and n >= 1
                and int(ev.get("turn", 0)) > 0
                and int(ev.get("turn", 0)) % n == 0)
    if want == "method_applied":
        return ev.get("kind") == "method_applied"
    return False


def evaluate(method: dict, event: dict, state: dict) -> tuple[list[dict], dict]:
    """(actions, next_state). Never mutates `state`.

    An action is {"rule": id, "mode": "nudge"|"run", "text"/"macro": ...}
    plus an optional "notice" the caller must surface as msg["notice"] --
    NEVER as assistant content, or the model reads its own housekeeping back
    as precedent (the 2026-07-21 instant-EOS bug).
    """
    st = {"turn": state.get("turn", 0),
          "fires_this_turn": state.get("fires_this_turn", 0),
          "consecutive": dict(state.get("consecutive") or {}),
          "muted": list(state.get("muted") or [])}

    turn = int(event.get("turn", 0) or 0)
    if turn != st["turn"]:
        st["turn"] = turn
        st["fires_this_turn"] = 0

    # guard 3, release half: the user is back, so nothing is runaway any more
    if event.get("user_spoke"):
        st["muted"] = []
        st["consecutive"] = {}

    # guard 1: never fire from an event our own action produced
    if event.get("by_trigger"):
        return [], st

    actions: list[dict] = []
    for rule in method.get("rules") or []:
        if rule.get("kind") != "trigger":
            continue
        rid = str(rule.get("id") or "")
        if rid in st["muted"]:
            continue
        if not _matches(rule.get("on") or {}, event):
            continue
        # guard 2
        if st["fires_this_turn"] >= MAX_FIRES_PER_TURN:
            break
        do = rule.get("do") or {}
        act = {"rule": rid, "mode": do.get("mode", "nudge")}
        if act["mode"] == "run":
            act["macro"] = do.get("macro", "")
        else:
            act["text"] = str(do.get("text") or "")
        st["fires_this_turn"] += 1

        # guard 3, arm half: count unattended fires and cut it off
        if not event.get("user_spoke"):
            n = st["consecutive"].get(rid, 0) + 1
            st["consecutive"][rid] = n
            if n >= MUTE_AFTER:
                st["muted"].append(rid)
                act["notice"] = (
                    f"Rule '{rid}' fired {n} turns running with no input "
                    "from you, so it is muted until you send a message.")
        actions.append(act)
    return actions, st
