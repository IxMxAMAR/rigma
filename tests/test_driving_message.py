"""The driving line — the one user-role message the model gets every turn.

Every bug here was watched happening in a live run on 2026-07-20: the plan
was 3/3 done, and the line kept ordering the model to perform a step that no
longer existed. It invented a step #4 to have something to obey, got told off
for inventing it, then emitted a bare tool name as prose twice in a row while
receiving the identical correction both times.

These are pure functions of (run, session), so none of this needs an engine.
"""
from rigma import runs as _runs
from rigma.serve import _driving_message


def _run(mission="m", **kw):
    r = _runs.create(mission, "sess-1")
    r.update(kw)
    # explicit, not setdefault: create() already sets iteration=0, and 0 routes
    # to the "New mission — build a plan first" branch, which would make these
    # tests pass without ever reaching the code under test
    r["iteration"] = kw.get("iteration", 3)
    _runs.save(r)
    return r


def _session(trace=None):
    return {"messages": [{"role": "assistant", "content": "",
                          "tool_trace": trace or []}]}


def _plan(run, *steps):
    """steps: (text, status) pairs. The plan is its own file, not run['plan']."""
    _runs.write_plan(run["id"], [{"id": i, "text": t, "status": s}
                                 for i, (t, s) in enumerate(steps, 1)])
    return run


# ---- bug 1: the plan is exhausted and the line still says "Do this now" ----

def test_exhausted_plan_does_not_order_a_nonexistent_step():
    # 3/3 done. next_pending() returns nothing, and the old code interpolated
    # the placeholder "(no pending steps — verify and finish)" straight into
    # "Do this now: {}", ordering the model to perform a step that is not there.
    run = _plan(_run(), ("a", "done"), ("b", "done"), ("c", "done"))
    msg = _driving_message(run, _session([{"name": "read_file", "result": "ok"}]))
    assert "Do this now: (no pending" not in msg
    assert "no pending steps" not in msg.lower() or "verify" in msg.lower()


def test_exhausted_plan_routes_to_verify_and_finish():
    run = _plan(_run(), ("a", "done"))
    msg = _driving_message(run, _session([{"name": "read_file", "result": "ok"}]))
    low = msg.lower()
    assert "verify" in low or "task_complete" in low, msg


# ---- bug 2: the bookkeeping nudge interpolated the placeholder ----

def test_bookkeeping_nudge_never_says_continue_with_a_placeholder():
    # the model literally read: "Continue with: (no pending steps — verify and
    # finish)". Adding a step when the plan was empty was a REASONABLE reply to
    # an incoherent instruction; the nudge then punished it for that.
    run = _plan(_run(), ("a", "done"))
    msg = _driving_message(run, _session([{"name": "manage_plan", "result": "added"}]))
    assert "Continue with: (no pending" not in msg
    assert "(no pending steps" not in msg


def test_bookkeeping_nudge_still_fires_when_real_work_is_pending():
    # the nudge itself is correct and must survive the fix
    run = _plan(_run(), ("a", "done"), ("write the notes", "pending"))
    msg = _driving_message(run, _session([{"name": "manage_plan", "result": "added"}]))
    assert "NOTHING" in msg
    assert "write the notes" in msg


# ---- bug 3: an identical nudge repeated verbatim after it had just failed ----

def test_repeated_echo_nudge_escalates_instead_of_repeating():
    run = _run()
    run["_echoed_tool"] = "view_sample"
    first = _driving_message(run, _session())

    run["_echoed_tool"] = "view_sample"          # it did it again
    second = _driving_message(run, _session())

    assert first != second, ("re-sending the identical correction that just "
                             "failed is a loop by construction")


def test_echo_streak_resets_after_the_model_recovers():
    # without a reset the run stays permanently in "escalated" mode, and a
    # single stumble 40 turns ago keeps shouting at a model that is fine now
    run = _plan(_run(), ("write the notes", "pending"))
    run["_echoed_tool"] = "view_sample"
    _driving_message(run, _session())                      # streak -> 1
    _driving_message(run, _session([{"name": "read_file", "result": "ok"}]))
    assert "_echo_streak" not in run

    run["_echoed_tool"] = "view_sample"                    # stumbles again
    again = _driving_message(run, _session())
    assert "is not a tool call" in again, "should be treated as a first offence"


def test_first_echo_nudge_is_unchanged():
    run = _run()
    run["_echoed_tool"] = "view_sample"
    msg = _driving_message(run, _session())
    assert "is not a tool call" in msg
    assert "view_sample" in msg
