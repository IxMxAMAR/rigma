"""The thinking budget, and reasoning carried between turns.

Both were switched ON as defaults on 2026-08-28 and both were reverted the same
day, for different reasons. The plumbing is kept and tested because it is
correct and useful when asked for; only the defaults changed.

  * reasoning_budget defaulted to 16384. Decode collapsed from ~26 t/s to
    0.59 t/s on a fresh 4K-context chat, and it was the only launch flag that
    had changed. Back to -1 until it is measured in isolation.
  * carry_think was ungated for chats. The four-turn window SLIDES, so every
    turn rewrote the prompt four turns back — and on a DeltaNet hybrid there is
    no KV shifting to recover from a mid-prompt edit, so every turn reprefilled.
    Back to runs, plus an explicit `carry_reasoning` opt-in.
"""
from rigma.models import BUDGET_EXHAUSTED, ComboFlags, GgufFile, RunPlan
from rigma.sessions import build_messages


def _args(**over):
    flags = ComboFlags(ctx=8192, **over)
    plan = RunPlan(model_slug="m", gguf=GgufFile(repo="r", file="m.gguf",
                                                 bytes=1, quant="Q4"),
                   backend="vulkan", flags=flags, origin="calculator")
    return plan.server_args("m.gguf", 11499)


def test_thinking_is_unbudgeted_by_default():
    """A 16384 default coincided with a 50x decode collapse, so it is off.

    Passing no flag leaves the engine on its own default, which is what every
    benchmark that measured 26-33 t/s on this machine actually ran.
    """
    args = _args()

    assert "--reasoning-budget" not in args
    assert "--reasoning-budget-message" not in args


def test_a_budget_that_is_asked_for_is_passed():
    args = _args(reasoning_budget=2048)

    assert args[args.index("--reasoning-budget") + 1] == "2048"


def test_a_budget_ends_in_a_conclusion_not_a_guillotine():
    # Truncating mid-sentence throws away whatever the model had worked out.
    # The message is what it reads as it is cut off.
    args = _args(reasoning_budget=2048)

    msg = args[args.index("--reasoning-budget-message") + 1]
    assert msg == BUDGET_EXHAUSTED
    assert "decisions you reached" in msg
    assert "next step" in msg


def test_a_zero_budget_is_still_a_budget():
    # 0 means "end thinking immediately" to the engine, which is a real
    # setting and must not be confused with "unset".
    args = _args(reasoning_budget=0)

    assert args[args.index("--reasoning-budget") + 1] == "0"


# --- reasoning carried between turns ---------------------------------------

def _session(msgs, **over):
    return {"messages": msgs, "use_tools": False, **over}


_CHAT = [
    {"role": "user", "content": "Plan the app."},
    {"role": "assistant", "content": "Here is step one.",
     "thinking": "I will use Kotlin and start with the data layer."},
]


def test_an_ordinary_chat_does_not_carry_reasoning():
    """Carrying rewrites the prompt four turns back on every turn.

    The window slides, so the edit position moves each turn, and this model
    cannot KV-shift — the engine disables --cache-reuse outright. Every turn
    reprefilled, which cost far more than the re-derivation it saved.
    """
    assert not any("reasoning_content" in m
                   for m in build_messages(_session(_CHAT)))


def test_a_chat_can_opt_in():
    out = build_messages(_session(_CHAT, carry_reasoning=True))

    assert any(m.get("reasoning_content") == "I will use Kotlin and start "
               "with the data layer." for m in out)


def test_runs_still_carry_it():
    # Qwen3.6's agent guidance asks for this, and a run's trajectory is
    # append-only in the way that matters.
    out = build_messages(_session(_CHAT, one_action=True))

    assert any("reasoning_content" in m for m in out)


def test_turning_thinking_off_stops_it_being_carried():
    s = _session(_CHAT, carry_reasoning=True, effort="off")

    assert not any("reasoning_content" in m for m in build_messages(s))


def test_only_the_last_four_assistant_turns_keep_their_reasoning():
    msgs = []
    for i in range(8):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}",
                     "thinking": f"t{i}"})

    out = build_messages(_session(msgs, carry_reasoning=True))
    carried = [m["reasoning_content"] for m in out if "reasoning_content" in m]

    assert carried == ["t4", "t5", "t6", "t7"]


def test_a_single_reasoning_block_cannot_flood_the_window():
    s = _session([{"role": "user", "content": "go"},
                  {"role": "assistant", "content": "done",
                   "thinking": "x" * 50_000}], carry_reasoning=True)

    carried = next(m["reasoning_content"] for m in build_messages(s)
                   if "reasoning_content" in m)

    assert len(carried) == 4000
