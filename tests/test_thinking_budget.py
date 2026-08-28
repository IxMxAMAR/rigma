"""A thinking budget, and reasoning that survives the turn that produced it.

Two problems, one cause. Left unbudgeted, this model has been measured
producing 15.7K characters of deliberation and no answer at all. And reasoning
was only ever carried forward on autonomous runs, so an ordinary chat computed
a plan, streamed it, dropped it, and rebuilt it from nothing next turn.
"""
from rigma.models import BUDGET_EXHAUSTED, ComboFlags, GgufFile, RunPlan
from rigma.sessions import build_messages


def _args(**over):
    flags = ComboFlags(ctx=8192, **over)
    plan = RunPlan(model_slug="m", gguf=GgufFile(repo="r", file="m.gguf",
                                                 bytes=1, quant="Q4"),
                   backend="vulkan", flags=flags, origin="calculator")
    return plan.server_args("m.gguf", 11499)


def test_thinking_is_budgeted_by_default():
    args = _args()

    assert "--reasoning-budget" in args
    assert args[args.index("--reasoning-budget") + 1] == "16384"


def test_the_budget_ends_in_a_conclusion_not_a_guillotine():
    # Truncating mid-sentence throws away whatever the model had worked out.
    # The message is what it reads as it is cut off.
    args = _args()

    assert "--reasoning-budget-message" in args
    msg = args[args.index("--reasoning-budget-message") + 1]
    assert msg == BUDGET_EXHAUSTED
    assert "decisions you reached" in msg
    assert "next step" in msg


def test_an_explicit_unlimited_budget_passes_neither_flag():
    args = _args(reasoning_budget=-1)

    assert "--reasoning-budget" not in args
    assert "--reasoning-budget-message" not in args


def test_a_zero_budget_is_still_a_budget():
    # 0 means "end thinking immediately" to the engine, which is a real
    # setting and must not be confused with "unset".
    args = _args(reasoning_budget=0)

    assert args[args.index("--reasoning-budget") + 1] == "0"


# --- reasoning carried between turns ---------------------------------------

def _session(msgs, **over):
    return {"messages": msgs, "use_tools": False, **over}


def test_an_ordinary_chat_now_carries_its_reasoning():
    s = _session([
        {"role": "user", "content": "Plan the app."},
        {"role": "assistant", "content": "Here is step one.",
         "thinking": "I will use Kotlin and start with the data layer."},
    ])

    out = build_messages(s)

    assert any(m.get("reasoning_content") == "I will use Kotlin and start "
               "with the data layer." for m in out)


def test_turning_thinking_off_stops_it_being_carried():
    s = _session([
        {"role": "user", "content": "Plan the app."},
        {"role": "assistant", "content": "Step one.", "thinking": "secret"},
    ], effort="off")

    assert not any("reasoning_content" in m for m in build_messages(s))


def test_only_the_last_four_assistant_turns_keep_their_reasoning():
    msgs = []
    for i in range(8):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}",
                     "thinking": f"t{i}"})

    out = build_messages(_session(msgs))
    carried = [m["reasoning_content"] for m in out
               if "reasoning_content" in m]

    assert carried == ["t4", "t5", "t6", "t7"]


def test_a_single_reasoning_block_cannot_flood_the_window():
    s = _session([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done", "thinking": "x" * 50_000},
    ])

    out = build_messages(s)
    carried = next(m["reasoning_content"] for m in out
                   if "reasoning_content" in m)

    assert len(carried) == 4000


def test_a_turn_with_no_thinking_carries_nothing():
    s = _session([{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}])

    assert not any("reasoning_content" in m for m in build_messages(s))
