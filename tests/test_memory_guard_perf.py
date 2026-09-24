r"""R3: the anchoring guard must not be a denial-of-service on long text.

`looks_like_raw_trace` runs on every stored rule and on every edit, and the
`_FILENAME` pattern used to be `\S+\.(?:ext)`. `\S+` can match the dots the
literal needs, so a long token containing no dot makes the engine retry every
split point: quadratic. Measured on this box before the fix (AUDIT R3-10-11):
60 000 chars -> 19.1 s, 200 000 chars -> 76 s, synchronously, on whichever
thread called add()/update(). The store is fed by a local model, so the text
length is not something the caller controls.

The fix is `[^\s.]+`, which cannot swallow the literal dot and therefore has
only one way to match. These tests pin BOTH halves: the timing, and that the
guard still recognises exactly the filenames it recognised before.

Raw string, because the patterns this file quotes are regexes: `\S` and `\s` in a
plain docstring are invalid escape sequences and Python says so on every import.
"""
import time

import pytest

from rigma import memory


def test_a_long_dotless_rule_is_guarded_in_linear_time():
    # 200 000 chars took ~76 s pre-fix. The bound is deliberately loose (the
    # point is the asymptotic class, not a benchmark on a shared box) and the
    # same call on the fixed pattern is ~30 ms.
    text = "A" * 200_000
    t0 = time.perf_counter()
    assert memory.looks_like_raw_trace(text) is False
    took = time.perf_counter() - t0
    assert took < 2.0, f"the anchoring guard took {took:.1f}s on 200k chars"


def test_a_long_rule_ending_in_a_space_is_still_fast():
    # the space gives `\S+` a place to stop, so this shape was merely slow
    # (5.3 s at 60 000) rather than catastrophic — still the same defect
    text = "A" * 200_000 + " "
    t0 = time.perf_counter()
    memory.looks_like_raw_trace(text)
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("text", [
    r"do not open C:\out\Comfy_UI_428.png again",
    r"never read \\server\share\a.txt",
    "check server.log for the trace",
    "the fix is in run_python('print(1)')",
    "see notes/plan.md before editing",
    "train.safetensors is the wrong quant",
    "the render landed in ui_v2/index.html",
])
def test_the_guard_still_rejects_the_shapes_it_did_before(text):
    assert memory.looks_like_raw_trace(text) is True


@pytest.mark.parametrize("text", [
    "Never type filenames; pass files by reference.",
    "Prefer q8_0 for the KV cache.",
    "Verify every plan item before calling task_complete.",
    "Use Python 3.12 with the venv activated.",
])
def test_ordinary_rules_are_still_allowed(text):
    assert memory.looks_like_raw_trace(text) is False


def test_a_dotless_run_before_a_real_filename_is_still_caught():
    # `[^\s.]+` must not have made the pattern miss a filename that follows a
    # long dotless token
    text = "A" * 5000 + " " + "shot.png"
    assert memory.looks_like_raw_trace(text) is True


def test_add_of_a_long_rule_does_not_hang_the_store(tmp_path):
    store = memory.MemoryStore(tmp_path / "m.jsonl")
    t0 = time.perf_counter()
    store.add(kind="technique", text="A" * 200_000)
    assert time.perf_counter() - t0 < 5.0
    assert len(store.all()) == 1
