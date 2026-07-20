"""Observation masking — deterministic context reclaim for autonomous runs.

LLM compaction summarises the whole span into prose, which preserves the gist
and destroys the exact strings the agent navigates by: filenames, error text,
the tool argument that actually worked. Masking instead shrinks only the
ENVIRONMENT's replies and leaves the model's own reasoning byte-identical.
"""
from rigma.context import mask_observations, session_chars


def _tool_msg(name, body, n=1):
    return {"role": "user", "kind": "tool_result",
            "content": f"TOOL RESULT {name}: " + body,
            "tools": [{"name": name, "ok": True}]}


def _msgs(n_pairs=6, body_len=500):
    out = []
    for i in range(n_pairs):
        out.append({"role": "assistant", "content": f"thinking about step {i}",
                    "tool_trace": [{"name": "sample_files"}]})
        out.append(_tool_msg("sample_files", f"file_{i}.png, " * (body_len // 14)))
    return out


def test_assistant_turns_are_never_touched():
    # the model's trajectory is what it orients by — masking must not paraphrase
    # a single token of it
    msgs = _msgs()
    before = [m["content"] for m in msgs if m["role"] == "assistant"]
    out, n = mask_observations(msgs, keep_recent=2, budget_chars=500)
    after = [m["content"] for m in out if m["role"] == "assistant"]
    assert before == after
    assert n > 0


def test_masking_shrinks_the_session():
    msgs = _msgs()
    big = session_chars(msgs)
    out, _ = mask_observations(msgs, keep_recent=2, budget_chars=1500)
    assert session_chars(out) < big


def test_recent_observations_survive():
    # the model needs the last few results in full — those are the ones it is
    # actually acting on
    msgs = _msgs()
    out, _ = mask_observations(msgs, keep_recent=4, budget_chars=100)
    tail = [m for m in out[-4:] if m.get("kind") == "tool_result"]
    assert tail, "fixture should leave tool results in the tail"
    assert all("masked" not in m["content"] for m in tail)


def test_oldest_are_masked_first():
    msgs = _msgs()
    out, _ = mask_observations(msgs, keep_recent=2, budget_chars=2000)
    obs = [m for m in out if m.get("kind") == "tool_result"]
    masked = ["masked" in m["content"] for m in obs]
    # once masking stops it must not resume — all the True values come first
    assert masked == sorted(masked, reverse=True), masked


def test_a_masked_result_still_names_its_tool():
    # "something happened here" is useless; the model must still see WHICH tool
    # ran, or it cannot tell a finished step from an unstarted one
    msgs = _msgs()
    out, _ = mask_observations(msgs, keep_recent=0, budget_chars=10)
    obs = [m for m in out if m.get("kind") == "tool_result"]
    assert all("sample_files" in m["content"] for m in obs)


def test_masking_is_idempotent():
    # it runs every turn; a second pass must not re-mask a placeholder into a
    # placeholder-of-a-placeholder
    msgs = _msgs()
    once, _ = mask_observations(msgs, keep_recent=2, budget_chars=1000)
    twice, n2 = mask_observations(once, keep_recent=2, budget_chars=1000)
    assert [m["content"] for m in once] == [m["content"] for m in twice]
    assert n2 == 0


def test_already_small_sessions_are_untouched():
    msgs = _msgs(n_pairs=2, body_len=20)
    out, n = mask_observations(msgs, keep_recent=2, budget_chars=1_000_000)
    assert n == 0
    assert [m["content"] for m in out] == [m["content"] for m in msgs]


def test_legacy_sessions_without_the_kind_tag_still_mask():
    # sessions written before this change have no kind/tools keys — the owner
    # has live runs on disk, so masking must recognise them by prefix
    msgs = _msgs()
    for m in msgs:
        m.pop("kind", None)
        m.pop("tools", None)
    out, n = mask_observations(msgs, keep_recent=2, budget_chars=500)
    assert n > 0, "legacy tool results must still be recognised"
    assert all(m["content"].startswith("thinking") or "TOOL RESULT" in m["content"]
               for m in out)


def test_non_tool_user_messages_are_never_masked():
    # the RUN STATE block and real user steering are not observations
    msgs = _msgs()
    msgs.insert(2, {"role": "user", "content": "### RUN STATE\nstep: 1 of 3"})
    out, _ = mask_observations(msgs, keep_recent=0, budget_chars=10)
    assert any(m["content"].startswith("### RUN STATE") for m in out)


# ---- review fixes (2026-07-20 adversarial review) ----

def test_user_steering_starting_with_the_prefix_is_not_masked():
    # a human typing "TOOL RESULT formats are failing, fix your syntax" is
    # steering, not an observation — masking it would erase the very
    # correction the user sent. Only the server's "TOOL RESULT <name>: " shape
    # counts as legacy.
    msgs = _msgs()
    msgs.insert(2, {"role": "user",
                    "content": "TOOL RESULT formats are failing, fix them"})
    for m in msgs:
        m.pop("kind", None)
        m.pop("tools", None)
    out, _ = mask_observations(msgs, keep_recent=0, budget_chars=10)
    kept = [m for m in out if "formats are failing" in m["content"]]
    assert kept, "steering text must survive masking untouched"


def test_a_tagged_non_tool_kind_is_never_sniffed():
    # a message that explicitly declares another kind must not fall through to
    # the legacy prefix sniff, whatever its text looks like
    msgs = _msgs()
    msgs.insert(2, {"role": "user", "kind": "steering",
                    "content": "TOOL RESULT sample_files: do it differently"})
    out, _ = mask_observations(msgs, keep_recent=0, budget_chars=10)
    assert any("do it differently" in m["content"] for m in out)


def test_legacy_masking_is_idempotent_too():
    # the masked form still starts with "TOOL RESULT <name>: ", so it still
    # LOOKS like an observation — a second pass must recognise the placeholder
    # and leave it alone rather than masking the mask
    msgs = _msgs()
    for m in msgs:
        m.pop("kind", None)
        m.pop("tools", None)
    once, n1 = mask_observations(msgs, keep_recent=0, budget_chars=10)
    twice, n2 = mask_observations(once, keep_recent=0, budget_chars=10)
    assert n1 > 0 and n2 == 0
    assert [m["content"] for m in once] == [m["content"] for m in twice]


def test_masking_a_long_session_is_fast():
    # session_chars() used to be recomputed inside the loop — O(n^2) ON the
    # asyncio event loop. This pins the running-total fix with a budget a
    # quadratic implementation cannot meet.
    import time as _t
    msgs = _msgs(n_pairs=3000, body_len=300)
    t0 = _t.perf_counter()
    out, n = mask_observations(msgs, keep_recent=6, budget_chars=50_000)
    took = _t.perf_counter() - t0
    assert n > 0
    assert took < 1.0, f"masking 6000 messages took {took:.2f}s"
