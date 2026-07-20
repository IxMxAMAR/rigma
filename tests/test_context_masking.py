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
