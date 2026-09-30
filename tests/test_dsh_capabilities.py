"""The capability bridge: what Rigma adds to DSH, and what it does with it.

TWO INDEPENDENT BLOCKERS, and this file covers both because either one alone
leaves the user with nothing to see:

1. `sdk-minimal` is a STANDALONE Cordis tree that mounts no goal, subagent,
   todo, skill, plan-mode, filesystem or compaction plugin at all. Those
   capabilities did not merely go unreported — they did not exist in Rigma's
   DSH runtime. `agent-capabilities.patch.yml` inserts them.

2. Even once mounted, their events ride inside `session.event` and were dropped
   by `_dsh_runner._notice_text`, which matched six keywords against a
   two-field summary and truncated to 200 characters. The runner now projects
   them as structured events.

A malformed patch row is SKIPPED WITH A WARNING rather than failing the boot, so
a typo silently yields a missing tool. That is why the first test below reads the
patch file and asserts the rows, rather than trusting that a successful boot
means a complete one.
"""
from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace

import pytest
import yaml

from rigma import _dsh_runner as runner
from rigma import harness, harness_dsh
from rigma._dsh_runner import _project


# The rows the bridge is expected to carry, grouped by the capability they buy.
# If a row is dropped from the patch this list is what notices.
EXPECTED_ROWS = {
    "filesystem": ["fs-local", "tool-fs", "tool-fs-search", "tool-str-replace-editor"],
    "instructions": ["agent-instructions"],
    "todos": ["tool-todo"],
    "goals": ["goal", "goal-round-driver", "tool-goal"],
    "skills": ["skill", "skill-filesystem", "tool-skill"],
    "subagents": [
        "subagent", "subagent-spawn-in-process", "subagent-fork-in-process",
        "tool-subagent-control", "tool-subagent-list-agents", "tool-subagent",
        "tool-subagent-fork",
    ],
    # "plan" is deliberately ABSENT — see `test_plan_mode_is_not_mounted_and_why`.
    "workflow": ["ptc-runtime", "workflow-ptc", "tool-workflow"],
    "context": ["token-meter", "compaction-basic"],
    "governance": ["approval", "permission"],
}


def _patch_rows() -> list[dict]:
    path = harness_dsh.capability_patch()
    assert path, "the capability patch is not shipped in the package"
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    rows: list[dict] = []
    for entry in doc:
        rows.extend(entry.get("insert") or [])
    return rows


def test_the_capability_patch_is_shipped_with_the_package():
    """It is read from the package, not the source tree, so the wheel has it."""
    path = harness_dsh.capability_patch()
    assert path.endswith("agent-capabilities.patch.yml")
    from pathlib import Path

    assert Path(path).is_file()


def test_every_capability_the_minimal_profile_omits_is_inserted():
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    missing = {
        group: [rid for rid in ids if rid not in by_id]
        for group, ids in EXPECTED_ROWS.items()
    }
    missing = {g: v for g, v in missing.items() if v}
    assert not missing, f"capabilities the patch no longer inserts: {missing}"


# --------------------------------------------------------------------------
# O4 (checkpoint 30): the approval/governance audit trail. `sandbox/mode` fires
# from `sandbox-policy`, which sdk-minimal mounts, so the panel looked live —
# but the three `approval/*` events come only from `dsh-user-approval` and
# `permission/preset` only from `dsh-permission-presets`, and neither plugin was
# mounted. The trail could not fill and nothing said why. These pin the mount
# rows through the same loader the runner uses, and pin the one config value
# that decides whether mounting the approval service is safe.
# --------------------------------------------------------------------------


def test_the_approval_and_permission_services_are_mounted():
    """The missing hop: without these rows the audit trail is unreachable.

    Read through `_patch_rows()`, i.e. `harness_dsh.capability_patch()` plus
    `yaml.safe_load` — the same loader the runner hands to DSH — so a row that
    is present but unparseable fails here rather than silently at boot.
    """
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["approval"]["name"] == "@deepseek-ai/dsh-user-approval"
    assert by_id["permission"]["name"] == "@deepseek-ai/dsh-permission-presets"


def test_the_approval_policy_matches_the_sandbox_rigma_actually_runs():
    """`never`, not the base bundle's `ask`.

    sdk-minimal hardcodes `sandbox-policy: danger-full-access`, and Rigma never
    sets DSH_PERMISSION_MODE — so the base's `!!js` expression would resolve to
    `ask` here. `ask` on a transport with no answerer fails CLOSED (the SDK wire
    has no approval-response method), which would deny calls the unconfined
    sandbox permits. `never` rejects deterministically without a prompt, which
    is the stance that matches the sandbox Rigma runs.
    """
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["approval"]["config"]["policy"] == "never"
    # A plain string is also the only thing that can be here: `_patch_rows()`
    # parses with `yaml.safe_load`, which raises on the base bundle's `!!js`
    # tag, so copying that expression verbatim is not an option.
    assert isinstance(by_id["approval"]["config"]["policy"], str)


def test_the_danger_full_access_preset_agrees_with_the_approval_policy():
    """A preset whose `approval` disagrees with the service's `policy` would make
    `derive()` find no preset for the composed defaults and refuse to mount."""
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    presets = by_id["permission"]["config"]["presets"]
    assert presets["danger-full-access"] == {
        "sandbox": "danger-full-access", "approval": "never"}
    assert presets["read-only"]["approval"] == "ask"
    assert presets["workspace-write"]["approval"] == "ask"


def test_the_two_subagent_tools_are_the_same_plugin_under_two_names():
    """dsh-base does it this way, and the toolName is what the model calls."""
    rows = [r for r in _patch_rows() if r.get("name") == "@deepseek-ai/dsh-tool-subagent"]
    assert len(rows) == 2, rows
    names = {r["config"]["toolName"] for r in rows}
    assert names == {"subagent", "subagent_fork"}


def test_each_subagent_tool_is_paired_with_the_right_provider():
    """F9: the PAIRING is load-bearing, and this used to assert only that each
    `provider` was truthy.

    That is a real gap rather than a stylistic one. `provider` is required by the
    plugin's schema, so the old assertion caught its ABSENCE — but swapping the two
    values passed it while inverting the tools' meaning: `spawn` starts a fresh child
    and `fork` continues from the parent's context, so a swap would make
    `subagent_fork` silently start a child with no memory of the conversation. The
    patch's own comments make the assignment load-bearing; this pins it.
    """
    rows = [r for r in _patch_rows() if r.get("name") == "@deepseek-ai/dsh-tool-subagent"]
    pairs = {(r["config"]["toolName"], r["config"]["provider"]) for r in rows}
    assert pairs == {("subagent", "spawn"), ("subagent_fork", "fork")}, (
        f"the tool/provider pairing was changed or swapped: {pairs}")


# ---------------------------------------------------------------------------
# R6-WEBFETCH: `web_fetch` is mounted, `web_search` is not.
#
# The split is the point. The old single comment omitted both for an API-key reason
# that only covers SEARCH — `dsh-web-fetch-http` has no credentials to check
# (`web-fetch-http/src/provider.ts:51`) — so `web_fetch` was absent for a constraint
# that does not apply to it. These guard the split in both directions, because either
# half can rot silently: mounting search without a provider gives the model a tool that
# fails on every call, and dropping the `search: false` is exactly how that happens.
# ---------------------------------------------------------------------------

def test_web_fetch_is_mounted_keylessly():
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["web"]["name"] == "@deepseek-ai/dsh-web"
    assert by_id["web-fetch-http"]["name"] == "@deepseek-ai/dsh-web-fetch-http"
    assert by_id["tool-web"]["name"] == "@deepseek-ai/dsh-tool-web"


def test_web_search_is_not_mounted():
    """No search provider, and no key is configured anywhere in this build."""
    names = {str(r.get("name")) for r in _patch_rows()}
    assert "@deepseek-ai/dsh-web-search-deepseek" not in names, (
        "a search provider needs DEEPSEEK_API_KEY and would fail on every call")


def test_tool_web_registers_fetch_only():
    """`search: false` is what makes the mount fetch-only.

    Without it `dsh-tool-web` registers `web_search` too (`src/index.ts`: `search`
    defaults true), and with no search provider registered the tool stays visible and
    returns a structured error on every call — the outcome the original comment was
    right to want to avoid. So the config is load-bearing, not decoration.
    """
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["tool-web"]["config"]["search"] is False
    # `fetch` is deliberately NOT set: it defaults true, and restating a default is
    # how a default quietly becomes a decision someone has to maintain.
    assert "fetch" not in by_id["tool-web"]["config"]


def test_the_capability_menu_declares_both_halves_of_the_web_decision():
    """F8's other half: the menu said nothing about web either way, so a user could not
    tell an absence from an oversight. Both halves must be named."""
    dsh = next(h for h in harness.list_harnesses() if h["name"] == harness.DSH)
    caps = " ".join(dsh["capabilities"])
    unsup = " ".join(dsh["unsupported"])
    assert "web_fetch" in caps, "the mounted capability is undeclared"
    assert "web search" in unsup, (
        "`web_search` is absent, and an undeclared absence reads as an oversight")


def test_rows_whose_config_has_no_default_carry_it():
    """These four are `z.<type>().required()` upstream — no default exists."""
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["tool-todo"]["config"]["allowParallelInProgress"] is True
    assert by_id["tool-fs-search"]["config"]["sampleOverCapGlobResults"] is False
    for row in _patch_rows():
        if row.get("name") == "@deepseek-ai/dsh-tool-subagent":
            assert row["config"]["provider"]


def test_every_inserted_row_names_a_package():
    """A row with no `name` would be a silent no-op."""
    for row in _patch_rows():
        assert row.get("id"), row
        assert str(row.get("name") or "").startswith("@deepseek-ai/"), row


# --------------------------------------------------------------------------
# The projection: what the runner does with a notification.


def _notif(method: str, payload: dict):
    return SimpleNamespace(method=method, payload=payload)


def _session_event(kind: str, data: dict):
    return _notif("session.event", {"sessionId": "s1", "event": {"type": kind, "data": data}})


def test_a_tool_call_carries_the_tool_name_and_arguments():
    """Before this, the parent was told the literal string "session.event
    tool/call" — every DSH turn rendered zero tool chips and the tool name was
    unrecoverable downstream."""
    out = runner._project(_session_event("tool/call", {
        "name": "read_file", "arguments": {"path": "a.py"}, "callId": "c1",
    }))
    assert out == [{"type": "tool", "name": "read_file",
                    "args": {"path": "a.py"}, "id": "c1"}]


def test_two_calls_to_one_tool_keep_distinct_ids():
    a = runner._project(_session_event("tool/call", {"name": "read", "callId": "c1"}))[0]
    b = runner._project(_session_event("tool/call", {"name": "read", "callId": "c2"}))[0]
    assert a["id"] != b["id"], "the id is what stops the chips collapsing"


def test_a_tool_result_reports_failure_from_the_backend():
    ok = runner._project(_session_event("tool/result", {"callId": "c1", "content": "fine"}))[0]
    bad = runner._project(_session_event("tool/result",
                                         {"callId": "c1", "isError": True}))[0]
    assert ok["ok"] is True and bad["ok"] is False


def test_a_goal_change_is_structured_not_a_progress_line():
    out = runner._project(_session_event("goal/change", {
        "operation": "create",
        "goal": {"id": "g1", "phase": "active", "objective": "ship it"},
        "roundsStarted": 2,
    }))
    assert out[0]["type"] == "state"
    assert out[0]["event"] == "goal/change"
    # The WHOLE envelope is kept: the snapshot AND the round count and the
    # operation, because a UI needs all three to say "round 4 of 60, paused".
    assert out[0]["data"]["goal"]["phase"] == "active"
    assert out[0]["data"]["roundsStarted"] == 2
    assert out[0]["data"]["operation"] == "create"


def test_a_todo_write_carries_the_whole_list():
    out = runner._project(_session_event("todo/write", {
        "todos": [{"content": "a", "status": "pending"}],
    }))
    assert out[0]["data"]["todos"] == [{"content": "a", "status": "pending"}]


def test_plan_mode_and_usage_are_both_carried():
    plan = runner._project(_session_event("plan/mode", {"active": True}))[0]
    assert plan["event"] == "plan/mode" and plan["data"]["active"] is True
    usage = runner._project(_session_event("assistant/message", {
        "usage": {"inputTokens": 10},
    }))[0]
    assert usage["event"] == "usage" and usage["data"]["inputTokens"] == 10


def test_subagent_lifecycle_is_top_level_and_no_longer_filtered_out():
    """`subagent.finished` contains "finish", not "fail", so the old keyword
    filter dropped it — a subagent could start AND end unseen."""
    started = runner._project(_notif("subagent.started", {
        "parentSessionId": "p", "childSessionId": "c",
    }))[0]
    assert started["type"] == "state" and started["event"] == "subagent.started"
    assert started["data"]["childSessionId"] == "c"
    finished = runner._project(_notif("subagent.finished", {
        "childSessionId": "c", "status": "ok",
    }))[0]
    assert finished["event"] == "subagent.finished"


def test_former_internal_chatter_is_now_an_explicit_notice():
    """B2: these four used to be dropped in SILENCE, and this test used to assert
    `== []`. The contract changed, deliberately.

    The reason for the old filter is still respected: a one-line turn produced
    fifteen notifications, so none of these becomes STRUCTURED state and none of
    them is given a payload. But silence is how `command/run`/`command/done` never
    reached the UI and how a newer DSH could add a type nobody ever saw. Each one
    is now the generic notice, naming its type and nothing else.

    `session/title` left this list earlier for its own reason (it is the server's
    authoritative title, worth a real line); that decision is unchanged.
    """
    for kind in ("agent/inbox/spliced", "step/start", "request/header",
                 "delivery-accepted"):
        out = runner._project(_session_event(kind, {}))
        assert len(out) == 1, (kind, out)
        assert out[0]["type"] == "notice", (kind, out)
        assert kind in out[0]["text"], (kind, out)


def test_a_malformed_notification_produces_nothing_and_does_not_raise():
    for bad in (SimpleNamespace(method="x", payload=None),
                SimpleNamespace(method="x", payload={"event": "not-a-dict"}),
                SimpleNamespace(method="x", payload={"event": {"type": 5}})):
        assert runner._project(bad) == []


# --------------------------------------------------------------------------
# B2 nit: every notice text is ONE line and <=200 characters, including the
# two paths that do not draw their content from `_UNKNOWN_NOTICE_FIELDS`. The
# projector-failure fallback used to cap only its detail and then prepend a
# ~40-character prefix (reachable ~240), and an exception message can carry
# arbitrary prose — the one place §0.3's no-owner-prose rule could leak. The
# raw event `type` is the other non-allow-listed input, named on purpose.
# --------------------------------------------------------------------------


class _ProjectorBoom(Exception):
    """A projector failure whose message is long, multi-line and prose-bearing."""


def test_a_projector_failure_notice_is_one_capped_line(monkeypatch):
    """The fallback must not exceed the bound the ordinary notices use.

    Before this, the emitted text was
    `"session.event could not be projected: " + f"{type}: {exc}"[:200]` — the
    prefix escaped the cap, so the finished notice could reach ~240 characters,
    and a multi-line exception message put its second line straight into the
    progress line.
    """
    secret = "owner prose that must not own the line\n" + "SECRET " * 80

    def explode(notification, streamed=None):
        raise _ProjectorBoom(secret)

    monkeypatch.setattr(runner, "_project", explode)
    emitted: list[dict] = []
    monkeypatch.setattr(runner, "_emit", emitted.append)

    runner._forward(_session_event("tool/call", {"name": "read"}))

    assert len(emitted) == 1, emitted
    text = emitted[0]["text"]
    assert emitted[0]["type"] == "notice"
    assert "\n" not in text and "\r" not in text, repr(text)
    assert len(text) <= 200, (len(text), text)
    assert text.startswith("session.event could not be projected:"), text
    # The exception's own type name is still named — the notice stays useful.
    assert "_ProjectorBoom" in text, text


def test_the_projector_failure_notice_is_built_by_the_shared_helper(monkeypatch):
    """Pinned to the helper, not to a hand-rolled cap: a future edit that
    re-inlines the f-string would re-introduce the un-capped prefix."""
    assert runner._notice_line("a" * 300) == "a" * runner._NOTICE_MAX
    assert runner._notice_line("one\ntwo") == "one two"


def test_an_unknown_event_type_cannot_add_a_line_or_outgrow_the_cap():
    """The type is the only non-allow-listed token in this notice, and it is
    named by design; the shared helper is what bounds it."""
    out = runner._project(_session_event("PROSE " + "y" * 400, {}))
    assert len(out) == 1 and out[0]["type"] == "notice"
    assert "\n" not in out[0]["text"]
    assert len(out[0]["text"]) <= 200
    assert out[0]["text"].startswith("session.event PROSE ")

    multiline = runner._project(_session_event("line one\nline two\nline three", {}))
    assert "\n" not in multiline[0]["text"]
    assert multiline[0]["text"] == "session.event line one line two line three"


# --------------------------------------------------------------------------
# The adapter's translation of those events onto the seam.


def test_the_adapter_passes_structured_state_through():
    ev = harness_dsh._event_for({
        "type": "state", "event": "goal/change", "data": {"goal": {"phase": "active"}},
    })
    assert ev is not None
    assert ev.kind == "state"
    assert ev.event == "goal/change"
    assert ev.data == {"goal": {"phase": "active"}}


def test_the_adapter_carries_the_call_id_on_tool_events():
    ev = harness_dsh._event_for({"type": "tool", "name": "read", "args": {}, "id": "c1"})
    assert ev is not None and ev.data == {"id": "c1"}
    res = harness_dsh._event_for({"type": "tool_result", "ok": False, "id": "c1"})
    assert res is not None and res.data == {"id": "c1"} and res.ok is False


def test_the_adapter_still_skips_an_unknown_kind():
    assert harness_dsh._event_for({"type": "invented_by_a_newer_dsh"}) is None


def test_a_turn_event_can_be_built_without_the_new_fields():
    """They were added with defaults, so every existing adapter and call site
    keeps working."""
    from rigma.harness import TurnEvent

    ev = TurnEvent(kind="text", text="hi")
    assert ev.event == "" and ev.data is None


@pytest.mark.parametrize("window,tokens", [(32768, 4096), (65536, 8192)])
def test_the_capability_patch_is_handed_to_the_runner(tmp_path, window, tokens):
    """The parent locates it; the child only receives a path, so there is
    exactly one copy of the file to keep correct."""
    job_path = harness_dsh.capability_patch()
    assert job_path, "the parent must be able to find the patch"
    from pathlib import Path

    assert Path(job_path).is_file()

# --------------------------------------------------------------------------
# DSH's governance events. These are the ones a reader most needs and could not
# see at all: DSH records when it ASKED for permission, what it was told, and
# which confinement was in force. All of them are `log-only` in DSH's own words —
# durable and replayable, never part of the model transcript — which is exactly
# why they belong beside the turn rather than inside it.
#
# The payload shapes are quoted from DSH's own declarations:
#   approval/asked    {id, toolName, callId?, reason?}
#                     packages/interaction/user-approval/src/types.ts:44
#   approval/decided  {id, outcome}   outcome in allowed-once|rejected|cancelled|unavailable
#                     packages/interaction/user-approval/src/types.ts:55
#   approval/policy   {policy, source?}
#                     packages/interaction/user-approval/src/index.ts:33
#   sandbox/mode      {mode, source?}  mode in read-only|workspace-write|danger-full-access
#                     packages/sandbox/sandbox-policy/src/session-mode.ts:33
#   permission/preset {preset}
#                     packages/interaction/permission-presets/src/index.ts:57
#   session/title     {title, messageSeqs, source}
#                     packages/session/session-format-v0-to-v1/src/dispositions.ts:86


def test_an_approval_ask_carries_the_tool_and_the_reason():
    evs = _project(_session_event("approval/asked", {
        "id": "ap_1", "toolName": "write", "callId": "c1",
        "reason": "outside the workspace",
    }))
    assert len(evs) == 1
    assert evs[0]["type"] == "state"
    assert evs[0]["event"] == "approval/asked"
    # The WHOLE payload, not a summary: `id` is what pairs the ask with its
    # decision, and dropping it would make the trail unpairable.
    assert evs[0]["data"]["id"] == "ap_1"
    assert evs[0]["data"]["toolName"] == "write"
    assert evs[0]["data"]["reason"] == "outside the workspace"


def test_an_approval_decision_carries_its_outcome_and_id():
    evs = _project(_session_event("approval/decided",
                                  {"id": "ap_1", "outcome": "allowed-once"}))
    assert evs[0]["event"] == "approval/decided"
    assert evs[0]["data"] == {"id": "ap_1", "outcome": "allowed-once"}


def test_the_fail_closed_outcome_is_preserved_verbatim():
    """`unavailable` means nobody could answer, so the tool was NOT permitted.
    It must survive the bridge unmangled — it is the one outcome a reader could
    mistake for a network problem."""
    evs = _project(_session_event("approval/decided",
                                  {"id": "ap_2", "outcome": "unavailable"}))
    assert evs[0]["data"]["outcome"] == "unavailable"


def test_the_sandbox_mode_and_permission_preset_are_carried():
    sandbox = _project(_session_event("sandbox/mode",
                                      {"mode": "danger-full-access"}))
    assert sandbox[0]["event"] == "sandbox/mode"
    assert sandbox[0]["data"]["mode"] == "danger-full-access"

    preset = _project(_session_event("permission/preset", {"preset": "ask"}))
    assert preset[0]["event"] == "permission/preset"
    assert preset[0]["data"]["preset"] == "ask"

    policy = _project(_session_event("approval/policy",
                                     {"policy": "ask", "source": "delegation"}))
    assert policy[0]["event"] == "approval/policy"
    assert policy[0]["data"]["policy"] == "ask"


def test_the_servers_own_title_is_carried():
    """Rigma invents its own chat title without ever learning DSH's, so the two
    can disagree. This is the authoritative one."""
    evs = _project(_session_event("session/title",
                                  {"title": "Fix the flaky test", "messageSeqs": [1, 2],
                                   "source": "llm"}))
    assert evs[0]["event"] == "session/title"
    assert evs[0]["data"]["title"] == "Fix the flaky test"


def test_every_governance_event_is_a_recognised_state_event():
    """A guard against the list drifting: each of these must be in
    `_STATE_EVENTS`, or the runner silently drops it and the UI shows nothing
    while every test above still passes on a hand-built payload."""
    from rigma import _dsh_runner as runner
    for name in ("approval/asked", "approval/decided", "approval/policy",
                 "sandbox/mode", "permission/preset", "session/title"):
        assert name in runner._STATE_EVENTS, name


def test_the_runner_no_longer_drops_governance_as_internal_chatter():
    """`_notice_text` matched six keywords and truncated to 200 characters, so
    these fell through it. They must now be structured events."""
    for kind in ("approval/asked", "sandbox/mode", "session/title"):
        evs = _project(_session_event(kind, {"id": "x", "mode": "read-only",
                                             "title": "t", "toolName": "bash"}))
        assert evs, kind
        assert evs[0]["type"] == "state", kind

# --------------------------------------------------------------------------
# DSH's compaction lifecycle. Rigma's NATIVE compaction already reports itself
# (`masked`, `housekeeping`, `compacted`, drawn at Transcript.tsx:310-322), but a
# DSH turn reported none of it — so a long turn busy summarising its own context
# looked exactly like a turn that had hung.
#
# The four events are a BRACKET, not four facts: `compaction/start` opens it and
# `compaction/end` closes it, paired by `compactionId`. The fold in
# frontend-v2/src/chat/compaction.ts does the pairing; these tests cover the
# transport half, which is that the events survive the runner at all.
#
# Shapes quoted from DSH's own declarations
# (packages/compaction/compaction/src/types.ts:24-89):
#   compaction/start    {compactionId, turn, sourceCommandId?}
#   compaction/summary  {compactionId, summary: ContentBlock[], shadowedRange,
#                        shadowedSeqs, shadowedTokenCount, provider, model, ...}
#   compaction/prune    {shadowedRange, shadowedSeqs, shadowedTokenCount}
#   compaction/end      {compactionId, turn, sourceCommandId?, error?}


def test_a_compaction_start_opens_a_bracket():
    evs = _project(_session_event("compaction/start",
                                  {"compactionId": "c1", "turn": 1}))
    assert len(evs) == 1
    assert evs[0]["type"] == "state"
    assert evs[0]["event"] == "compaction/start"
    assert evs[0]["data"]["compactionId"] == "c1"


def test_a_compaction_summary_passes_through_whole():
    """The server deliberately does NOT reshape this payload.

    `summary` is a ContentBlock list and `shadowedSeqs` is a list. Flattening
    either in serve.py would mean the fold in the UI could no longer COUNT what
    it was given — `shadowedSeqs` is how the message count is known at all.
    """
    seqs = [0, 1, 2, 3]
    evs = _project(_session_event("compaction/summary", {
        "compactionId": "c1",
        "summary": [{"type": "text", "text": "they discussed the sea"}],
        "shadowedRange": {"start": 0, "end": 3},
        "shadowedSeqs": seqs,
        "shadowedTokenCount": 12345,
        "provider": "deepseek-official",
        "model": "DeepSeek-V4.1-Flash",
    }))
    assert evs[0]["event"] == "compaction/summary"
    d = evs[0]["data"]
    assert d["summary"] == [{"type": "text", "text": "they discussed the sea"}]
    assert d["shadowedSeqs"] == seqs
    assert d["shadowedTokenCount"] == 12345
    assert d["provider"] == "deepseek-official"
    assert d["model"] == "DeepSeek-V4.1-Flash"


def test_a_compaction_prune_carries_its_token_price():
    """`prune` is a model-free replacement and has NO compactionId of its own —
    it is priced by the metering event immediately before it, so the fold has to
    attribute it to whichever bracket is open. Losing `shadowedTokenCount` here
    would lose the only number it carries."""
    evs = _project(_session_event("compaction/prune", {
        "shadowedRange": {"start": 0, "end": 4},
        "shadowedSeqs": [0, 1, 2, 3],
        "shadowedTokenCount": 900,
    }))
    assert evs[0]["event"] == "compaction/prune"
    assert evs[0]["data"]["shadowedTokenCount"] == 900
    assert "compactionId" not in evs[0]["data"]


def test_a_failed_compaction_carries_its_error():
    """The one compaction fact a reader must not miss: the context did NOT
    shrink, so the turn may be about to fail for want of room."""
    evs = _project(_session_event("compaction/end", {
        "compactionId": "c1", "turn": 1,
        "error": "summariser returned nothing",
    }))
    assert evs[0]["event"] == "compaction/end"
    assert evs[0]["data"]["error"] == "summariser returned nothing"


def test_a_successful_compaction_end_has_no_error():
    evs = _project(_session_event("compaction/end",
                                  {"compactionId": "c1", "turn": 1}))
    assert evs[0]["event"] == "compaction/end"
    assert evs[0]["data"].get("error") in (None, "")


def test_every_compaction_event_is_a_recognised_state_event():
    """A guard against the list drifting: if one of these leaves `_STATE_EVENTS`
    the runner drops it silently, and every test above still passes because they
    build their own payload."""
    from rigma import _dsh_runner as runner
    for name in ("compaction/start", "compaction/summary",
                 "compaction/prune", "compaction/end"):
        assert name in runner._STATE_EVENTS, name


def test_the_runner_no_longer_treats_compaction_as_internal_chatter():
    """`_notice_text` matched six keywords against a truncated summary, so these
    fell through it and a DSH turn compacting its context showed nothing."""
    for kind in ("compaction/start", "compaction/summary",
                 "compaction/prune", "compaction/end"):
        evs = _project(_session_event(kind, {"compactionId": "c1", "turn": 1}))
        assert evs, kind
        assert evs[0]["type"] == "state", kind

# --------------------------------------------------------------------------
# DSH's LLM retry. `dsh-llm-retry` is a dependency of `sdk-minimal` ITSELF, so it
# is always loaded and always firing — this is not an opt-in capability. When a
# request to the model fails and DSH schedules another attempt it writes these two
# events and says nothing on any surface Rigma reads, so against a local
# llama-server that stalls or returns a malformed tool call the visible effect was
# a turn that sat there producing nothing.
#
# Shapes quoted from DSH's own declarations
# (packages/llm/llm-retry/src/types.ts:16-42, LlmFailure at
# packages/llm/llm/src/types.ts:40):
#   llm/retry          {retryId, turn, step, provider, mode, policyKey, retry,
#                       delayMs, failure: {message, code, status?}, maxRetries?}
#   llm/retry-started  {retryId, turn, step, retry}
#
# `maxRetries` is ABSENT for `mode: "always"`, which is an unbounded policy. That
# absence is the fact the UI needs, so it must survive the bridge as an absence
# rather than being defaulted to a number.


def test_a_scheduled_retry_carries_its_attempt_and_failure():
    evs = _project(_session_event("llm/retry", {
        "retryId": "r1", "turn": 1, "step": 1, "provider": "deepseek-official",
        "mode": "normal", "policyKey": "default", "retry": 2, "maxRetries": 5,
        "delayMs": 1500,
        "failure": {"message": "connection reset", "code": "TRANSPORT_ERROR"},
    }))
    assert len(evs) == 1
    assert evs[0]["type"] == "state"
    assert evs[0]["event"] == "llm/retry"
    d = evs[0]["data"]
    assert d["retry"] == 2
    assert d["maxRetries"] == 5
    assert d["delayMs"] == 1500
    # The failure is an OBJECT, not a flattened string: the code is the
    # provider-neutral routing fact and the message is the human one.
    assert d["failure"]["code"] == "TRANSPORT_ERROR"
    assert d["failure"]["message"] == "connection reset"


def test_an_unbounded_retry_keeps_maxretries_absent():
    """`mode: "always"` has no cap. If the bridge invented one, the UI would claim
    "attempt 3 of 0" for a policy that retries forever."""
    evs = _project(_session_event("llm/retry", {
        "retryId": "r2", "turn": 1, "step": 1, "provider": "p",
        "mode": "always", "policyKey": "k", "retry": 3, "delayMs": 500,
        "failure": {"message": "boom", "code": "TRANSPORT_ERROR"},
    }))
    assert "maxRetries" not in evs[0]["data"]


def test_a_retry_started_marks_the_wait_over():
    evs = _project(_session_event("llm/retry-started", {
        "retryId": "r1", "turn": 1, "step": 1, "retry": 2,
    }))
    assert evs[0]["event"] == "llm/retry-started"
    assert evs[0]["data"]["retryId"] == "r1"


def test_a_retry_status_survives_when_the_provider_gave_one():
    evs = _project(_session_event("llm/retry", {
        "retryId": "r3", "turn": 1, "step": 1, "provider": "p", "mode": "normal",
        "policyKey": "k", "retry": 1, "maxRetries": 3, "delayMs": 100,
        "failure": {"message": "rate limited", "code": "HTTP_ERROR", "status": 429},
    }))
    assert evs[0]["data"]["failure"]["status"] == 429


def test_both_retry_events_are_recognised_state_events():
    """A guard against the list drifting: if either leaves `_STATE_EVENTS` the
    runner drops it silently and a retrying turn goes back to looking frozen."""
    from rigma import _dsh_runner as runner
    for name in ("llm/retry", "llm/retry-started"):
        assert name in runner._STATE_EVENTS, name


def test_the_runner_no_longer_treats_a_retry_as_internal_chatter():
    for kind in ("llm/retry", "llm/retry-started"):
        evs = _project(_session_event(kind, {"retryId": "r1", "retry": 1}))
        assert evs, kind
        assert evs[0]["type"] == "state", kind



# --- R4-SKILL-1: pointing DSH's skill provider at Rigma's skills -------------
#
# The capability patch mounts `skill`, `skill-filesystem` and `tool-skill`, which
# makes the skill TOOL exist. It does not make Rigma's skills REACHABLE: the
# provider scans `<dshHome>/skills` (= ~/.rigma/dsh/skills, because the runner
# boots DSH with dsh_home=~/.rigma/dsh) and `~/.agents/skills`, while Rigma's
# Skills page writes `~/.rigma/skills`. Two disjoint stores, so an authored skill
# was invisible to every harness turn with nothing on screen saying so.


def test_the_skills_patch_points_the_provider_at_rigmas_own_directory(tmp_path):
    """The added root is Rigma's, and it is added rather than substituted.

    `config:` REPLACES a row's whole config, so `includeDefaultRoots` has to be
    restated: dropping it would silently stop a repository's own `.dsh/skills`
    from being found, which is a capability the provider had before Rigma
    touched it.
    """
    path = harness_dsh.skills_patch_file(tmp_path)
    assert path, "the skills patch was not generated"
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    row = next(r for r in doc if r.get("id") == "skill-filesystem")
    cfg = row["config"]
    assert cfg["includeDefaultRoots"] is True
    from rigma.skills import skills_dir

    dirs = [str(d) for d in cfg["customSkillDirs"]]
    assert str(skills_dir()) in dirs


def test_the_skills_patch_targets_the_row_the_capability_patch_inserts(tmp_path):
    """A patch row naming an id that does not exist is SKIPPED WITH A WARNING,
    not fatally — so a typo here would silently leave the provider unconfigured
    and look exactly like the bug this fixes."""
    path = harness_dsh.skills_patch_file(tmp_path)
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    ids = {r.get("id") for r in doc}
    inserted = {r["id"] for r in _patch_rows()}
    assert ids <= inserted, ids - inserted


def test_the_skills_patch_is_written_where_it_is_told(tmp_path):
    """It is generated per runtime into the runner's own scratch directory.

    Passing the chat's `cwd` instead would drop a generated YAML file into the
    user's project directory — a file they never asked for, in the folder their
    work lives in.
    """
    from pathlib import Path

    path = harness_dsh.skills_patch_file(tmp_path)
    assert Path(path).parent == Path(tmp_path)


def test_a_missing_skills_directory_yields_no_patch_rather_than_a_bad_root(tmp_path, monkeypatch):
    """An empty string means "leave DSH exactly as it was". A root that cannot
    resolve is worse than no extra root."""
    monkeypatch.setattr("rigma.skills.skills_dir",
                        lambda: tmp_path / "does-not-exist")
    assert harness_dsh.skills_patch_file(tmp_path) == ""


def test_the_skills_patch_composes_over_the_capability_patch():
    """The two patches must be independent: the capability patch INSERTS the row
    and this one CONFIGURES it. If the insert and the config ever landed in one
    layer, order would decide the result and one of them would win."""
    caps = yaml.safe_load(open(harness_dsh.capability_patch(), encoding="utf-8"))
    for entry in caps:
        for row in entry.get("insert") or []:
            assert "config" not in row or row["id"] != "skill-filesystem", (
                "the capability patch now configures skill-filesystem; the "
                "generated skills patch would silently override it")


# --- R6-MCP: Rigma's own tools must reach DSH, not only mcode -----------------
#
# WHY THIS EXISTS. `harness_mcode.ensure_mcp` has given mcode an `mcp.json`
# naming Rigma's four MCP tools since R4. DSH got nothing — `patch_file` passed
# no MCP configuration and no shipped profile mounts `dsh-mcp-client` — so a DSH
# turn could not search the user's indexed documents or remember anything, and
# NO LINE ANYWHERE SAID SO. A capability menu that lists what a backend can do,
# with no entry for the four tools one backend has and the other lacks, is the
# same class of defect as the plan-mode claim R5-PLANMODE removed.
#
# These tests pin the generated row against the plugin's REAL schema, because a
# row that fails schema validation does not fail the harness: `failOnStartupError`
# defaults to false, so a bad row logs a warning and the four tools are simply
# absent — silently, which is exactly the bug being fixed.

_MCP_ROW_ID = "mcp-rigma"
_MCP_PLUGIN = "@deepseek-ai/dsh-mcp-client"


def _mcp_rows(tmp_path, monkeypatch, *, offered=True, cwd=None):
    """The generated MCP patch's ROWS, with the roster gate stubbed.

    R6-MCP-INSERT: this used to return the loaded document, and the document is a list
    of patch OPERATIONS — so the tests indexed `doc[0]` and got the operation rather than
    the row. They passed only because the file used to be a bare row, which is the very
    thing that was wrong: a bare row is read as a REPLACE, so DSH answered
    `patch: entry "mcp-rigma" not found`, dropped the overlay, and exited 0. Rigma's own
    tools therefore never reached a DSH turn while the capability menu said they were
    mounted.

    The helper now unwraps the operation, which means every test below is asserting on
    the row DSH would actually insert — and `test_the_mcp_patch_is_an_insert_operation`
    pins the wrapper itself, so the two cannot drift apart again.
    """
    from rigma import mcp_server

    monkeypatch.setattr(mcp_server, "offered",
                        lambda **kw: (["remember"] if offered else []))
    path = harness_dsh.mcp_patch_file(tmp_path, cwd=cwd)
    if not path:
        return None
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    # Every operation, flattened to its rows. A `replace`/`remove` would contribute
    # nothing, and the assert below is what catches a patch that has stopped inserting.
    rows = [row for entry in doc if isinstance(entry, dict)
            for row in (entry.get("insert") or [])]
    assert rows, f"the MCP patch inserts no rows: {doc}"
    return rows


def test_the_mcp_patch_is_an_insert_operation(tmp_path, monkeypatch):
    """R6-MCP-INSERT. The wrapper is load-bearing, and its absence was SILENT.

    A patch list is a list of operations. A bare row is a REPLACE, and a replace of a row
    that does not exist is not an error DSH stops for — it warns on stderr, drops the
    whole overlay, and exits 0. So this is the difference between "DSH has Rigma's four
    tools" and "nothing happened", with no failure to notice.
    """
    from rigma import mcp_server

    monkeypatch.setattr(mcp_server, "offered", lambda **kw: ["remember"])
    path = harness_dsh.mcp_patch_file(tmp_path, cwd=None)
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    assert isinstance(doc, list) and len(doc) == 1, doc
    assert list(doc[0]) == ["insert"], (
        f"a bare row is read as a REPLACE and DSH silently drops it: {doc}")


def test_the_mcp_patch_mounts_the_plugin_the_runtime_actually_ships(tmp_path, monkeypatch):
    """The row must name a package that exists, or it is a row that does nothing.

    `dsh-mcp-client` is a NAMESPACE plugin: it exports `name`, `inject` and
    `apply` and deliberately no default export, so the row's `name` must be the
    package name and the id must be ours.
    """
    rows = _mcp_rows(tmp_path, monkeypatch)
    assert rows, "the MCP patch was not generated"
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["id"] == _MCP_ROW_ID
    assert row["name"] == _MCP_PLUGIN


def test_the_mcp_row_satisfies_every_required_field_of_the_real_schema(tmp_path, monkeypatch):
    """`transport`, `serverName` and `command` are the fields with NO default.

    Read from the plugin's own schema: `serverName` must match
    `[A-Za-z0-9_-]{1,32}` and be unique, and `command` is required for stdio.
    A row missing any of them is rejected at boot — quietly, because
    `failOnStartupError` defaults to false.
    """
    import re

    row = _mcp_rows(tmp_path, monkeypatch)[0]
    cfg = row["config"]
    assert cfg["transport"] == "stdio"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}", cfg["serverName"]), cfg["serverName"]
    assert cfg["command"], "stdio requires a command"
    assert cfg["args"] == ["-m", "rigma.mcp_server"]


def test_the_mcp_command_is_an_absolute_interpreter_not_a_bare_python(tmp_path, monkeypatch):
    """`sys.executable`, never `python`.

    The server imports Rigma, so it must be launched by the interpreter that has
    Rigma installed. A bare `python` may be a different one, and the failure is
    silent: the server does not start and the four tools are simply absent.
    """
    import os
    import sys

    cfg = _mcp_rows(tmp_path, monkeypatch)[0]["config"]
    assert cfg["command"] == sys.executable
    assert os.path.isabs(cfg["command"])


def test_the_mcp_env_carries_what_dsh_would_otherwise_scrub_away(tmp_path, monkeypatch):
    """DSH hands the child a SCRUBBED ambient environment plus exactly this map.

    So the two variables the server needs cannot be inherited — they must be
    written here. `RIGMA_MCP_ALLOW_CODE` is what makes `undo_last_change` exist
    at all (the server is pessimistic by default) and `RIGMA_MCP_WORKSPACE` is
    the chat's directory, passed rather than guessed.
    """
    cfg = _mcp_rows(tmp_path, monkeypatch, cwd=r"C:\some\project")[0]["config"]
    assert cfg["env"]["RIGMA_MCP_ALLOW_CODE"] == "1"
    assert cfg["env"]["RIGMA_MCP_WORKSPACE"] == r"C:\some\project"
    assert cfg["env"]["RIGMA_HOME"]


def test_the_mcp_workspace_is_omitted_rather_than_guessed_when_there_is_none(tmp_path, monkeypatch):
    """A tool that silently operated on the wrong directory is worse than one
    that refused, so an absent cwd must not become some default directory."""
    cfg = _mcp_rows(tmp_path, monkeypatch, cwd=None)[0]["config"]
    assert "RIGMA_MCP_WORKSPACE" not in cfg["env"]


def test_no_mcp_patch_is_written_when_the_roster_is_empty(tmp_path, monkeypatch):
    """An empty roster still costs a process launch, and DSH starts a TEMPORARY
    PROBE PROCESS before the serving one. Paying that for nothing is worse than
    the tool appearing a turn later than it could."""
    assert _mcp_rows(tmp_path, monkeypatch, offered=False) is None


def test_the_mcp_patch_is_written_where_it_is_told(tmp_path, monkeypatch):
    """Generated per runtime into the runner's scratch directory.

    Writing it beside the chat's cwd would drop a generated YAML file into the
    user's own project — a file they never asked for, in the folder their work
    lives in.
    """
    from pathlib import Path
    from rigma import mcp_server

    monkeypatch.setattr(mcp_server, "offered", lambda **kw: ["remember"])
    path = harness_dsh.mcp_patch_file(tmp_path, cwd=None)
    assert Path(path).parent == Path(tmp_path)


def test_the_mcp_patch_does_not_fail_the_turn_when_it_cannot_be_built(tmp_path, monkeypatch):
    """A missing Rigma tool set is a smaller loss than no harness turn at all."""
    from rigma import mcp_server

    def boom(**kw):
        raise RuntimeError("roster exploded")

    monkeypatch.setattr(mcp_server, "offered", boom)
    assert harness_dsh.mcp_patch_file(tmp_path, cwd=None) == ""


def test_the_mcp_patch_never_configures_a_row_the_capability_patch_inserts(tmp_path, monkeypatch):
    """It INSERTS its own row, so it must not collide with a shipped id.

    A patch row naming an id that does not exist is skipped with a warning, and
    two layers configuring one row means order decides the result.
    """
    doc = _mcp_rows(tmp_path, monkeypatch)
    assert doc, "the MCP patch was not generated"
    inserted = {r["id"] for r in _patch_rows()}
    assert _MCP_ROW_ID not in inserted, (
        "the capability patch now inserts mcp-rigma; the generated MCP patch "
        "would silently override or duplicate it")


def test_dsh_declares_the_mcp_capability_it_now_has():
    """The disclosure must match the code, in the direction that matters.

    DSH gained these four tools, so the menu must say so — and `_LEAVES_BEHIND`
    must stop claiming RAG and undo are left behind, because that would now be a
    FALSE NEGATIVE, which is the same defect as the plan-mode claim.
    """
    from rigma.harness import DSH, BACKENDS

    entry = BACKENDS[DSH]
    caps = " ".join(entry.capabilities)
    assert "mcp__rigma__remember" in caps
    assert "mcp__rigma__recall" in caps
    assert "mcp__rigma__search_my_documents" in caps
    assert "mcp__rigma__undo_last_change" in caps
    behind = " ".join(entry.unsupported)
    assert "no MCP equivalent" in behind
    assert "Rigma's tools (image-by-reference, undo, sample_files, RAG, methods)" not in behind, (
        "the old blanket claim is back; RAG and undo DO reach both backends now")


# --- R6-DRIFT: the ACP control plane, guarded across all three layers ---------
#
# WHY. An ACP event travels four hops: the mapper in `harness_mcode_acp` names it,
# `serve.py` must route that name to an SSE name, and `chatStore.ts` must have an
# arm for the SSE name. A name that stops matching at ANY hop is dropped SILENTLY:
# no error, no log, and the panel simply never shows the queue, the delegation tree
# or the plan review. Nothing else in the suite would notice, which is exactly why
# this guard exists rather than a comment asking people to keep them in step.
#
# It is the same shape as the R4-DRIFT guard below, extended to the new names.


# The ONE ACP event with no store arm, on purpose. `acp_current_session` says which
# session mcode considers current, which is meaningful only to mcode: `serve.py`
# routes it and then drops it explicitly, so that a later reader cannot mistake it
# for an unhandled event. It is named here, in one place, rather than excluded
# silently inside two functions.
_ACP_NO_STORE_ARM = {"acp_current_session"}


def _acp_event_names() -> set:
    """Every `event=` the ACP mapper can emit, read from its source.

    Read rather than restated: a constant here would be a fourth place to keep in
    step, which is the problem this test is about.
    """
    import re

    from rigma import harness_mcode_acp

    src = pathlib.Path(harness_mcode_acp.__file__).read_text(encoding="utf-8")
    return set(re.findall(r'event="([a-z_/]+)"', src))


def _serve_routes() -> set:
    """Every backend event name `serve.py` accepts.

    TWO SHAPES, and missing the second made this guard report routed events as
    unrouted. `serve.py` matches most names exactly (`_ev == "goal"`), but a family
    is matched by prefix (`_ev.startswith("approval/")`), which is how one SSE name
    carries `approval/asked`, `approval/decided` and `approval/policy`. A guard that
    only understood `==` would have been "fixed" by weakening it.
    """
    import re

    from rigma import serve

    src = pathlib.Path(serve.__file__).read_text(encoding="utf-8")
    # The character class includes `/` because backend-side names carry one —
    # `plan/mode` and `session/title`.
    exact = set(re.findall(r'_ev == "([a-z_/]+)":', src))
    prefixes = set(re.findall(r'_ev\.startswith\("([a-z_/]+)"\)', src))
    return exact | prefixes


def _is_routed(name: str) -> bool:
    """Whether `serve.py` accepts this name, by either shape."""
    routes = _serve_routes()
    return name in routes or any(
        p.endswith("/") and name.startswith(p) for p in routes)


def _store_arms() -> set:
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "frontend-v2" / "src" / "chat"
    src = (root / "chatStore.ts").read_text(encoding="utf-8")
    return set(re.findall(r'case "([a-z_]+)":', src))


def test_every_acp_event_the_mapper_emits_is_routed_by_serve():
    """Hop 1 -> 2. An unrouted name reaches the frontend as nothing at all."""
    emitted = _acp_event_names()
    missing = sorted(n for n in emitted if not _is_routed(n))
    assert not missing, f"the ACP mapper emits these and serve.py drops them: {missing}"


def test_every_acp_event_serve_routes_has_a_store_arm():
    """Hop 2 -> 3. A routed name with no arm falls through the reducer's default
    case, so the panel stays empty while the server believes it delivered."""
    routed = {n for n in _serve_routes()
              if n.startswith("acp_")} - _ACP_NO_STORE_ARM
    arms = _store_arms()
    missing = sorted(n for n in routed if n not in arms)
    assert not missing, f"serve.py emits these and chatStore.ts ignores them: {missing}"
    # And the excluded one really is excluded on purpose, not by accident.
    assert _ACP_NO_STORE_ARM <= _serve_routes(), (
        "acp_current_session is excluded from the store check, so serve.py must at "
        "least route it — otherwise the exclusion hides a genuinely dropped event")


def test_the_acp_drift_guard_would_actually_catch_a_break(monkeypatch):
    """A guard that cannot fail is not a guard.

    Checked by asserting the three sets are non-empty and that a deliberately
    invented name is reported missing by the same expression the guard uses.
    """
    emitted = _acp_event_names()
    assert emitted, "the mapper regex found no event names — the guard is broken"
    assert {n for n in _serve_routes() if n.startswith("acp_")}, "no ACP routes found"
    assert _store_arms(), "no store arms found"
    fake = "acp_definitely_not_real"
    assert fake not in _serve_routes()
    assert fake not in _store_arms()


def test_the_goal_update_reuses_the_existing_goal_channel():
    """`mcode/session/goal_update` is the SAME fact DSH reports as `goal/change`, so
    it must travel as `goal` rather than growing a second panel for one concept."""
    from rigma import harness_mcode_acp

    out = harness_mcode_acp.map_acp_update({
        "method": "mcode/session/goal_update",
        "params": {"sessionId": "s1", "goal": {"objective": "x", "status": "active"}},
    })
    assert out[0].event == "goal"
    assert "goal" in _serve_routes()


# --- R4-DRIFT: the four capabilities the UI draws, guarded the same way ------
#
# WHY. Governance, compaction and retry each already had a drift guard — a test
# that fails if the event name leaves `_STATE_EVENTS`, because the runner then
# drops it SILENTLY and the UI shows nothing with no error anywhere. The four
# capabilities the AgentState panel draws had no such guard, so the same silent
# loss was possible for a goal, a todo list, a plan-mode switch or a subagent.
#
# `serve.py` maps each of these to an SSE name, and the frontend reducer keys off
# that name; if a name here stops being a state event, the panel simply goes
# quiet. Nothing else in the suite would notice.

# backend event name -> the SSE name serve.py must emit for it
_PANEL_EVENTS = {
    "goal/change": "goal",
    "todo/write": "todos",
    "plan/mode": "plan_mode",
}


def test_every_panel_capability_is_still_a_state_event():
    """A guard against the list drifting: if one of these leaves `_STATE_EVENTS`,
    the runner drops it and the panel renders nothing."""
    for name in _PANEL_EVENTS:
        assert name in runner._STATE_EVENTS, name


def test_the_panel_capabilities_are_not_treated_as_internal_chatter():
    """Being IN the list is not enough — the projection has to yield them."""
    for name in _PANEL_EVENTS:
        evs = _project(_session_event(name, {}))
        assert evs, name
        assert evs[0]["type"] == "state", name


def test_the_subagent_lifecycle_is_still_recognised():
    """The subagent rows come from the top-level lifecycle pair, not from
    `_STATE_EVENTS` alone, so they need their own guard."""
    for name, payload in (
        ("subagent.started", {"parentSessionId": "p", "childSessionId": "c"}),
        ("subagent.finished", {"childSessionId": "c", "status": "ok"}),
    ):
        evs = _project(_notif(name, payload))
        assert evs, name
        assert evs[0]["type"] == "state", name
        assert evs[0]["event"] == name, name


def test_the_mcode_names_are_bridged_by_the_server_not_the_runner():
    """R4-MCODE-1, pinned so the two naming schemes cannot silently diverge again.

    The RUNNER normalises DSH's names; mcode's adapter emits its OWN names
    (`goal`, `todos`, `subagent`) and they are bridged in `serve.py`. This test
    records that split, because the bug it fixes was exactly a name that no layer
    claimed: the adapter emitted it, `_STATE_EVENTS` did not know it, and
    `serve.py`'s elif chain did not match it.
    """
    from rigma import harness_mcode

    import inspect

    src = inspect.getsource(harness_mcode)
    for name in ('event="goal"', 'event="todos"', 'event="subagent"'):
        assert name in src, name
    # And the DSH spelling must NOT be what mcode emits, or the bridge in
    # serve.py is dead code and the mcode path is unreachable again.
    for dsh_name in ('event="goal/change"', 'event="todo/write"'):
        assert dsh_name not in src, dsh_name


# --- R5-STEPTEXT: a multi-step turn must not lose its earlier prose ----------
#
# `assistant/message` carries the step's text AND its token accounting. The runner
# kept only `usage` and dropped `data.message` on the floor. That was a real loss,
# because the SDK's `final_response` walks the events in REVERSE and returns the
# FIRST `assistant/message` it finds (python/sdk/src/deepseek_harness/api.py:211-228)
# — so it returns only the LAST step's text. Every earlier step's prose reached
# nobody, and a turn with tool calls read as if the agent had said nothing between
# them.
#
# The fix streams each step's prose as it arrives, which means the final answer
# would arrive twice unless something subtracts it. `_Streamed` is that something,
# and `harness_dsh._Run.spoke` is what stops an empty `done` being misread as a
# failed model call.


def _assistant_message(text: str, usage: dict | None = None):
    return _session_event("assistant/message", {
        "turn": 1, "step": 1,
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": text}]},
        **({"usage": usage} if usage else {}),
    })


def test_a_steps_prose_is_emitted_and_not_only_its_usage():
    out = runner._project(_assistant_message("thinking out loud", {"input": 5}))
    kinds = [e["type"] for e in out]
    assert "text" in kinds, out
    assert any(e["type"] == "state" and e["event"] == "usage" for e in out), out
    # Prose FIRST, so the transcript reads in order.
    assert kinds[0] == "text"
    assert out[0]["text"] == "thinking out loud"


def test_a_message_with_no_text_still_reports_its_usage():
    """A step that only called tools has no prose but does have a token count."""
    out = runner._project(_session_event("assistant/message", {
        "turn": 1, "step": 1,
        "message": {"role": "assistant",
                    "content": [{"type": "tool_call", "name": "x"}]},
        "usage": {"input": 3},
    }))
    assert [e["type"] for e in out] == ["state"], out


def test_the_text_comes_from_the_message_not_the_envelope():
    """The block list is `data.message.content`, the same shape the SDK's own
    `final_response` reads — so the two cannot drift apart."""
    out = runner._project(_assistant_message("from the message"))
    assert out[0]["text"] == "from the message"


def test_every_step_of_a_multi_step_turn_is_emitted():
    """The bug: only the LAST step's text survived."""
    streamed = runner._Streamed()
    got = []
    for step, text in enumerate(["first", "second", "third"], start=1):
        for e in runner._project(_session_event("assistant/message", {
            "turn": 1, "step": step,
            "message": {"content": [{"type": "text", "text": text}]},
        }), streamed):
            if e["type"] == "text":
                got.append(e["text"])
    assert got == ["first", "second", "third"], got


def test_the_final_answer_is_not_emitted_twice():
    """The cost of streaming: `done` would repeat the last message."""
    streamed = runner._Streamed()
    for text in ("first", "the answer"):
        runner._project(_assistant_message(text), streamed)
    # final_response returns the LAST assistant/message, which was already sent.
    assert streamed.remaining("the answer") == ""


def test_only_the_unstreamed_tail_is_emitted():
    """If final_response says more than the streamed copy, the difference goes
    out rather than being dropped.

    The two must share a genuine PREFIX for this to apply — a streamed snapshot of
    a message that then finished streaming. ("the ans" and "the answer" look like
    that pair and are NOT: the fifth character is `e` against `a`. That is a
    DIVERGENCE, covered by `test_a_divergent_final_answer_is_emitted_whole`.)
    """
    streamed = runner._Streamed()
    runner._project(_assistant_message("the answ"), streamed)
    assert streamed.remaining("the answer") == "er"


def test_a_divergent_final_answer_is_emitted_whole():
    """A guess here would either duplicate a reply or silently drop one, so an
    unmatched final is returned whole."""
    streamed = runner._Streamed()
    runner._project(_assistant_message("something else"), streamed)
    assert streamed.remaining("the answer") == "the answer"


def test_an_empty_final_emits_nothing():
    streamed = runner._Streamed()
    runner._project(_assistant_message("a reply"), streamed)
    assert streamed.remaining("") == ""


def test_a_streamed_turn_records_that_it_spoke():
    """`_Run.spoke` is what tells an empty `done` apart from a dead model call.
    Without it, EVERY successful multi-step turn would be reported as a failure,
    because `done` is now legitimately empty."""
    from rigma import harness_dsh

    state = harness_dsh._Run()
    assert state.spoke is False
    ev = harness_dsh._event_for({"type": "text", "text": "hello"}, state)
    assert ev is not None and ev.kind == "text"
    assert state.spoke is True


def test_thinking_and_notices_do_not_count_as_speaking():
    """Only real prose makes an empty `done` legitimate: `thinking` is private
    reasoning and `notice` is backend chatter."""
    from rigma import harness_dsh

    for payload in ({"type": "thinking", "text": "hmm"},
                    {"type": "notice", "text": "working"}):
        state = harness_dsh._Run()
        harness_dsh._event_for(payload, state)
        assert state.spoke is False, payload


def test_plan_mode_is_not_mounted_and_why():
    """Plan mode is UNREACHABLE, so the patch must not mount it.

    The row used to be here, copied from the shipped `sdk`/`web` profiles. In
    those profiles `dsh-commands` and `dsh-user-questions` are mounted; in
    sdk-minimal (and in this patch) neither is. Plan mode has exactly two
    registrations (`packages/plan/plan-mode/src/index.ts`):

      :225  ctx.inject(['commands'], ...)   -> the `/plan` command, the ONLY way IN
      :273  ctx.tools.register(EXIT_PLAN_MODE)  -> the only way OUT, and it is
             registered UNCONDITIONALLY, outside that inject

    So with `commands` unmounted there is no way in, while `exit_plan_mode` is
    still in the tool catalog and throws `is only available in plan mode` on its
    first call. Mounting it therefore advertises a capability AND hands the model a
    tool that can only fail — strictly worse than not mounting it.

    This is a DRIFT GUARD, not a preference: if `dsh-commands` and
    `dsh-user-questions` are ever mounted, delete this test and restore the row
    (the config is preserved in a comment beside where it used to live). Until
    then, re-adding the row silently is the regression this catches.
    """
    ids = {str(r.get("id")) for r in _patch_rows()}
    assert "plan-mode" not in ids, (
        "plan-mode is mounted again, but nothing can enter or leave plan mode: "
        "`dsh-commands` (the /plan command) and `dsh-user-questions` "
        "(exit_plan_mode's review channel) are both unmounted. Mount those first."
    )
    # The other two plan-mode-adjacent registrations the audit found, so the
    # guard covers the capability rather than one row's id.
    names = {str(r.get("name")) for r in _patch_rows()}
    assert "@deepseek-ai/dsh-plan-mode" not in names


# --- B2: an unknown session-event type must not be dropped in silence ---------
#
# THE DEFECT. `_project` handled 21 `_STATE_EVENTS` plus `tool/call`,
# `tool/result`, `assistant/message` and the two top-level `subagent.*`. Anything
# else fell through to `_notice_text`, which returned `None` unless
# `method + inner-type` happened to contain one of six substrings
# (`tool, error, fail, denied, retry, timeout`). So 27 of DSH's 51 session-event
# types were discarded with no branch, no notice and no log — including
# `command/run` and `command/done`, where a person watching a turn could not tell
# that a command had run. This is information loss, not a crash, which is exactly
# why it went unnoticed.
#
# THE FIX. An unknown-but-well-formed inner event is projected as the EXISTING
# notice shape (one string, no second wire format), naming the type and the
# identifying fields the event actually carries; the first occurrence of each type
# is logged once; and a projector that raises yields a notice instead of vanishing
# into the old `except: pass`.
#
# WHAT IS NOT HERE. The state chain's missing `else` (`serve.py:2449-2451`) and
# the UI rendering of these notices are a later wave, per the item's scope.

# The concrete types the probe confirmed are dropped on the unmodified base
# (`.scratch/orchestrator/b2-probe-dropped.py`; the 51-type vocabulary is doc 27
# §1.1). Kept here as the oracle for "unknown" — a type that stops being unknown
# must be moved into `_STATE_EVENTS` or given its own branch, not left ambiguous.
DROPPED_ON_BASE = (
    "step/start", "step/end",
    "request/header", "request/context",
    "agent/inbox/spliced",
    "session/end-seed",
    "assistant/attempt", "assistant/chunk",
    "user/message", "system/message",
    "session-log-deepseek/delivery-accepted",
    "session/title-llm-request",
    "web/deepseek-search-llm-request",
    "hook/invoked", "hook/result",
    "command/run", "command/done",
    "model/selection",
    "agent-preset/selected",
    "subagent/model-selection-policy",
    "schedule/change",
    "feedback/record",
)


def test_every_type_dropped_on_base_now_yields_exactly_one_notice(monkeypatch):
    """The whole item in one assertion, over the verified drop list."""
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    for kind in DROPPED_ON_BASE:
        out = _project(_session_event(kind, {}))
        assert len(out) == 1, (kind, out)
        assert out[0]["type"] == "notice", (kind, out)
        assert kind in out[0]["text"], (kind, out)


def test_an_unknown_event_names_the_identifying_fields_it_has(monkeypatch):
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    out = _project(_session_event("command/run", {
        "command": "/compact", "sessionId": "s1", "status": "running",
        "callId": "c1",
    }))
    assert len(out) == 1, out
    text = out[0]["text"]
    assert "command/run" in text
    assert "command=/compact" in text
    assert "sessionId=s1" in text
    assert "status=running" in text
    assert "callId=c1" in text


def test_the_notice_never_invents_a_field_the_event_lacks(monkeypatch):
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    text = _project(_session_event("command/done", {"status": "ok"}))[0]["text"]
    assert "command/done" in text and "status=ok" in text
    assert "command=" not in text, text
    assert "sessionId=" not in text, text
    assert "callId=" not in text, text


def test_an_unknown_notice_never_carries_message_content(monkeypatch):
    """§0.3: a type name, an id and a status only — never prose.

    The allow-list is the mechanism, so this pins that a payload full of content
    cannot leak through it.
    """
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    text = _project(_session_event("command/run", {
        "command": "/compact",
        "text": "SECRET PROSE",
        "message": {"content": [{"type": "text", "text": "SECRET PROSE"}]},
        "prompt": "SECRET PROSE",
    }))[0]["text"]
    assert "SECRET" not in text, text
    assert "command=/compact" in text


def test_an_unknown_notice_is_one_line(monkeypatch):
    """A value with a newline in it must not turn one notice into two."""
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    text = _project(_session_event("command/run", {
        "command": "/goal\nship\tit",
    }))[0]["text"]
    assert "\n" not in text and "\r" not in text and "\t" not in text, repr(text)
    assert "command=/goal ship it" in text


def test_the_first_occurrence_of_an_unknown_type_is_logged_once(monkeypatch, caplog):
    """Several occurrences, one line — the log is a discovery aid, not a stream."""
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    with caplog.at_level(logging.WARNING, logger="rigma._dsh_runner"):
        for _ in range(3):
            _project(_session_event("command/run", {"command": "/x"}))
    hits = [r for r in caplog.records if "command/run" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert hits[0].levelno == logging.WARNING


def test_a_different_unknown_type_gets_its_own_log_line(monkeypatch, caplog):
    monkeypatch.setattr(runner, "_WARNED_UNKNOWN_TYPES", set())
    with caplog.at_level(logging.WARNING, logger="rigma._dsh_runner"):
        _project(_session_event("command/run", {}))
        _project(_session_event("model/selection", {}))
    seen = {r.getMessage().split("'")[1] for r in caplog.records
            if "'" in r.getMessage()}
    assert {"command/run", "model/selection"} <= seen, seen


def test_handled_types_project_exactly_as_before():
    """The oracle for "nothing else moved": 2-3 handled shapes, pinned exactly."""
    assert _project(_session_event("goal/change", {"operation": "create"})) == [
        {"type": "state", "event": "goal/change", "data": {"operation": "create"}}]
    assert _project(_session_event("tool/call",
                                   {"name": "read", "callId": "c1"})) == [
        {"type": "tool", "name": "read", "id": "c1"}]
    assert _project(_session_event("assistant/message",
                                   {"usage": {"input": 5}})) == [
        {"type": "state", "event": "usage", "data": {"input": 5}}]


def test_a_top_level_notification_without_an_event_type_is_unchanged():
    """`session.status` has no inner event TYPE to name, so the 51-type vocabulary
    does not apply and the old keyword filter still decides. Otherwise every turn
    would emit a "working" notice."""
    assert _project(_notif("session.status", {"status": "working"})) == []
    assert _project(_notif("session.status", {"status": "failed"})) == [
        {"type": "notice", "text": "session.status failed"}]


def test_a_projector_that_raises_yields_a_notice_and_does_not_propagate(
        monkeypatch, caplog):
    """`except: pass` meant a projector bug ate the event invisibly."""
    sent: list[dict] = []
    monkeypatch.setattr(runner, "_emit", sent.append)

    def _boom(*_a, **_k):
        raise RuntimeError("projector bug")

    monkeypatch.setattr(runner, "_project", _boom)
    with caplog.at_level(logging.WARNING, logger="rigma._dsh_runner"):
        runner._forward(_notif("session.event", {"event": {"type": "command/run"}}))
    assert len(sent) == 1, sent
    assert sent[0]["type"] == "notice", sent
    assert "could not be projected" in sent[0]["text"], sent
    assert any("projector" in r.getMessage().lower() for r in caplog.records)


def test_forward_emits_what_the_projector_returns(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(runner, "_emit", sent.append)
    runner._forward(_session_event("goal/change", {"operation": "create"}))
    assert sent == [{"type": "state", "event": "goal/change",
                     "data": {"operation": "create"}}]
