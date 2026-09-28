"""Run exactly one DeepSeek Harness turn, in a process Rigma is allowed to kill.

WHY THIS IS A SEPARATE MODULE AND A SEPARATE PROCESS.

Rigma's process must never import `deepseek_harness`. The SDK is not installed in
Rigma's venv, its runtime is a Node CLI, and a wedged turn must not be able to
wedge the server that is streaming a chat. So the import happens HERE, in a child
whose PYTHONPATH names the checkout's `python/sdk/src` for the length of one turn,
and whose entire contract with the parent is one JSON object per stdout line.

The protocol is deliberately dumb: one job object in on stdin, NDJSON out, no
shared state. Diagnostics go to stderr — including anything the SDK decides to
print, because stdout is reserved for events and a stray `print()` would corrupt
the stream. `close()` runs in `finally` so the Node child is reaped even when the
turn failed.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_CLI_REL = ("python", "sdk-runtime", "node_modules", ".bin", "dsh.CMD")

# Captured at import: after main() redirects sys.stdout, this is the only handle
# that still points at the event stream.
_STDOUT = sys.stdout


def _emit(obj: dict) -> None:
    """One JSON object per line, and nothing else, ever."""
    _STDOUT.write(json.dumps(obj, default=str) + "\n")
    _STDOUT.flush()


# What a person watching a turn actually wants to see. DSH emits its whole
# internal lifecycle as notifications — measured 2026-09-21, a one-line turn
# produced fifteen of them (agent/inbox/spliced, step/start, request/header,
# session/title, delivery-accepted…) and forwarding every one buried the reply
# in noise on a progress line. Tool activity and failures are the signal.
_NOTICE_WORTHY = ("tool", "error", "fail", "denied", "retry", "timeout")

# Session-event types Rigma renders as STRUCTURED state rather than as a
# progress line. These are the capabilities the `sdk-minimal` profile did not
# mount at all until Rigma patched them in (see
# data/dsh/agent-capabilities.patch.yml) — so before that patch this table had
# nothing to carry, and afterwards it is what makes the capability visible.
#
# These are passed through as their WHOLE payload, not unwrapped to the one
# interesting field. `goal/change` is the reason: its payload is
# {operation, goal: <snapshot>, roundsStarted, createdAt, updatedAt}, and
# handing on only the snapshot throws away the round count and the operation —
# which is exactly the context a UI needs to say "round 4 of 60" or "paused".
_STATE_EVENTS = (
    "goal/change",
    "todo/write",
    "plan/mode",
    "subagent/descriptor",
    "subagent/catalog",
    # The governance events. These are the ones a user most needs and could not
    # see at all: DSH records when it ASKED for permission, what it was allowed
    # to do, and which confinement was in force — and every one of them is
    # `log-only` in DSH's own words, meaning they are durable and replayable but
    # never enter the model transcript. That is exactly the shape of fact a UI
    # should show and the conversation should not carry.
    #
    # Worth stating plainly, because it bounds what the UI may do with them: the
    # SDK wire has NO approval-response method (HarnessSdkRequestMap is exactly
    # initialize / session/prompt / shutdown). Approval is decided by the policy
    # engine, not by the client. So these are DISPLAY-ONLY — an audit trail — and
    # a clickable "Allow?" button would misrepresent the transport.
    "approval/asked",
    "approval/decided",
    "approval/policy",
    "sandbox/mode",
    "permission/preset",
    # The server's own title for the session, which Rigma otherwise never learns:
    # it invents its own title and the two can disagree.
    "session/title",
    # The compaction lifecycle. Rigma's NATIVE compaction already reports itself
    # (`masked`, `housekeeping`, `compacted`), but a DSH turn reported none of it,
    # so a long turn busy summarising its own context looked exactly like a turn
    # that had hung. `start`/`end` are a bracket paired by `compactionId`, and
    # `summary`/`prune` carry the shadowed token count — the number worth showing.
    "compaction/start",
    "compaction/summary",
    "compaction/prune",
    "compaction/end",
    # A retry is the same class of problem as a compaction: the turn is working and
    # says nothing. `dsh-llm-retry` is a dependency of sdk-minimal itself, so it is
    # always loaded and always firing — with a local llama-server that stalls or
    # returns a malformed tool call, the visible effect is a turn that sits there.
    "llm/retry",
    "llm/retry-started",
    # R6-WORKFLOW: programmatic tool calling, which Rigma mounts
    # (`data/dsh/agent-capabilities.patch.yml` rows `ptc-runtime`, `workflow-ptc`,
    # `tool-workflow`) and therefore fires in a real PTC turn.
    #
    # WHY THIS IS NOT COSMETIC. These four were reaching the UI as the literal
    # string "session.event tool-workflow/agent-start", and that was an ACCIDENT:
    # the notice path matches on `_NOTICE_WORTHY`, whose list contains "tool", and
    # "tool-workflow" contains it. So a run's name, each agent's label and phase,
    # each agent's outcome and the run's stop reason were all discarded — the user
    # saw a line naming an event they could not interpret and nothing else.
    #
    # The payloads (`dispositions.ts:102-105`):
    #   run-start   {runId, name}
    #   agent-start {runId, seq, label, childId, phase?}
    #   agent-end   {runId, seq, outcome}
    #   run-end     {runId, stopReason}
    # `runId` and `seq` are what let the frontend join the four into one run rather
    # than four unrelated lines.
    # DELIBERATELY ABSENT: `subagent/model-selection-policy`, and the reason is worth
    # recording because the name LOOKS like something that should be here. It is
    # appended only by `recordSubagentModelSelection`, whose sole caller is gated on
    # `config.modelSelectionSettings === true` — which defaults FALSE
    # (`tool-subagent/src/index.ts:109`) and which Rigma's two `dsh-tool-subagent` rows
    # do not set (`data/dsh/agent-capabilities.patch.yml`; a row's `config:` REPLACES
    # the object, so there is no inherited value). So the event never fires in this
    # build: adding it here would create a branch nothing can reach, which is worse
    # than its absence because it reads as coverage.
    #
    # AND IT WOULD NOT BE USEFUL IF IT DID FIRE. Its payload is
    # `{allowedModels: [{provider, model}]}` — the routes a session may pick a child
    # model from. Rigma points DSH at ONE local model, so the list would be a choice of
    # one, and the opt-in exists to let a MODEL choose its children's routes. Rigma's
    # user picks the model in the UI instead, which is the same decision made in the
    # place a person can see it. Declared under `unsupported` in harness.py.
    "tool-workflow/run-start",
    "tool-workflow/agent-start",
    "tool-workflow/agent-end",
    "tool-workflow/run-end",
)

# `assistant/message` carries the step's token accounting. It is the ONLY
# place a DSH turn reports usage, and it is worth a real number in the UI.
_USAGE_EVENT = "assistant/message"


def _message_text(data: dict) -> str:
    """The prose in one `assistant/message`, or "" when it carries none.

    The payload is `{turn, step, message, usage?, interrupted?}`, where `message`
    is an assistant message whose `content` is a list of blocks
    (`packages/session/session-format-v0-to-v1/src/dispositions.ts:47-50`, and the
    appender at `packages/core/agent-loop/src/agent.ts:406`). This is the same
    shape the SDK's own `final_response` reads
    (`python/sdk/src/deepseek_harness/api.py:218-227`), so the two cannot drift.
    """
    message = data.get("message")
    owner = message if isinstance(message, dict) else data
    content = owner.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(b.get("text") or "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


class _Streamed:
    """Text already sent for the current turn, so the final answer is not doubled.

    WHY THIS EXISTS. Every `assistant/message` is streamed as it arrives, because
    the alternative loses content: the SDK's `final_response` walks the events in
    REVERSE and returns the FIRST `assistant/message` it finds
    (`api.py:211-228`), so it returns only the LAST step's text. In a multi-step
    turn — the normal shape once tools are involved — every earlier step's prose
    reached nobody, and the transcript read as if the agent had said nothing
    between tool calls.

    The cost of streaming them is that the final answer would then arrive twice:
    once as its own `assistant/message` and again in `done`. `remaining()` subtracts
    what was already sent, so a caller can emit only the unmatched tail.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add(self, text: str) -> None:
        if text:
            self._parts.append(text)

    def remaining(self, final: str) -> str:
        """The part of `final` not already streamed.

        Compares against the LAST streamed message, NOT against all of them
        concatenated — `final_response` returns the last `assistant/message`, so a
        turn whose steps said "first" then "the answer" has a final of exactly
        "the answer", and joining the two would never match it. (That was the first
        version of this method, and it silently failed to suppress the duplicate.)

        An exact match means everything was streamed, so there is nothing left —
        the common case. A `final` that STARTS WITH the last streamed message means
        the streamed copy was an earlier snapshot of that same message and has since
        been completed, so only the tail is new. (Prefix, not suffix: the streamed
        text is the beginning of the final text, not its end.) Anything else — an
        empty final, or one that diverges — is returned whole rather than guessed
        at: showing a reply twice is a visible bug, and dropping it would be a
        silent one.
        """
        last = self._parts[-1] if self._parts else ""
        if not final or final == last:
            return ""
        if last and final.startswith(last):
            return final[len(last):]
        return final


def _event_and_data(notification) -> tuple[str, dict]:
    """The inner session event and its payload, or ("", {}) when there is none.

    DSH puts almost everything interesting inside `session.event`, whose
    `event` field is the full session-log envelope: `{type, data, ...}`.
    `subagent.started` and `subagent.finished` are the exception — they are
    top-level methods carrying their payload directly.
    """
    payload = getattr(notification, "payload", None)
    if not isinstance(payload, dict):
        return "", {}
    event = payload.get("event")
    if not isinstance(event, dict):
        return "", {}
    kind = str(event.get("type") or "")
    data = event.get("data")
    return kind, data if isinstance(data, dict) else {}


def _project(notification, streamed: "_Streamed | None" = None) -> list[dict]:
    """Every event this one notification should produce, in Rigma's vocabulary.

    Returns a LIST because one notification can carry more than one fact, and
    because the old single-line shape is still what most notifications are.
    A `session.event` for a tool call becomes a real `tool` event WITH the tool
    name and arguments — before this, the parent was told the literal string
    "session.event tool/call", so every DSH turn rendered zero tool chips and
    the tool name was unrecoverable downstream.

    `streamed` carries the text already sent for this turn, so the final answer is
    not emitted twice; see `_Streamed`. It is optional so a caller with no turn in
    flight (a test, a probe) can project one notification on its own.
    """
    if streamed is None:
        streamed = _Streamed()
    method = str(getattr(notification, "method", "") or "notification")
    payload = getattr(notification, "payload", None)
    if not isinstance(payload, dict):
        payload = {}

    # Subagent lifecycle is top-level and carries its payload directly. It also
    # did not match the old keyword filter: "subagent.finished" contains
    # "finish", not "fail", so a subagent could start AND end unseen.
    if method in ("subagent.started", "subagent.finished"):
        return [{"type": "state", "event": method, "data": payload}]

    kind, data = _event_and_data(notification)

    if kind == "tool/call":
        out = {"type": "tool", "name": str(data.get("name") or "")}
        args = data.get("arguments")
        if isinstance(args, dict):
            out["args"] = args
        call_id = data.get("callId") or data.get("id")
        if call_id:
            # The parent keys a chip by this. Without it two calls to the same
            # tool in one turn collapse onto one chip.
            out["id"] = str(call_id)
        return [out]

    if kind == "tool/result":
        out = {"type": "tool_result", "ok": not data.get("isError", False)}
        call_id = data.get("callId") or data.get("id")
        if call_id:
            out["id"] = str(call_id)
        content = data.get("content")
        if isinstance(content, str):
            out["text"] = content[:4000]
        return [out]

    if kind in _STATE_EVENTS:
        return [{"type": "state", "event": kind, "data": data}]

    if kind == _USAGE_EVENT:
        # R5-STEPTEXT: the step's PROSE goes out as well as its token count.
        #
        # This branch used to return only `usage` and drop `data.message` on the
        # floor. Because the SDK's `final_response` returns only the LAST
        # `assistant/message` (api.py:211-228), every earlier step's text reached
        # nobody — a multi-step turn read as if the agent said nothing between
        # tool calls. Both are emitted now, prose first so the transcript reads in
        # order; `_Streamed` keeps the final answer from arriving twice.
        out: list[dict] = []
        prose = _message_text(data)
        if prose:
            streamed.add(prose)
            out.append({"type": "text", "text": prose})
        usage = data.get("usage")
        if isinstance(usage, dict):
            out.append({"type": "state", "event": "usage", "data": usage})
        return out

    # Everything else keeps the old behaviour: a short progress line, and only
    # when it is one a person watching would want.
    line = _notice_text(notification)
    return [{"type": "notice", "text": line}] if line else []


def _notice_text(notification) -> str | None:
    """A one-line summary worth showing, or None when it is internal chatter.

    The parent renders these live, so a whole payload would be both noise and
    potentially enormous — only the method and the inner event type are used.
    Structured facts do NOT come through here; they come through `_project`,
    which is why this stays deliberately lossy.
    """
    method = str(getattr(notification, "method", "") or "notification")
    detail = ""
    payload = getattr(notification, "payload", None)
    if isinstance(payload, dict):
        event = payload.get("event")
        if isinstance(event, dict):
            detail = str(event.get("type") or "")
        elif payload.get("status"):
            detail = str(payload.get("status"))
    blob = f"{method} {detail}".lower()
    if not any(word in blob for word in _NOTICE_WORTHY):
        return None
    return f"{method} {detail}".strip()[:200]

def _cli_for(home: str) -> Path | None:
    path = Path(home).joinpath(*_CLI_REL) if home else None
    return path if path is not None and path.is_file() else None


class _Live:
    """One SDK runtime, kept alive so the NEXT turn can continue its session.

    Continuity is a property of the PROCESS, not of the session id. Measured
    2026-09-21 against 0.1.6-alpha.2: a second `run()` on the same
    `DeepSeekHarness` continues the session correctly — the follow-up request
    carried the first turn's history — while a second PROCESS handed the same id
    dies with `JsonRpcError: session "..." already exists`. So a runner that is
    spawned per turn can never resume, and this one is not.
    """

    def __init__(self) -> None:
        self.harness = None
        self.key: tuple | None = None
        self.session_id = ""
        self.scratch = ""

    def close(self) -> None:
        if self.harness is not None:
            try:
                self.harness.close()
            except Exception:
                pass
            self.harness = None
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.scratch = ""


def _run_turn(job: dict, live: _Live) -> int:
    """Run one turn on the live runtime and emit the outcome. Never raises.

    Returns 0 when a `done` was emitted. Anything else means this runtime is not
    trustworthy for a second turn, and `main` exits so the parent spawns a fresh
    one rather than writing into a broken session.
    """
    try:
        from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
    except Exception as exc:
        _emit(
            {
                "type": "error",
                "text": (
                    "cannot import deepseek_harness — is the SDK source on "
                    f"PYTHONPATH? ({type(exc).__name__}: {exc})"
                ),
            }
        )
        return 1

    try:
        prompt = str(job.get("prompt") or "")
        system_prompt = str(job.get("system_prompt") or "").strip()
        if system_prompt:
            # DSH owns its system prompt, so a Rigma session's prompt can only
            # reach it as a delimited preamble on the input.
            prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        home = str(job.get("dsh_home") or "")       # the CHECKOUT: CLI + SDK
        cli = _cli_for(home)
        if cli is None:
            _emit({"type": "error", "text": f"no dsh CLI under {home!r}"})
            return 1
        # DSH's own session/config dir, which is NOT the checkout. Its session
        # logs live here, so sharing this with a source tree makes one turn
        # resume another and return empty without ever calling the model.
        data_home = str(job.get("data_home") or "") or home

        max_tokens = int(job.get("max_tokens") or 4096)
        context_window = int(job.get("context_window") or 32768)
        cwd = str(job.get("cwd") or "")
        if not cwd.strip() or not Path(cwd).is_dir():
            # Stable for the LIFE of this runtime, not per turn: DSH keys its
            # session directories off the cwd, so a cwd that moved between turns
            # would put the session somewhere the next turn cannot find.
            if not live.scratch:
                live.scratch = tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            cwd = live.scratch

        # Everything that would make the running runtime WRONG for this job. The
        # patch path is deliberately NOT part of it: it is per-turn by nature,
        # and the patch below is regenerated here instead, once per runtime.
        key = (home, data_home, str(job.get("base_url") or ""),
               str(job.get("model") or ""), cwd, context_window, max_tokens)
        if live.harness is not None and live.key != key:
            live.close()            # a different endpoint or model: rebuild

        if live.harness is None:
            if not live.scratch:
                live.scratch = tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            # The parent's patch may already be gone, or may belong to a turn
            # that is over; this runtime generates the one it will keep using.
            patch = str(job.get("patch_path") or "")
            if not patch or not Path(patch).is_file():
                from rigma.harness_dsh import patch_file

                patch = str(patch_file(context_window, max_tokens, live.scratch))
            # The capabilities the minimal profile does not mount — goals,
            # subagents, todos, skills, plan mode, filesystem tools, compaction.
            # The minimal tree is standalone and deliberately excludes them, so
            # without this patch those capabilities do not exist in the runtime
            # at all and no amount of event forwarding could show them.
            #
            # The capability patch goes FIRST and the per-turn llm patch LAST:
            # patch layers are applied in order and the last write wins per row.
            cap = str(job.get("capability_path") or "")
            # R4-SKILL-1: the skill provider's roots. Generated HERE, into this
            # runtime's own scratch directory, because it carries an absolute path
            # that the shipped capability patch cannot — and because the parent
            # has no scratch directory to write into. Writing it beside the chat's
            # cwd would put a generated file in the user's project.
            skills = ""
            try:
                from rigma.harness_dsh import skills_patch_file

                skills = skills_patch_file(live.scratch)
            except Exception:
                skills = ""     # a missing skill root is a smaller loss than no turn
            # R6-MCP: Rigma's OWN tools, as an MCP server. mcode has received
            # these since `harness_mcode.ensure_mcp`; DSH received nothing, so a
            # DSH turn could not search the user's indexed documents or remember
            # anything, and no line in the capability menu said so. Generated
            # here for the same two reasons as the skills patch: it carries
            # `sys.executable` and the chat's cwd, neither of which the shipped
            # patch can hold. Registered only when the roster is non-empty.
            mcp = ""
            try:
                from rigma.harness_dsh import mcp_patch_file

                mcp = mcp_patch_file(live.scratch, cwd=cwd)
            except Exception:
                mcp = ""        # missing Rigma tools beat no harness turn
            patches = tuple(p for p in (cap, skills, mcp, patch) if p)
            config = DeepSeekHarnessConfig(
                profile="sdk-minimal",
                dsh_bin=str(cli),
                dsh_home=data_home,
                base_url=str(job.get("base_url") or ""),
                api_key="local",
                model=str(job.get("model") or ""),
                max_tokens=max_tokens,
                cwd=cwd,
                runtime_cwd=cwd,
                patches=patches,
            )
            live.harness = DeepSeekHarness(config)
            live.harness.start()
            live.key = key

        # R5-STEPTEXT: per-turn, so one turn's text can never suppress the next
        # turn's. `done` emits only what this has not already sent.
        streamed = _Streamed()

        def on_notification(notification) -> None:
            try:
                for ev in _project(notification, streamed):
                    _emit(ev)
            except Exception:
                pass  # a progress line must never break the turn

        # The session this runtime already owns, which is what makes the second
        # and later turns continue instead of starting the agent from nothing.
        result = live.harness.run(
            prompt,
            session_id=live.session_id or None,
            on_notification=on_notification,
        )
        live.session_id = str(getattr(result, "session_id", "") or "") or live.session_id
        # R5-STEPTEXT: the final answer, MINUS whatever was already streamed as
        # this message's own `assistant/message`. Usually that is all of it, so
        # this emits nothing and `done` carries no duplicate text. When the SDK's
        # final_response says more than the last step's event did, the difference
        # goes out here rather than being lost.
        final = str(getattr(result, "final_response", "") or "")
        tail = streamed.remaining(final)
        _emit(
            {
                "type": "done",
                "text": tail,
                "finish_reason": str(getattr(result, "finish_reason", "") or ""),
                # Reported for the record and for the cross-process case, where
                # it is what the NEXT process would need if the SDK ever learns
                # to open a session it did not create.
                "session_id": live.session_id,
            }
        )
        return 0
    except Exception as exc:
        _emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"[:2000]})
        return 1


def main() -> int:
    # From here on, anything the SDK prints lands on stderr: stdout carries
    # events, and a stray print would be an unparseable line to the parent.
    sys.stdout = sys.stderr
    live = _Live()
    code = 0
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
                if not isinstance(job, dict):
                    raise ValueError("job must be a JSON object")
            except Exception as exc:
                _emit({"type": "error", "text": f"bad job on stdin: {exc}"})
                return 1
            code = _run_turn(job, live)
            if code != 0:
                # This runtime cannot be trusted for another turn. Exit, so the
                # parent spawns a fresh one instead of writing into a session
                # that just failed.
                return code
    finally:
        live.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
