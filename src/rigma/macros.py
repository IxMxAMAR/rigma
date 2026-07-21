"""Running a Method's step list: substitution, safety, trust, interpreter.

Rules, Macros and Workflows are one primitive -- a step list -- so this is
the only place that knows how to execute one. Everything the model touches
(a real chat turn, an aux completion) is INJECTED rather than imported, which
is what lets the whole interpreter be tested without an engine running.

Spec: docs/superpowers/specs/2026-07-21-custom-methods-design.md
"""
from __future__ import annotations

import asyncio
import copy
import json
import re

from . import method_schema as ms
from .runtime import rigma_home

_TRANSCRIPT_PER_MSG = 2000
_TRANSCRIPT_TOTAL = 8000
_TRANSCRIPT_MSGS = 12


# --- context and substitution --------------------------------------------

def build_context(session: dict, method: dict, *, answers: dict | None = None,
                  selection: str = "", results: list[str] | None = None
                  ) -> dict:
    """Everything a placeholder can resolve to, computed fresh. Callers
    rebuild this between steps so {{last_reply}} and {{step:N}} see the
    world as it is NOW, not as it was when the macro was clicked."""
    msgs = session.get("messages") or []
    last_reply = next(
        (str(m.get("content") or "") for m in reversed(msgs)
         if m.get("role") == "assistant" and isinstance(m.get("content"), str)),
        "")
    # assistant prose only: the user's own prompt text is never material for
    # a summary step, and keeping it out means a macro cannot smuggle the
    # user's wording into an aux call.
    prose = "\n".join(
        str(m.get("content", ""))[:_TRANSCRIPT_PER_MSG]
        for m in msgs[-_TRANSCRIPT_MSGS:]
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    )[-_TRANSCRIPT_TOTAL:]
    title = str(session.get("title") or "")
    mt = re.search(r"\d+", title)
    title_next = (title[:mt.start()] + str(int(mt.group()) + 1)
                  + title[mt.end():]) if mt else title
    return {
        "vars": {k: str((v or {}).get("default", ""))
                 for k, v in (method.get("vars") or {}).items()},
        "answers": dict(answers or {}),
        "results": list(results or []),
        "last_reply": last_reply,
        "transcript": prose,
        "title_next": title_next,
        "selection": selection,
    }


def _resolve(name: str, arg: str, ctx: dict) -> str | None:
    if name == "step":
        if not arg.isdigit():
            return None
        i = int(arg)
        # a not-yet-run step resolves EMPTY, never to the literal braces --
        # the model would otherwise read "{{step:2}}" as content to imitate
        return ctx["results"][i] if i < len(ctx["results"]) else ""
    if name == "ask":
        return ctx["answers"].get(arg, "")
    if name in ("last_reply", "transcript", "title_next", "selection"):
        return ctx[name]
    return ctx["vars"].get(name)


def substitute(obj, ctx: dict):
    """Deep copy of `obj` with every placeholder resolved. An unknown name is
    left verbatim so validate() can be the one that complains about it."""
    if isinstance(obj, str):
        def sub(m):
            got = _resolve(m.group(1), m.group(2) or "", ctx)
            return m.group(0) if got is None else got
        return ms.PLACEHOLDER.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: substitute(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, ctx) for v in obj]
    return copy.deepcopy(obj)


def asks(steps: list[dict]) -> list[str]:
    """Distinct {{ask:...}} labels, in first-seen order -- the questions the
    client must answer before the macro can run."""
    out: list[str] = []
    for step in steps or []:
        for name, arg in ms.find_placeholders(step):
            if name == "ask" and arg not in out:
                out.append(arg)
    return out


# --- safety ---------------------------------------------------------------

def is_effectful(steps: list[dict]) -> bool:
    """True if running this unattended could change something the user cares
    about. Conservative by construction: anything not on the read-only
    allowlist counts, including every MCP tool."""
    for step in steps or []:
        kind = step.get("kind")
        if kind == "new_chat":
            return True
        if kind == "tool" and step.get("name") not in ms.SAFE_TOOLS:
            return True
    return False


def preview_line(macro: dict, ctx: dict) -> str:
    """The confirm-bar sentence, generated FROM THE STEPS after substitution
    so it names the real files rather than a template."""
    bits: list[str] = []
    for step in macro.get("steps") or []:
        s = substitute(step, ctx)
        kind = s.get("kind")
        if kind == "tool" and s.get("name") not in ms.SAFE_TOOLS:
            target = (s.get("args") or {}).get("path") or ""
            bits.append(f"{s['name']}" + (f" on {target}" if target else ""))
        elif kind == "new_chat":
            bits.append("open a new chat")
    what = ", ".join(dict.fromkeys(bits)) or "read-only steps"
    return f"{macro.get('label') or macro.get('id')}: will run {what}"


# --- trust ----------------------------------------------------------------

def trust_path():
    return rigma_home() / "macro_trust.json"


def _trust_all() -> dict:
    try:
        return json.loads(trust_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_trusted(method_id: str, macro_id: str) -> bool:
    return bool(_trust_all().get(f"{method_id}:{macro_id}"))


def trust(method_id: str, macro_id: str) -> None:
    """Remember 'Always allow'. This lives in ~/.rigma, NOT in the method
    file: a method file is the shareable export artifact, and importing
    someone else's method must never import their trust decisions."""
    all_ = _trust_all()
    all_[f"{method_id}:{macro_id}"] = True
    p = trust_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(all_, indent=2), encoding="utf-8")


# --- the interpreter ------------------------------------------------------

async def run_macro(session: dict, method: dict, macro: dict, *,
                    emit, drive_turn, aux_complete, tool_ctx: dict,
                    answers: dict | None = None,
                    selection: str = "") -> dict:
    """Execute one step list. Model-facing work is injected (`drive_turn`,
    `aux_complete`) so this is testable without an engine, and so the caller
    keeps ownership of the SSE plumbing and the idle watchdog.

    A step that returns an engine/tool error STOPS the macro: a scripted
    sequence that carries on past a failed read is how you get a write step
    that appends the word 'error' to someone's story bible.
    """
    from . import sessions, tools as toolkit
    results: list[str] = []
    new_session_id: str | None = None
    steps = macro.get("steps") or []
    dirty = False

    for i, raw in enumerate(steps):
        ctx = build_context(session, method, answers=answers,
                            selection=selection, results=results)
        step = substitute(raw, ctx)
        kind = step.get("kind")
        await emit("macro_step", {"index": i, "kind": kind,
                                  "label": macro.get("label", ""),
                                  "total": len(steps)})

        if kind == "tool":
            name = str(step.get("name") or "")
            args = step.get("args") or {}
            call_id = f"{macro.get('id', 'm')}{i}"
            await emit("tool", {"id": call_id, "name": name, "args": args})
            # cached_run -> run_tool: the ONE choke point where every result
            # passes _defuse_control_bytes. Never call a handler directly.
            result = await asyncio.to_thread(
                toolkit.cached_run, name, args, tool_ctx)
            await emit("tool_result", {"id": call_id, "name": name,
                                       "result": result[:900]})
            results.append(result)
            if result.lstrip().lower().startswith("error"):
                await emit("error", {"message": f"step {i} ({name}) failed: "
                                                f"{result[:200]}"})
                break

        elif kind == "prompt":
            text = str(step.get("text") or "")
            if step.get("to") == "aux":
                # fresh context on the aux slot: no transcript pollution and
                # it cannot evict the conversation's prompt cache
                results.append(await aux_complete(text, 400))
            else:
                session.setdefault("messages", []).append(
                    {"role": "user", "content": text})
                dirty = True
                results.append(await drive_turn(session))

        elif kind == "settings":
            st = step.get("set") or {}
            if "effort" in st and st["effort"] in ms.EFFORTS:
                session["effort"] = st["effort"]
            if isinstance(st.get("params"), dict):
                session["params"] = {**(session.get("params") or {}),
                                     **sessions.validate_params(st["params"])}
            for f in ("use_tools", "allow_code"):
                if f in st:
                    session[f] = bool(st[f])
            results.append("")
            dirty = True

        elif kind == "note":
            text = str(step.get("text") or "")
            if step.get("op") == "replace":
                session["notes"] = text
            else:
                cur = str(session.get("notes") or "")
                session["notes"] = (cur + ("\n" if cur and not
                                           cur.endswith("\n") else "") + text)
            results.append("")
            dirty = True

        elif kind == "new_chat":
            if dirty:
                sessions.save(session)
                dirty = False
            nxt = sessions.create(str(step.get("title") or "New chat"))
            for f in step.get("carry") or []:
                if f in ms.CARRY_FIELDS:
                    nxt[f] = copy.deepcopy(session.get(f))
            if nxt.get("method"):
                # re-apply so prompt/params/effort match the method, then put
                # the carried notes back -- apply only fills EMPTY notes, so
                # ordering here is what keeps a carried bible intact
                carried_notes = nxt.get("notes")
                from . import methods as _methods
                _methods.apply_to_session(nxt, nxt["method"])
                if carried_notes:
                    nxt["notes"] = carried_notes
            # keep the macro's title through auto-titling
            nxt["title_source"] = "auto" if step.get("title") else ""
            sessions.save(nxt)
            new_session_id = nxt["id"]
            results.append(nxt["id"])

    if dirty:
        sessions.save(session)
    return {"results": results, "new_session_id": new_session_id}
