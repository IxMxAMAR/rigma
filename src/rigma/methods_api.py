"""HTTP surface for Methods: catalog CRUD, macro preview, macro run.

This lives outside serve.py on purpose. serve.py is 3.3K lines of closures
inside build_app and a router split was already deferred once
(docs/backlog-post-audit.md); a whole feature's endpoints piled on top is
exactly what that note was warning about. Everything serve.py owns -- the SSE
encoder, a real chat turn, the aux slot, the tool context -- arrives by
injection instead.
"""
from __future__ import annotations

import asyncio

from fastapi.responses import JSONResponse, StreamingResponse

from . import macros, method_drafts, methods as _methods, sessions

_NO_STORE = {"cache-control": "no-store"}

# SHORT and action-first, and that is a measured constraint rather than a
# style preference. Live A/B on the owner's 35B (2026-07-21): no system
# message -> 3K chars of thinking and a clean tool call; a 9-rule doctrine ->
# 15.7K chars of spiralling and NO reply at all; a 5-rule rewrite -> 870
# chars, a reply and the call. Every meta-rule is something to reason ABOUT
# before acting, and either/or framings are the worst offenders.
# tests/test_creation_chat.py pins the length so this cannot creep back.
BUILDER_PROMPT = (
    "You are building a Method with the user. "
    "Ask ONE question, then call a builder tool. "
    "Never explain the schema, just build it. "
    "Keep any prompt you write short and imperative. "
    "When they are happy, call save_method.")

GREETING = ("Hello — let's build your method. What kind of work is it for?")


def register(app, *, sse, drive_turn, aux_complete, tool_ctx_for) -> None:
    """Mount the method routes on `app`.

    sse(data: dict, event: str = "") -> bytes
    drive_turn(session) -> awaitable[str]      one real agentic turn
    aux_complete(prompt, max_tokens) -> awaitable[str]
    tool_ctx_for(session) -> dict              the ctx tools expect
    """

    def _resolve(sid: str, macro_id: str):
        """(session, method, macro) or a JSONResponse to return instead."""
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        m = _methods.get(str(s.get("method") or ""))
        if m is None:
            return JSONResponse({"error": "this chat has no method"},
                                status_code=404)
        macro = next((x for x in m.get("macros") or []
                      if x.get("id") == macro_id), None)
        if macro is None:
            return JSONResponse(
                {"error": f"no macro '{macro_id}' in method '{m['id']}'"},
                status_code=404)
        return s, m, macro

    @app.get("/api/methods")
    async def list_methods():
        """Workflow methods: built-ins merged with the user's own."""
        return {"methods": _methods.catalog()}

    @app.get("/api/methods/{mid}")
    async def get_method(mid: str):
        m = _methods.get(mid)
        if m is None:
            return JSONResponse({"error": "no such method"}, status_code=404)
        return m

    @app.post("/api/methods")
    async def save_method(body: dict):
        saved, errs = _methods.save_user(body or {})
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        return saved

    @app.delete("/api/methods/{mid}")
    async def delete_method(mid: str):
        if not _methods.delete_user(mid):
            return JSONResponse(
                {"error": "built-in methods cannot be deleted -- save a user "
                          "method with the same id to override it instead"},
                status_code=400)
        return {"deleted": mid}

    @app.post("/api/methods/draft")
    async def create_draft(body: dict | None = None):
        """Open a chat that can only build a method."""
        d = method_drafts.new_draft()
        s = sessions.create("New method")
        s["method_draft_id"] = d["id"]
        s["system_prompt"] = BUILDER_PROMPT
        s["use_tools"] = True
        # allow_code is irrelevant here -- builder_only withholds every
        # non-builder tool anyway -- but leave it off so nothing about this
        # session reads as permission to touch the disk.
        s["allow_code"] = False
        # one_action turns on serve.py's existing force_call path, so this
        # chat inherits tool_choice:"required" AND its per-model 400 fallback
        s["one_action"] = True
        s["effort"] = "off"
        # Pre-seeded, not generated: instant, deterministic, un-rambleable,
        # and it still reads as the model's voice in the transcript.
        s["messages"] = [{"role": "assistant", "content": GREETING}]
        s["title_source"] = "user"
        sessions.save(s)
        return {"session_id": s["id"], "draft_id": d["id"]}

    @app.get("/api/methods/draft/{did}")
    async def get_draft(did: str):
        d = method_drafts.load(did)
        if d is None:
            return JSONResponse({"error": "no such draft"}, status_code=404)
        return d

    @app.post("/api/methods/draft/{did}/promote")
    async def promote_draft(did: str):
        saved, errs = method_drafts.promote(did)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        return saved

    @app.post("/api/sessions/{sid}/method")
    async def apply_method(sid: str, body: dict):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        out = _methods.apply_to_session(s, str((body or {}).get("id", "")))
        if out is None:
            return JSONResponse({"error": "no such method"}, status_code=404)
        sessions.save(out)
        return out

    @app.post("/api/sessions/{sid}/macro/preview")
    async def preview_macro(sid: str, body: dict):
        got = _resolve(sid, str((body or {}).get("macro_id") or ""))
        if isinstance(got, JSONResponse):
            return got
        s, m, macro = got
        steps = macro.get("steps") or []
        ctx = macros.build_context(s, m, answers=body.get("answers"),
                                   selection=str(body.get("selection") or ""))
        # allow_code decides whether a plain `prompt` step can reach a write
        # tool at all -- see macros.is_effectful
        code = bool(s.get("allow_code"))
        effectful = macros.is_effectful(steps, allow_code=code)
        trusted = macros.is_trusted(m["id"], macro["id"])
        return {"macro_id": macro["id"], "label": macro.get("label", ""),
                "effectful": effectful, "trusted": trusted,
                "needs_confirm": effectful and not trusted,
                "preview": macros.preview_line(macro, ctx, allow_code=code),
                "asks": macros.asks(steps)}

    @app.post("/api/sessions/{sid}/macro")
    async def run_macro_ep(sid: str, body: dict):
        body = body or {}
        got = _resolve(sid, str(body.get("macro_id") or ""))
        if isinstance(got, JSONResponse):
            return got
        s, m, macro = got
        steps = macro.get("steps") or []
        confirm = str(body.get("confirm") or "")
        code = bool(s.get("allow_code"))
        if (macros.is_effectful(steps, allow_code=code)
                and not macros.is_trusted(m["id"], macro["id"])
                and confirm not in ("run", "always")):
            return JSONResponse(
                {"error": "this macro changes things -- POST again with "
                          "confirm: 'run' or 'always'",
                 "preview": macros.preview_line(
                     macro, macros.build_context(s, m), allow_code=code)},
                status_code=403)
        if confirm == "always":
            macros.trust(m["id"], macro["id"])

        async def gen():
            # Live, not buffered: a step's event reaches the client the moment
            # it is emitted. run_macro drives on its own task and pushes into
            # a queue; None is the end-of-stream sentinel.
            q: asyncio.Queue = asyncio.Queue()

            async def emit(event, data):
                await q.put(sse(data, event))

            async def drive():
                try:
                    out = await macros.run_macro(
                        s, m, macro, emit=emit, drive_turn=drive_turn,
                        aux_complete=aux_complete, tool_ctx=tool_ctx_for(s),
                        answers=body.get("answers"),
                        selection=str(body.get("selection") or ""))
                    await q.put(sse({"new_session_id": out["new_session_id"],
                                     "steps": len(out["results"])},
                                    "macro_done"))
                except asyncio.CancelledError:
                    raise
                except Exception as e:                  # never a bare 500
                    await q.put(sse({"message": f"macro failed: {e}"},
                                    "error"))
                finally:
                    await q.put(None)

            task = asyncio.create_task(drive())
            try:
                while True:
                    chunk = await q.get()
                    if chunk is None:
                        break
                    yield chunk
                yield b"data: [DONE]\n\n"
            finally:
                # the client hung up mid-macro: stop the work, don't leak it
                if not task.done():
                    task.cancel()

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=_NO_STORE)
