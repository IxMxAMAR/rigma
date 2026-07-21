"""HTTP surface for Methods: catalog CRUD, macro preview, macro run.

This lives outside serve.py on purpose. serve.py is 3.3K lines of closures
inside build_app and a router split was already deferred once
(docs/backlog-post-audit.md); a whole feature's endpoints piled on top is
exactly what that note was warning about. Everything serve.py owns -- the SSE
encoder, a real chat turn, the aux slot, the tool context -- arrives by
injection instead.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse, StreamingResponse

from . import macros, methods as _methods, sessions

_NO_STORE = {"cache-control": "no-store"}


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
        effectful = macros.is_effectful(steps)
        trusted = macros.is_trusted(m["id"], macro["id"])
        return {"macro_id": macro["id"], "label": macro.get("label", ""),
                "effectful": effectful, "trusted": trusted,
                "needs_confirm": effectful and not trusted,
                "preview": macros.preview_line(macro, ctx),
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
        if (macros.is_effectful(steps)
                and not macros.is_trusted(m["id"], macro["id"])
                and confirm not in ("run", "always")):
            return JSONResponse(
                {"error": "this macro changes things -- POST again with "
                          "confirm: 'run' or 'always'",
                 "preview": macros.preview_line(
                     macro, macros.build_context(s, m))},
                status_code=403)
        if confirm == "always":
            macros.trust(m["id"], macro["id"])

        async def gen():
            # Plan 1 buffers: every step runs to completion before a byte is
            # yielded, so progress arrives in one burst. Deliberate -- it
            # keeps the endpoint synchronous and testable without an engine,
            # and macros are short. Plan 2 swaps in a live asyncio.Queue once
            # there is a UI that can show progress.
            queue: list[bytes] = []

            async def emit(event, data):
                queue.append(sse(data, event))

            try:
                out = await macros.run_macro(
                    s, m, macro, emit=emit, drive_turn=drive_turn,
                    aux_complete=aux_complete, tool_ctx=tool_ctx_for(s),
                    answers=body.get("answers"),
                    selection=str(body.get("selection") or ""))
                for chunk in queue:
                    yield chunk
                yield sse({"new_session_id": out["new_session_id"],
                           "steps": len(out["results"])}, "macro_done")
            except Exception as e:                      # never a bare 500
                for chunk in queue:
                    yield chunk
                yield sse({"message": f"macro failed: {e}"}, "error")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=_NO_STORE)
