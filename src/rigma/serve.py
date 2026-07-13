from __future__ import annotations

from importlib import resources

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import sessions
from . import state as st

_FALLBACK_HTML = "<!doctype html><html><body><h1>Rigma</h1></body></html>"
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


def _chat_html() -> str:
    try:
        return resources.files("rigma").joinpath("data/ui/chat.html").read_text(
            encoding="utf-8")
    except Exception:
        return _FALLBACK_HTML


def build_app(upstream_port: int, default_prompt: str | None = None) -> FastAPI:
    app = FastAPI(title="rigma", docs_url=None, redoc_url=None)
    base = f"http://127.0.0.1:{upstream_port}"
    client = httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(600.0))

    def _default_prompt() -> str:
        if default_prompt is not None:
            return default_prompt
        try:
            return sessions.default_prompt()
        except Exception:
            return ""

    @app.get("/", response_class=HTMLResponse)
    async def root():
        # no-store: llama-server's own webui once bound this port on a user
        # machine and the browser kept serving its cached SPA long after —
        # never let any UI (ours included) outlive its server via cache.
        return HTMLResponse(_chat_html(),
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def status():
        s = st.server_running()
        if s is None:
            return JSONResponse({"error": "not running"}, status_code=404)
        return {**{k: s[k] for k in ("model", "quant", "public_port", "started_at")},
                "default_system_prompt": _default_prompt()}

    @app.get("/api/sessions")
    async def list_sessions():
        return sessions.list_sessions()

    @app.post("/api/sessions")
    async def create_session(body: dict | None = None):
        body = body or {}
        return sessions.create(title=body.get("title", "New chat"),
                               system_prompt=body.get("system_prompt", ""))

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return s

    @app.post("/api/sessions/{sid}")
    async def update_session(sid: str, body: dict | None = None):
        body = body or {}
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        for k in sessions.MUTABLE_FIELDS:
            if k in body:
                s[k] = body[k]
        sessions.save(s)
        return s

    @app.delete("/api/sessions/{sid}")
    async def delete_session(sid: str):
        if not sessions.delete(sid):
            return JSONResponse({"error": "no such session"}, status_code=404)
        return {"ok": True}

    @app.api_route("/v1/{path:path}",
                   methods=["GET", "POST", "OPTIONS", "DELETE"])
    async def proxy(request: Request, path: str):
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS}
        upstream = client.build_request(
            request.method, f"/v1/{path}", headers=headers,
            content=await request.body())
        resp = await client.send(upstream, stream=True)
        media = resp.headers.get("content-type", "application/json")
        if "text/event-stream" in media:
            async def gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
            return StreamingResponse(gen(), status_code=resp.status_code,
                                     media_type=media)
        body = await resp.aread()
        await resp.aclose()
        return Response(content=body, status_code=resp.status_code,
                        media_type=media)

    return app


def run_ui(public_port: int, upstream_port: int) -> None:
    import uvicorn
    uvicorn.run(build_app(upstream_port), host="127.0.0.1", port=public_port,
                log_level="warning")
