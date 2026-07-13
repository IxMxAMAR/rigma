from __future__ import annotations

import json
from importlib import resources

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import sessions
from . import state as st

_FALLBACK_HTML = "<!doctype html><html><body><h1>Rigma</h1></body></html>"
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
_NO_STORE = {"Cache-Control": "no-store"}


def _sse(data: dict, event: str = "") -> bytes:
    head = f"event: {event}\n" if event else ""
    return (head + "data: " + json.dumps(data) + "\n\n").encode()


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
        return HTMLResponse(_chat_html(), headers=_NO_STORE)

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

    async def _llm_turn(s: dict):
        msgs = sessions.build_messages(s, _default_prompt())
        text = ""
        try:
            req = client.build_request("POST", "/v1/chat/completions",
                                       json={"messages": msgs, "stream": True})
            resp = await client.send(req, stream=True)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = (json.loads(payload)["choices"][0]["delta"]
                             .get("content"))
                except Exception:
                    delta = None
                if delta:
                    text += delta
                    yield _sse({"delta": delta})
            await resp.aclose()
        except Exception as e:
            yield _sse({"message": f"model unreachable: {e}"}, event="error")
        if text:
            s["messages"].append({"role": "assistant", "content": text})
            sessions.save(s)
        yield b"data: [DONE]\n\n"

    @app.post("/api/sessions/{sid}/chat")
    async def chat_turn(sid: str, body: dict):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        message = body.get("message")
        if message:
            s["messages"].append({"role": "user", "content": message})
            if s.get("title") == "New chat":
                s["title"] = message[:40]
            sessions.save(s)
        if not s["messages"]:
            return JSONResponse({"error": "session has no messages"},
                                status_code=400)
        return StreamingResponse(_llm_turn(s), media_type="text/event-stream",
                                 headers=_NO_STORE)

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
