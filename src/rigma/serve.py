from __future__ import annotations

import asyncio
import json
from importlib import resources

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import presets
from . import sessions
from . import state as st

_FALLBACK_HTML = "<!doctype html><html><body><h1>Rigma</h1></body></html>"
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
_NO_STORE = {"Cache-Control": "no-store"}


def _sse(data: dict, event: str = "") -> bytes:
    head = f"event: {event}\n" if event else ""
    return (head + "data: " + json.dumps(data) + "\n\n").encode()


async def _upstream_error(resp) -> str:
    body = await resp.aread()
    try:
        err = json.loads(body)["error"]
        return err["message"] if isinstance(err, dict) else str(err)
    except Exception:
        return (body.decode(errors="replace")[:200]
                or f"upstream HTTP {resp.status_code}")


_UI_FILES = {"app.js": "text/javascript", "md.js": "text/javascript",
             "style.css": "text/css"}


def _ui_file(name: str) -> str:
    try:
        return resources.files("rigma").joinpath(f"data/ui/{name}").read_text(
            encoding="utf-8")
    except Exception:
        return _FALLBACK_HTML if name == "index.html" else ""


def build_app(upstream_port: int, default_prompt: str | None = None,
              registry=None) -> FastAPI:
    app = FastAPI(title="rigma", docs_url=None, redoc_url=None)
    base = f"http://127.0.0.1:{upstream_port}"
    client = httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(600.0))
    ingest_state = {"busy": False, "error": ""}
    ingest_tasks: set = set()

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
        return HTMLResponse(_ui_file("index.html"), headers=_NO_STORE)

    @app.get("/ui/{name}")
    async def ui_asset(name: str):
        if name not in _UI_FILES:
            return JSONResponse({"error": "not found"}, status_code=404,
                                headers=_NO_STORE)
        return Response(_ui_file(name), media_type=_UI_FILES[name],
                        headers=_NO_STORE)

    @app.get("/api/status")
    async def status():
        s = st.server_running()
        if s is None:
            return JSONResponse({"error": "not running"}, status_code=404)
        return {**{k: s[k] for k in ("model", "quant", "public_port", "started_at")},
                "ctx": s.get("ctx", 0),
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
        if "params" in body:
            try:
                body["params"] = sessions.validate_params(body["params"])
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
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

    @app.get("/api/presets")
    async def list_presets():
        return presets.list_presets(registry)

    @app.post("/api/presets")
    async def create_preset(body: dict | None = None):
        body = body or {}
        return presets.create(name=body.get("name", "New preset"),
                              system_prompt=body.get("system_prompt", ""),
                              greeting=body.get("greeting", ""),
                              params=body.get("params"))

    @app.post("/api/presets/{pid}")
    async def update_preset(pid: str, body: dict | None = None):
        if presets.is_builtin(pid):
            return JSONResponse({"error": "builtin preset"}, status_code=403)
        p = presets.load(pid)
        if p is None:
            return JSONResponse({"error": "no such preset"}, status_code=404)
        for k in presets.MUTABLE_FIELDS:
            if k in (body or {}):
                p[k] = body[k]
        presets.save(p)
        return p

    @app.delete("/api/presets/{pid}")
    async def delete_preset(pid: str):
        if presets.is_builtin(pid):
            return JSONResponse({"error": "builtin preset"}, status_code=403)
        if not presets.delete(pid):
            return JSONResponse({"error": "no such preset"}, status_code=404)
        return {"ok": True}

    async def _llm_turn(s: dict):
        preset = presets.resolve(s.get("preset_id", ""), registry) \
            if s.get("preset_id") else None
        msgs = sessions.build_messages(s, _default_prompt(), preset)
        body = {"messages": msgs, "stream": True,
                "stream_options": {"include_usage": True}}
        body.update(sessions.effective_params(s, preset))
        text, failed, resp = "", False, None
        usage, timings = {}, {}
        try:
            req = client.build_request("POST", "/v1/chat/completions", json=body)
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                # llama-server rejects some requests outright (e.g. prompt
                # exceeds ctx) with a JSON error body, not an exception —
                # surface its message or the turn silently vanishes
                raise RuntimeError(await _upstream_error(resp))
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if "error" in obj:
                    err = obj["error"]
                    raise RuntimeError(err.get("message", "upstream error")
                                       if isinstance(err, dict) else str(err))
                usage = obj.get("usage") or usage
                timings = obj.get("timings") or timings
                try:
                    delta = obj["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    text += delta
                    yield _sse({"delta": delta})
        except Exception as e:
            failed = True
            yield _sse({"message": str(e) or "model unreachable"}, event="error")
        finally:
            if resp is not None:
                await resp.aclose()
        if not failed:
            meta = {"ctx": (st.read_state() or {}).get("ctx", 0)}
            if usage.get("prompt_tokens"):
                meta["prompt_tokens"] = usage["prompt_tokens"]
            if timings.get("predicted_per_second"):
                meta["predicted_per_second"] = timings["predicted_per_second"]
            if len(meta) > 1 or meta["ctx"]:
                yield _sse(meta, event="meta")
        if text and not failed:
            s["messages"].append({"role": "assistant", "content": text})
            sessions.save(s)
        yield b"data: [DONE]\n\n"

    @app.get("/api/rag/status")
    async def rag_status():
        from . import rag
        port = rag.recorded_sidecar_port()
        health = rag.sidecar_health(port) if port else None
        return {"running": health is not None, "health": health,
                "sources": rag.load_sources(),
                "indexing": ingest_state["busy"], "error": ingest_state["error"]}

    @app.post("/api/rag/sources")
    async def rag_add_source(body: dict):
        from pathlib import Path
        from . import rag
        path = str(body.get("path", "")).strip()
        if not path or not Path(path).exists():
            return JSONResponse({"error": f"path does not exist: {path}"},
                                status_code=400)
        srcs = rag.add_source(path)
        ingest_state["busy"], ingest_state["error"] = True, ""

        async def _ingest():
            try:
                await asyncio.to_thread(rag.ingest)
            except Exception as e:
                ingest_state["error"] = str(e)
            finally:
                ingest_state["busy"] = False

        task = asyncio.get_running_loop().create_task(_ingest())
        ingest_tasks.add(task)          # asyncio keeps only a weak ref
        task.add_done_callback(ingest_tasks.discard)
        return JSONResponse({"sources": srcs, "indexing": True}, status_code=202)

    async def _rag_turn(s: dict):
        from . import rag
        q = s["messages"][-1]["content"]
        try:
            await asyncio.to_thread(rag.ensure_sidecar)
            a = await asyncio.to_thread(rag.ask, q)
            if not isinstance(a, dict):
                raise RuntimeError(f"unexpected sidecar reply: {type(a).__name__}")
        except Exception as e:
            yield _sse({"message": f"documents unavailable: {e}"}, event="error")
            yield b"data: [DONE]\n\n"
            return
        text = a.get("answer", "")
        if a.get("abstained"):
            text = "(abstained — not enough evidence in your documents)\n" + text
        if not text:
            yield _sse({"message": "documents returned an empty answer"},
                       event="error")
            yield b"data: [DONE]\n\n"
            return
        yield _sse({"delta": text})
        cites = a.get("citations") or []
        if cites:
            yield _sse({"citations": cites}, event="citations")
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
        gen = _rag_turn(s) if s.get("use_rag") else _llm_turn(s)
        return StreamingResponse(gen, media_type="text/event-stream",
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
