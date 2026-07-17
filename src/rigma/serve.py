from __future__ import annotations

import asyncio
import json
import os
import threading
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


def _with_prefill(prefill: str, text: str) -> str:
    """llama-server continues from a trailing assistant message AND echoes that
    prefix in its output, so `text` normally already begins with the prefill.
    Prepend it only for a (rare) engine that streams just the continuation —
    so the steered opening is never doubled (the echo case) nor lost."""
    if not prefill:
        return text
    if text.lstrip().startswith(prefill.strip()[:24]):
        return text
    return prefill + text


_COMPACT_PROMPT = (
    "Summarize this conversation transcript into a dense digest for the model "
    "to continue from. Preserve: named characters/entities and their traits, "
    "established facts and decisions, tone/style commitments, and open threads. "
    "Write plain prose, no preamble, under 300 words.")


async def _upstream_error(resp) -> str:
    body = await resp.aread()
    try:
        err = json.loads(body)["error"]
        return err["message"] if isinstance(err, dict) else str(err)
    except Exception:
        return (body.decode(errors="replace")[:200]
                or f"upstream HTTP {resp.status_code}")


_UI_FILES = {"app.js": "text/javascript", "md.js": "text/javascript",
             "store.js": "text/javascript", "panels.js": "text/javascript",
             "features.js": "text/javascript", "style.css": "text/css",
             "manifest.webmanifest": "application/manifest+json"}


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
    telemetry = {"tg": None}   # last observed decode speed, for the verdict
    switch_lock = threading.Lock()
    activity = {"last": 0.0}   # last inference request, for idle auto-unload
    pull_samples = {}          # file -> (time, bytes) for download-rate calc

    def _now() -> float:
        import time
        return time.time()

    def _bump_stats(timings: dict) -> None:
        """Lifetime odometer: total tokens + turns, persisted to ~/.rigma."""
        n = timings.get("predicted_n") or 0
        if not n:
            return
        try:
            f = st.rigma_home() / "stats.json"
            cur = {}
            if f.exists():
                cur = json.loads(f.read_text(encoding="utf-8"))
            cur["total_tokens"] = int(cur.get("total_tokens", 0)) + int(n)
            cur["total_turns"] = int(cur.get("total_turns", 0)) + 1
            model = (st.read_state() or {}).get("model", "")
            by = cur.setdefault("by_model", {})
            by[model] = int(by.get(model, 0)) + int(n)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(cur), encoding="utf-8")
            tmp.replace(f)
        except Exception:
            pass   # stats are best-effort; never break a turn

    async def _ensure_loaded() -> None:
        """Auto-reload the engine if it was idle-unloaded (Ollama parity).

        Waits out an in-progress unload/switch instead of racing a request
        straight into a dying engine — the lock is only ever held for the
        seconds of a kill/launch."""
        from . import server_ops
        for _ in range(60):        # up to ~30s: covers a graceful kill + relaunch
            s = st.read_state()
            if s is None:
                return
            if not s.get("unloaded"):
                # a concurrent op may still hold the lock mid-relaunch; wait
                if switch_lock.acquire(blocking=False):
                    switch_lock.release()
                    return
                await asyncio.sleep(0.5)
                continue
            if switch_lock.acquire(blocking=False):
                try:
                    await asyncio.to_thread(server_ops.perform_load, registry)
                except Exception:
                    pass   # reload failed; the turn will surface its own error
                finally:
                    switch_lock.release()
                return
            await asyncio.sleep(0.5)   # someone else is switching; let them

    def _default_prompt() -> str:
        if default_prompt is not None:
            return default_prompt
        try:
            return sessions.default_prompt()
        except Exception:
            return ""

    def _model_defaults() -> dict:
        """Model-card sampling for the running model (weakest param layer)."""
        try:
            from .registry import Registry
            reg = registry if registry is not None else Registry.load()
            return reg.models[(st.read_state() or {}).get("model", "")
                              ].default_params
        except Exception:
            return {}

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
        caps: list = []
        try:
            from .registry import Registry
            reg = registry if registry is not None else Registry.load()
            caps = reg.models[s["model"]].capabilities
        except Exception:
            pass
        return {**{k: s[k] for k in ("model", "quant", "public_port", "started_at")},
                "ctx": s.get("ctx", 0), "capabilities": caps,
                "unloaded": bool(s.get("unloaded")),
                "calibrating": server_ops.read_calib_marker(),
                "default_system_prompt": _default_prompt()}

    @app.get("/api/sessions")
    async def list_sessions():
        # file scans block the loop with many/large chats — thread them
        return await asyncio.to_thread(sessions.list_sessions)

    @app.post("/api/sessions")
    async def create_session(body: dict | None = None):
        body = body or {}
        return sessions.create(title=body.get("title", "New chat"),
                               system_prompt=body.get("system_prompt", ""))

    @app.get("/api/sessions/search")
    async def search_sessions(q: str = ""):
        return await asyncio.to_thread(sessions.search, q)

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return s

    @app.get("/api/sessions/{sid}/export")
    async def export_session(sid: str, fmt: str = "md"):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        from urllib.parse import quote
        raw = s.get("title") or "chat"
        stem = "".join(ch for ch in raw if ch.isascii()
                       and (ch.isalnum() or ch in " -_")).strip() or "chat"
        ext = ".md" if fmt == "md" else ".json"
        disp = (f'attachment; filename="{stem}{ext}"; '
                f"filename*=UTF-8''{quote(raw)}{ext}")
        if fmt == "md":
            return Response(sessions.export_markdown(s), media_type="text/markdown",
                            headers={"Content-Disposition": disp})
        if fmt == "json":
            return JSONResponse(s, headers={"Content-Disposition": disp})
        return JSONResponse({"error": f"unknown format: {fmt}"}, status_code=400)

    @app.post("/api/sessions/{sid}/duplicate")
    async def duplicate_session(sid: str):
        d = sessions.duplicate(sid)
        if d is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return d

    @app.post("/api/sessions/{sid}")
    async def update_session(sid: str, body: dict | None = None):
        body = body or {}
        if "params" in body:
            try:
                body["params"] = sessions.validate_params(body["params"])
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        if "effort" in body and body["effort"] not in sessions.EFFORT_LEVELS:
            return JSONResponse(
                {"error": "effort: must be one of off/auto/on (or blank)"},
                status_code=400)
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        for k in sessions.MUTABLE_FIELDS:
            if k in body:
                s[k] = body[k]
        sessions.save(s)
        return s

    @app.post("/api/sessions/{sid}/compact")
    async def compact_session(sid: str, body: dict | None = None):
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        try:
            keep = max(0, int((body or {}).get("keep", 6)))
        except (TypeError, ValueError):
            return JSONResponse({"error": "keep: must be an integer"},
                                status_code=400)
        msgs = s.get("messages", [])
        old = msgs[:-keep] if keep else list(msgs)
        recent = msgs[-keep:] if keep else []
        if not old:
            return JSONResponse({"error": "nothing to compact"}, status_code=400)
        parts = []
        if s.get("digest"):
            parts.append("Previous summary:\n" + s["digest"])
        for m in old:
            content = m.get("content", "")
            if not isinstance(content, str):   # vision parts: keep the text
                content = " ".join(
                    p.get("text", "") if p.get("type") == "text" else "[image]"
                    for p in content if isinstance(p, dict))
            parts.append(f"{m.get('role', 'user')}: {content}")
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content":
                                    _COMPACT_PROMPT + "\n\n" + "\n".join(parts)}],
                      "stream": False, "temperature": 0.3},
                timeout=120.0)
            if resp.status_code != 200:
                raise RuntimeError(await _upstream_error(resp))
            digest = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            if not digest:
                raise RuntimeError("summarizer returned an empty digest")
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        s["digest"] = digest
        s["archive"] = s.get("archive", []) + old   # nothing is ever destroyed
        s["messages"] = recent
        sessions.save(s)
        return {"session": s, "archived": len(old)}

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

    async def _llm_turn(s: dict, cont: bool = False):
        preset = presets.resolve(s.get("preset_id", ""), registry) \
            if s.get("preset_id") else None
        msgs = sessions.build_messages(s, _default_prompt(), preset)
        # steer the reply's opening: llama-server continues from a trailing
        # assistant message AND echoes that prefix back in its output, so we
        # must NOT also add it ourselves. Prefill doesn't combine with tools.
        use_tools = bool(s.get("use_tools")) and not cont
        prefill = "" if (cont or use_tools) else (s.get("prefill") or "")
        if not prefill.strip():
            prefill = ""
        if prefill:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        params = sessions.effective_params(s, preset, _model_defaults())
        effort = s.get("effort", "")
        specs, tctx, trace = None, None, []
        if use_tools:
            from pathlib import Path as _Path

            from . import rag
            from . import tools as toolkit
            tctx = {"allow_code": bool(s.get("allow_code")),
                    "workspace": s.get("workspace") or str(_Path.home())}
            specs = toolkit.tool_specs(
                allow_code=tctx["allow_code"],
                has_rag=bool(rag.recorded_sidecar_port()),
                workspace=tctx["workspace"])
        text, thinking, failed, resp = "", "", False, None
        usage, timings = {}, {}
        # agentic loop: stream a round; if the model called tools, run them,
        # feed results back and stream again; otherwise this round is the answer
        for _round in range(6 if use_tools else 1):
            body = {"messages": msgs, "stream": True,
                    "stream_options": {"include_usage": True}}
            body.update(params)
            if specs:
                body["tools"] = specs
            if effort == "off":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            elif effort == "on":
                body["chat_template_kwargs"] = {"enable_thinking": True}
            rtext, calls = "", {}
            try:
                req = client.build_request("POST", "/v1/chat/completions",
                                           json=body)
                resp = await client.send(req, stream=True)
                if resp.status_code != 200:
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
                        raise RuntimeError(
                            err.get("message", "upstream error")
                            if isinstance(err, dict) else str(err))
                    usage = obj.get("usage") or usage
                    timings = obj.get("timings") or timings
                    try:
                        d = obj["choices"][0]["delta"]
                    except Exception:
                        d = {}
                    rdelta = d.get("reasoning_content")
                    if rdelta:
                        thinking += rdelta
                        yield _sse({"delta": rdelta}, event="think")
                    for tc in d.get("tool_calls") or []:   # accumulate by index
                        slot = calls.setdefault(tc.get("index", 0),
                                                {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                    delta = d.get("content")
                    if delta:
                        rtext += delta
                        yield _sse({"delta": delta})
            except Exception as e:
                failed = True
                msg = str(e) or "model unreachable"
                if isinstance(e, httpx.ConnectError):
                    msg = ("the engine is unloaded — load it again from "
                           "⚙ → Server (or run: rigma load)"
                           if (st.read_state() or {}).get("unloaded")
                           else "engine unreachable — check ⚙ → Server → log")
                yield _sse({"message": msg}, event="error")
            finally:
                if resp is not None:
                    await resp.aclose()
                    resp = None
            if failed:
                break
            if use_tools and calls:            # the model asked to run tools
                from . import tools as toolkit
                msgs.append({"role": "assistant", "content": rtext,
                             "tool_calls": [
                                 {"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": c["args"]}}
                                 for c in calls.values()]})
                for c in calls.values():
                    try:
                        cargs = json.loads(c["args"] or "{}")
                    except Exception:
                        cargs = {}
                    yield _sse({"name": c["name"], "args": cargs}, event="tool")
                    result = await asyncio.to_thread(toolkit.run_tool,
                                                     c["name"], cargs, tctx)
                    yield _sse({"name": c["name"], "result": result[:400]},
                               event="tool_result")
                    trace.append({"name": c["name"], "args": cargs,
                                  "result": result})
                    msgs.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": result})
                continue                       # stream the next round
            text = rtext                       # no tool calls -> this is final
            break
        if not failed:
            meta = {"ctx": (st.read_state() or {}).get("ctx", 0)}
            if usage.get("prompt_tokens"):
                meta["prompt_tokens"] = usage["prompt_tokens"]
            if timings.get("predicted_per_second"):
                meta["predicted_per_second"] = timings["predicted_per_second"]
                telemetry["tg"] = timings["predicted_per_second"]
            if len(meta) > 1 or meta["ctx"]:
                yield _sse(meta, event="meta")
        # persist if we have an answer OR tools already ran — a mid-loop
        # failure must not erase the record of files written / commands run
        if (text or trace) and not (cont and failed):
            # reload before saving: a minutes-long generation must not
            # clobber title/param/notes edits that landed meanwhile (TOCTOU).
            # Messages stay OUR snapshot + this turn — the turn owns them.
            fresh = sessions.load(s["id"])
            if fresh is None:
                # the session was deleted mid-generation — discard the turn,
                # never resurrect the file the user just removed
                yield b"data: [DONE]\n\n"
                return
            for k in ("title", "system_prompt", "params", "notes",
                      "digest", "preset_id", "effort", "use_rag",
                      "authors_note", "authors_note_depth", "archive"):
                s[k] = fresh.get(k, s.get(k))
            if prefill:
                s["prefill"] = ""   # consumed once, like a variant
            last = s["messages"][-1] if s["messages"] else None
            if cont and last and last.get("role") == "assistant":
                last["content"] = last.get("content", "") + text
                if thinking:
                    last["thinking"] = last.get("thinking", "") + thinking
            else:
                msg = {"role": "assistant", "content": _with_prefill(prefill, text)}
                if thinking:
                    msg["thinking"] = thinking
                if trace:
                    msg["tool_trace"] = trace
                if timings.get("predicted_per_second"):
                    msg["stats"] = {
                        "tps": round(timings["predicted_per_second"], 1),
                        "tokens": timings.get("predicted_n"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "model": (st.read_state() or {}).get("model", "")}
                s["messages"].append(msg)
            sessions.save(s)
            _bump_stats(timings)
        yield b"data: [DONE]\n\n"

    @app.get("/api/server")
    async def server_info():
        from . import server_ops
        s = st.server_running()
        if s is None:
            return JSONResponse({"error": "not running"}, status_code=404)
        exp = server_ops.expected_tg(s["model"], s["quant"],
                                     s.get("backend", "unknown"))
        info = {k: s.get(k) for k in ("model", "quant", "backend", "use_case",
                                      "ctx", "started_at", "public_port",
                                      "unloaded")}
        info.update(server_ops.ram_snapshot())
        info["engine_version"] = server_ops.engine_version()
        info["last_tg"] = telemetry["tg"]
        info["expected_tg"] = exp
        info["verdict"] = server_ops.verdict(telemetry["tg"], exp)
        info["openai_base"] = f"http://127.0.0.1:{s['public_port']}/v1"
        return info

    @app.get("/api/server/stats")
    async def server_stats():
        try:
            f = st.rigma_home() / "stats.json"
            data = json.loads(f.read_text(encoding="utf-8")) if f.exists() \
                else {}
        except Exception:
            data = {}
        return {"total_tokens": data.get("total_tokens", 0),
                "total_turns": data.get("total_turns", 0),
                "by_model": data.get("by_model", {})}

    @app.get("/api/server/log")
    async def server_log(lines: int = 200):
        from . import server_ops
        return Response(server_ops.log_tail(lines), media_type="text/plain",
                        headers=_NO_STORE)

    @app.get("/api/server/switch-options")
    async def server_switch_options():
        from . import server_ops
        s = st.server_running()
        if s is None:
            return JSONResponse({"error": "not running"}, status_code=404)
        return await asyncio.to_thread(server_ops.switch_options, s, registry)

    @app.post("/api/server/switch")
    async def server_switch(body: dict | None = None):
        from . import server_ops
        model = str((body or {}).get("model", "")).strip()
        if not model:
            return JSONResponse({"error": "model required"}, status_code=400)
        if not switch_lock.acquire(blocking=False):
            return JSONResponse({"error": "a switch is already in progress"},
                                status_code=409)
        try:
            new_state = await asyncio.to_thread(server_ops.perform_switch,
                                                model, registry)
            telemetry["tg"] = None
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            switch_lock.release()
        return new_state

    @app.post("/api/server/ctx")
    async def server_ctx(body: dict):
        """Relaunch the running model at a requested context size (owner
        finding 2026-07-18: combos launch at 32K and the UI had no way up)."""
        from . import server_ops
        try:
            want = int(body.get("ctx", 0))
        except (TypeError, ValueError):
            want = 0
        if want < 2048:
            return JSONResponse({"error": "ctx must be at least 2048"},
                                status_code=400)
        s = st.read_state()
        if s is None:
            return JSONResponse({"error": "not running"}, status_code=404)
        if not switch_lock.acquire(blocking=False):
            return JSONResponse({"error": "a switch is already in progress"},
                                status_code=409)
        try:
            new_state = await asyncio.to_thread(
                server_ops.perform_switch, s["model"], registry, None, want)
            telemetry["tg"] = None
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            switch_lock.release()
        return new_state

    @app.post("/api/server/unload")
    async def server_unload():
        from . import server_ops
        if not switch_lock.acquire(blocking=False):
            return JSONResponse({"error": "a switch is already in progress"},
                                status_code=409)
        try:
            s = await asyncio.to_thread(server_ops.perform_unload)
            telemetry["tg"] = None
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        finally:
            switch_lock.release()
        return s

    @app.post("/api/server/load")
    async def server_load():
        from . import server_ops
        if not switch_lock.acquire(blocking=False):
            return JSONResponse({"error": "a switch is already in progress"},
                                status_code=409)
        try:
            s = await asyncio.to_thread(server_ops.perform_load, registry)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            switch_lock.release()
        return s

    @app.post("/api/workspace/pack")
    async def workspace_pack(body: dict):
        from . import workspace
        folder = str(body.get("folder", "")).strip().strip('"')
        if not folder:
            return JSONResponse({"error": "folder required"}, status_code=400)
        try:
            return await asyncio.to_thread(workspace.pack_folder, folder)
        except workspace.WorkspaceError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    # ---- Hugging Face browser (Bazaar) --------------------------------------

    @app.get("/api/hf/search")
    async def hf_search(q: str = ""):
        from . import hangar
        from . import hf_browse
        if not q.strip():
            return []
        try:
            return await asyncio.to_thread(hf_browse.search, q.strip())
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=502)

    @app.get("/api/hf/repo")
    async def hf_repo(id: str):
        from . import hangar
        from . import hf_browse
        try:
            return await asyncio.to_thread(hf_browse.inspect_repo, id,
                                           registry)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/hf/add")
    async def hf_add(body: dict):
        from . import hangar
        from . import hf_browse
        repo = str(body.get("repo", "")).strip()
        if not repo:
            return JSONResponse({"error": "repo required"}, status_code=400)
        try:
            spec = await asyncio.to_thread(hf_browse.add_model, repo, registry)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"slug": spec.slug, "capabilities": spec.capabilities}

    # ---- model manager (Hangar) ---------------------------------------------

    @app.get("/api/models")
    async def models_list():
        from . import hangar
        out = await asyncio.to_thread(hangar.list_models, registry)
        now = _now()

        def _with_rate(item):
            p = item.get("pull")
            if p and p.get("status") == "downloading":
                done = hangar.pull_progress(item["file"], item["bytes"])
                prev = pull_samples.get(item["file"])   # ~1.5s between polls
                bps = ((done - prev[1]) / (now - prev[0])
                       if prev and now > prev[0] and done >= prev[1] else None)
                pull_samples[item["file"]] = (now, done)
                eta = int((item["bytes"] - done) / bps) if bps and bps > 0 \
                    else None
                item["pull"] = {**p, "done": done, "bps": bps, "eta": eta}
            elif p is None:
                pull_samples.pop(item["file"], None)

        for m in out["models"]:
            for q in m["quants"]:
                _with_rate(q)
            if m.get("mmproj"):
                _with_rate(m["mmproj"])
        return JSONResponse(out, headers=_NO_STORE)

    @app.post("/api/models/install")
    async def models_install(body: dict):
        from . import hangar
        path = str(body.get("path", "")).strip().strip('"')
        if not path:
            return JSONResponse({"error": "path required"}, status_code=400)
        try:
            spec = await asyncio.to_thread(hangar.install_model, path,
                                           body.get("attach_to") or None)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"slug": spec.slug, "capabilities": spec.capabilities}

    @app.post("/api/models/upload")
    async def models_upload(request: Request, filename: str = "",
                            attach_to: str = ""):
        """Drag-drop install: raw body stream (no multipart dependency)."""
        from pathlib import Path as _P

        from . import hangar
        name = _P(filename).name        # strips any path-traversal attempt
        if not name.lower().endswith(".gguf"):
            return JSONResponse({"error": "only .gguf files can be installed"},
                                status_code=400)
        inbox = hangar.rigma_home() / "custom" / "incoming"
        inbox.mkdir(parents=True, exist_ok=True)
        tmp, staged = inbox / (name + ".part"), inbox / name
        try:
            with open(tmp, "wb") as f:
                async for chunk in request.stream():
                    await asyncio.to_thread(f.write, chunk)   # GBs: don't block
            os.replace(tmp, staged)
            spec = await asyncio.to_thread(hangar.install_model, staged,
                                           attach_to or None)
        except hangar.HangarError as e:
            staged.unlink(missing_ok=True)
            return JSONResponse({"error": str(e)}, status_code=400)
        finally:
            tmp.unlink(missing_ok=True)
        return {"slug": spec.slug, "capabilities": spec.capabilities}

    @app.post("/api/models/{slug}/pull")
    async def models_pull(slug: str, body: dict):
        from . import hangar
        try:
            return hangar.start_pull(slug, str(body.get("file", "")), registry)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.delete("/api/models/{slug}/files/{file}")
    async def models_delete_file(slug: str, file: str):
        from . import hangar
        try:
            await asyncio.to_thread(hangar.delete_file, slug, file, registry)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"ok": True}

    @app.delete("/api/models/{slug}")
    async def models_delete(slug: str):
        from . import hangar
        try:
            await asyncio.to_thread(hangar.delete_model, slug, registry)
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"ok": True}

    @app.patch("/api/models/{slug}")
    async def models_patch(slug: str, body: dict):
        from . import hangar
        try:
            spec = hangar.patch_capabilities(
                slug, list(body.get("capabilities", [])))
        except hangar.HangarError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"slug": spec.slug, "capabilities": spec.capabilities}

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
        if isinstance(q, list):   # vision parts -> text only for the sidecar
            q = " ".join(p.get("text", "") for p in q
                         if isinstance(p, dict) and p.get("type") == "text")
        yield _sse({"delta": ""}, event="think")   # keep-alive: retrieval is slow
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
        if sessions.load(s["id"]) is None:   # deleted mid-retrieval: discard
            yield b"data: [DONE]\n\n"
            return
        s["messages"].append({"role": "assistant", "content": text})
        sessions.save(s)
        yield b"data: [DONE]\n\n"

    @app.post("/api/sessions/{sid}/chat")
    async def chat_turn(sid: str, body: dict):
        activity["last"] = _now()   # keep-alive
        await _ensure_loaded()      # reload if idle-unloaded
        s = sessions.load(sid)
        if s is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        message = body.get("message")

        def _has_img(content):
            return (isinstance(content, list) and
                    any(isinstance(p, dict) and p.get("type") == "image_url"
                        for p in content))

        has_image = _has_img(message) or any(
            _has_img(m.get("content")) for m in s.get("messages", []))
        if has_image:
            caps = None   # None = capabilities unknown (stale cache etc.)
            try:
                from .registry import Registry
                reg = registry if registry is not None else Registry.load()
                caps = reg.models[(st.read_state() or {}).get("model", "")
                                  ].capabilities
            except Exception:
                caps = None
            # only block when we POSITIVELY know the model lacks vision;
            # unknown -> pass through and let the engine answer honestly
            if caps is not None and "vision" not in caps:
                return JSONResponse(
                    {"error": "this model can't see images — switch to a "
                              "vision-capable model (⚙ → Server) or "
                              "delete the image message"},
                    status_code=400)
        if message:
            s["messages"].append({"role": "user", "content": message})
            if s.get("title") == "New chat":
                title = message if isinstance(message, str) else next(
                    (p.get("text", "") for p in message
                     if isinstance(p, dict) and p.get("type") == "text"), "chat")
                s["title"] = str(title)[:40]
            sessions.save(s)
        if not s["messages"]:
            return JSONResponse({"error": "session has no messages"},
                                status_code=400)
        if body.get("continue") and s.get("use_rag"):
            return JSONResponse(
                {"error": "continue is not available for grounded chats"},
                status_code=400)
        # _llm_turn now folds in the agentic tool loop (streams each round,
        # runs any tool_calls, loops) — RAG is the only separate path
        if s.get("use_rag"):
            gen = _rag_turn(s)
        else:
            gen = _llm_turn(s, cont=bool(body.get("continue")))
        return StreamingResponse(gen, media_type="text/event-stream",
                                 headers=_NO_STORE)

    @app.on_event("startup")
    async def _keepalive_task():
        import os
        mins = float(os.environ.get("RIGMA_KEEP_ALIVE_MIN", "0") or 0)
        if mins <= 0:
            return   # opt-in: 0 disables idle auto-unload

        async def _loop():
            from . import server_ops
            while True:
                await asyncio.sleep(30)
                s = st.read_state()
                if not s or s.get("unloaded"):
                    continue
                if activity["last"] and _now() - activity["last"] > mins * 60:
                    if switch_lock.acquire(blocking=False):
                        try:
                            await asyncio.to_thread(server_ops.perform_unload)
                            telemetry["tg"] = None
                        except Exception:
                            pass
                        finally:
                            switch_lock.release()
        asyncio.create_task(_loop())

    @app.api_route("/v1/{path:path}",
                   methods=["GET", "POST", "OPTIONS", "DELETE"])
    async def proxy(request: Request, path: str):
        activity["last"] = _now()   # keep-alive: external agents count too
        await _ensure_loaded()      # auto-reload if idle-unloaded
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS}
        upstream = client.build_request(
            request.method, f"/v1/{path}", headers=headers,
            content=await request.body())
        resp = await client.send(upstream, stream=True)
        media = resp.headers.get("content-type", "application/json")
        # pass upstream headers through (ratelimit, cors, etc.); drop hop-by-hop
        out_headers = {k: v for k, v in resp.headers.items()
                       if k.lower() not in _HOP_HEADERS
                       and k.lower() != "content-type"}
        if "text/event-stream" in media:
            async def gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
            return StreamingResponse(gen(), status_code=resp.status_code,
                                     media_type=media, headers=out_headers)
        body = await resp.aread()
        await resp.aclose()
        return Response(content=body, status_code=resp.status_code,
                        media_type=media, headers=out_headers)

    return app


def run_ui(public_port: int, upstream_port: int) -> None:
    import uvicorn
    uvicorn.run(build_app(upstream_port), host="127.0.0.1", port=public_port,
                log_level="warning")
