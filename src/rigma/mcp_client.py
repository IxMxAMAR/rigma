"""MCP host: every MCP server ever written becomes a Rigma tool.

Config lives at ~/.rigma/mcp.json in the de-facto standard notation (the
Claude Desktop / Cursor / LM Studio `mcpServers` schema — never invent a
format):

    {"mcpServers": {
        "filesystem": {"command": "npx",
                       "args": ["-y", "@modelcontextprotocol/server-filesystem",
                                "D:/docs"],
                       "env": {}}}}

stdio transport only (newline-delimited JSON-RPC 2.0) — it covers the bulk
of the ecosystem; remote SSE/HTTP servers can come later. No SDK: the client
side of MCP-over-stdio is initialize → notifications/initialized →
tools/list → tools/call, which is ~200 lines of subprocess + json.

Discovered tools are namespaced `mcp__<server>__<tool>` and merged into the
ordinary registry surface (tools.tool_specs / tools.run_tool), where they
ride the existing gates: allow_code sessions only, excluded under the
no-network and confined run profiles. The built-in tools become the curated
floor instead of the ceiling.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

from .runtime import rigma_home

PROTOCOL_VERSION = "2025-06-18"
_START_TIMEOUT = 20.0     # server boot + initialize handshake
_CALL_TIMEOUT = 60.0
_RESULT_MAX = 8000        # same cap philosophy as built-in tools


def config_path():
    return rigma_home() / "mcp.json"


def load_config() -> dict:
    try:
        raw = json.loads(config_path().read_text(encoding="utf-8"))
        servers = raw.get("mcpServers") or {}
        return servers if isinstance(servers, dict) else {}
    except (FileNotFoundError, OSError, ValueError):
        return {}


class McpError(RuntimeError):
    pass


class McpServer:
    """One stdio MCP server: a subprocess we speak JSON-RPC to."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self._id = 0
        self._lock = threading.Lock()
        self._replies: dict[int, queue.Queue] = {}

    # -- plumbing --------------------------------------------------------------
    def _reader(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                mid = msg.get("id")
                q = self._replies.get(mid) if mid is not None else None
                if q is not None:
                    q.put(msg)
        except Exception:
            pass

    def _send(self, method: str, params: dict | None = None,
              notify: bool = False, timeout: float = _CALL_TIMEOUT):
        if self.proc is None or self.proc.poll() is not None:
            raise McpError(f"mcp server '{self.name}' is not running")
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if notify:
            with self._lock:
                self.proc.stdin.write(json.dumps(msg) + "\n")
                self.proc.stdin.flush()
            return None
        with self._lock:
            self._id += 1
            mid = self._id
            msg["id"] = mid
            self._replies[mid] = queue.Queue()
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        try:
            reply = self._replies[mid].get(timeout=timeout)
        except queue.Empty:
            raise McpError(f"'{self.name}' timed out on {method}") from None
        finally:
            self._replies.pop(mid, None)
        if "error" in reply:
            e = reply["error"]
            raise McpError(str(e.get("message", e)) if isinstance(e, dict)
                           else str(e))
        return reply.get("result")

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        cmd = [str(self.spec.get("command", ""))] + [
            str(a) for a in (self.spec.get("args") or [])]
        if not cmd[0]:
            raise McpError(f"'{self.name}': no command configured")
        env = {**os.environ, **{str(k): str(v) for k, v in
                                (self.spec.get("env") or {}).items()}}
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            env=env, **kw)
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "rigma", "version": "0.9"}},
            timeout=_START_TIMEOUT)
        self._send("notifications/initialized", {}, notify=True)
        listed = self._send("tools/list", {}, timeout=_START_TIMEOUT) or {}
        self.tools = [t for t in listed.get("tools", [])
                      if isinstance(t, dict) and t.get("name")]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def call(self, tool: str, args: dict) -> str:
        result = self._send("tools/call",
                            {"name": tool, "arguments": args or {}}) or {}
        parts = []
        for c in result.get("content", []):
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
            elif isinstance(c, dict):
                parts.append(f"[{c.get('type', 'content')}]")
        text = "\n".join(parts).strip() or "(no content)"
        if result.get("isError"):
            text = "error: " + text
        return text[:_RESULT_MAX] + ("\n…(truncated)"
                                     if len(text) > _RESULT_MAX else "")


class McpManager:
    """Lazy singleton over the configured servers. First use starts them;
    a server that fails to boot is remembered as dead and skipped (one loud
    line in its place, not a crash)."""

    def __init__(self):
        self._servers: dict[str, McpServer] = {}
        self._failed: dict[str, str] = {}
        self._started = False
        self._lock = threading.Lock()
        self._cfg_key = None

    def _ensure(self) -> None:
        cfg = load_config()
        key = json.dumps(cfg, sort_keys=True)
        with self._lock:
            if self._started and key == self._cfg_key:
                return
            # config changed (or first use): restart the world
            for s in self._servers.values():
                s.stop()
            self._servers, self._failed = {}, {}
            self._cfg_key, self._started = key, True
            for name, spec in cfg.items():
                if not isinstance(spec, dict):
                    continue
                srv = McpServer(str(name), spec)
                try:
                    srv.start()
                    self._servers[srv.name] = srv
                except Exception as e:
                    srv.stop()
                    self._failed[str(name)] = str(e)[:200]

    def tool_specs(self) -> list[dict]:
        """OpenAI-format specs for every discovered MCP tool, namespaced so
        run_tool can route them back."""
        self._ensure()
        out = []
        for srv in self._servers.values():
            for t in srv.tools:
                schema = t.get("inputSchema") or {"type": "object",
                                                  "properties": {}}
                desc = str(t.get("description") or t["name"])[:1000]
                out.append({"type": "function", "function": {
                    "name": f"mcp__{srv.name}__{t['name']}",
                    "description": f"[{srv.name}] {desc}",
                    "parameters": schema}})
        return out

    def call(self, namespaced: str, args: dict) -> str:
        self._ensure()
        try:
            _, server, tool = namespaced.split("__", 2)
        except ValueError:
            return f"error: malformed mcp tool name '{namespaced}'"
        srv = self._servers.get(server)
        if srv is None:
            why = self._failed.get(server, "not configured")
            return f"error: mcp server '{server}' is unavailable ({why})"
        try:
            return srv.call(tool, args)
        except Exception as e:
            return f"error: mcp {server}/{tool}: {e}"

    def status(self) -> dict:
        cfg = load_config()
        self._ensure()
        return {"configured": sorted(cfg),
                "running": sorted(self._servers),
                "failed": dict(self._failed),
                "tools": [t["function"]["name"] for t in self.tool_specs()]}

    def stop_all(self) -> None:
        with self._lock:
            for s in self._servers.values():
                s.stop()
            self._servers = {}
            self._started = False


_manager: McpManager | None = None


def manager() -> McpManager:
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager
