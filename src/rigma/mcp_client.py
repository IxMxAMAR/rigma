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
import re
import subprocess
import sys
import threading

from .runtime import rigma_home

PROTOCOL_VERSION = "2025-06-18"
_START_TIMEOUT = 20.0     # server boot + initialize handshake
_CALL_TIMEOUT = 60.0
_RESULT_MAX = 8000        # same cap philosophy as built-in tools
# One JSON-RPC frame is one line, and a legitimate "read this big file" call can
# be huge — but it must not be held in memory whole, twice, before the 8000-char
# display cap is applied (AUDIT F55).
_FRAME_MAX = 4_000_000
_ID_RE = re.compile(r'"id"\s*:\s*(\d+)')


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
        # A reader that exited is a DEAD server even while proc.poll() still says
        # alive: every later call would otherwise block a worker thread for the
        # full timeout with no explanation (AUDIT F55).
        self.reader_error: str = ""
        self.oversize_frames = 0

    # -- plumbing --------------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except ValueError:
            return
        mid = msg.get("id")
        q = self._replies.get(mid) if mid is not None else None
        if q is not None:
            q.put(msg)

    def _fail_frame(self, head: str) -> None:
        """Fail the caller whose reply lives in an OVERSIZE frame, by the id at
        the frame's head. Without this the caller waits out the full call
        timeout for a reply that was deliberately dropped, and the model is told
        nothing at all."""
        m = _ID_RE.search(head)
        if not m:
            return
        q = self._replies.get(int(m.group(1)))
        if q is not None:
            q.put({"jsonrpc": "2.0", "error": {
                "code": -32001,
                "message": (f"response exceeded {_FRAME_MAX} characters and "
                            "was dropped")}})

    def _drain_one_frame(self, budget: int = 4):
        """Read past the newline that ends an OVERSIZE frame.

        Returns the remainder of the read (so framing stays in sync), or None
        when the server is gone or streamed one endless line."""
        for _ in range(budget):
            tail = self.proc.stdout.readline(_FRAME_MAX + 1)
            if not tail:
                return None
            if "\n" in tail:
                return tail.split("\n", 1)[1]
        return None

    def _reader(self):
        """Read newline-delimited JSON-RPC frames with a HARD bound.

        This used `for line in iter(stdout.readline, "")`, which holds one whole
        line in memory before parsing — and _RESULT_MAX is applied only
        afterwards, so two copies of the payload lived inside the process that
        also holds the chat sessions. Worse, a MemoryError was swallowed by the
        bare `except`, the thread exited for good, and `proc.poll()` still
        reported the child alive, so every later call blocked a worker for the
        full 60s with no explanation and `_ensure()` would not rebuild it
        (AUDIT F55)."""
        pending = ""
        try:
            while True:
                chunk = self.proc.stdout.readline(_FRAME_MAX + 1)
                if not chunk:
                    break                       # EOF: the child is gone
                if "\n" not in chunk and len(chunk) > _FRAME_MAX:
                    self.oversize_frames += 1
                    self._fail_frame(chunk)
                    rest = self._drain_one_frame()
                    if rest is None:
                        self.reader_error = ("a single response exceeded "
                                             f"{_FRAME_MAX} characters")
                        return
                    pending = rest
                    continue
                pending += chunk
                while "\n" in pending:
                    line, _, pending = pending.partition("\n")
                    self._handle_line(line)
        except Exception as e:
            self.reader_error = f"{type(e).__name__}: {e}"[:200]
        finally:
            # wake anyone waiting on a reply that can no longer arrive
            with self._lock:
                for q in self._replies.values():
                    q.put({"jsonrpc": "2.0", "error": {
                        "code": -32000,
                        "message": (f"mcp server '{self.name}' is no longer "
                                    "answering")}})
                self._replies.clear()

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
        """Terminate, escalate, and release the pipes.

        `terminate()` then dropping the reference left a server that survived it
        unkillable forever — a repeat `stop_all()` was a no-op — and the reader
        thread stayed blocked in readline() holding the stdout pipe, so every
        config-change restart and every shutdown leaked a live thread plus a
        handle. On Windows a server configured as `cmd /c npx …` left the node
        grandchild running. The codebase gets this right four other times
        (runtime.py, state.py, rag.py, tools._kill_tree); this mirrors
        ServerProcess.stop: terminate -> wait -> kill -> wait, then close the
        pipes. Closing stdin is also the MCP graceful-shutdown signal
        (AUDIT F54)."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()      # MCP's own graceful-shutdown signal
        except Exception:
            pass
        if proc.poll() is None:
            killed = False
            if sys.platform == "win32":
                # terminate() reaches only the direct child, so a server
                # configured as `cmd /c npx …` leaves the node grandchild
                # running; taskkill /T takes the tree.
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False,
                        creationflags=getattr(subprocess,
                                              "CREATE_NO_WINDOW", 0))
                    killed = True
                except Exception:
                    pass
            if not killed:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

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
            # A server whose READER died is dead even though its process is
            # alive: it can never answer again, so it must not be left in
            # `_servers` to block a worker thread per call (AUDIT F55).
            stale = {n: s.reader_error for n, s in self._servers.items()
                     if s.reader_error}
            if self._started and key == self._cfg_key and not stale:
                return
            # config changed (or first use): restart the world
            for s in self._servers.values():
                s.stop()
            # a config edit may fix a dead server, so only carry the death
            # forward when the config is the one it died under
            carried = dict(stale) if key == self._cfg_key else {}
            self._servers, self._failed = {}, carried
            self._cfg_key, self._started = key, True
            for name, spec in cfg.items():
                if not isinstance(spec, dict) or str(name) in carried:
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
        if srv.reader_error:
            return (f"error: mcp server '{server}' stopped responding "
                    f"({srv.reader_error}) — it will be restarted on the next "
                    "config reload")
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
                "dead": {n: s.reader_error for n, s in self._servers.items()
                         if s.reader_error},
                "oversize_frames": {n: s.oversize_frames
                                    for n, s in self._servers.items()
                                    if s.oversize_frames},
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
