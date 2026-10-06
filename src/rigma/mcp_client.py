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

Three properties of the host are load-bearing and easy to get subtly wrong, so
they are stated here rather than only where they are implemented:

* **The launcher is resolved, not passed through.** `"command": "npx"` is what
  every MCP README and Claude Desktop config says, and Node-based clients
  launch it because they spawn through a shell. `subprocess.Popen` on Windows
  goes to CreateProcess, which does NOT consult PATHEXT and so cannot find
  `npx.cmd` — the identical config block worked in Claude and failed in Rigma.
  `shutil.which` honours PATHEXT, so it is what the command goes through.

* **A non-text result is an artifact, not a marker.** `tools/call` may answer
  with base64 image/audio data or a resource body. Rendering those as the
  literal string `[image]` gives the model a word where the picture was, which
  is indistinguishable from a broken server. They are written to
  `$RIGMA_HOME/mcp_out/` and the model is handed the PATH.

* **The config is reconciled per server, every turn.** `mcp.json` is re-read on
  every call and each running server is compared against the spec it was
  STARTED under, so editing one server restarts that one server and deleting
  one stops it. A model that installs an MCP server and then edits it must not
  keep talking to the old process with no way to find out — and, because a
  restart discards the server's in-process state, a server nobody edited must
  not be restarted at all.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .atomicio import atomic_write_bytes
from .runtime import rigma_home

PROTOCOL_VERSION = "2025-06-18"
_START_TIMEOUT = 20.0     # server boot + initialize handshake
_CALL_TIMEOUT = 60.0
# A server that accepts a request and never answers is DEAD for our purposes,
# but a timeout changed no state — so `reader_error` stayed empty and `_ensure`
# kept the mute process in `_servers` forever, costing a worker thread per call
# (09-5). Consecutive timeouts past this mark it wedged and restart it.
_WEDGE_AFTER = 2
_RESULT_MAX = 8000        # same cap philosophy as built-in tools
# One JSON-RPC frame is one line, and a legitimate "read this big file" call can
# be huge — but it must not be held in memory whole, twice, before the 8000-char
# display cap is applied (AUDIT F55).
_FRAME_MAX = 4_000_000
_ID_RE = re.compile(r'"id"\s*:\s*(\d+)')

# Non-text content (base64 `data` + `mimeType`) is decoded to a file under
# $RIGMA_HOME/mcp_out/ and the model is handed the PATH. That turns a result the
# model could not use into one it can, but it also turns a server's answer into
# WRITES on this machine — so there is a hard per-artifact ceiling.
#
# 4 MiB, with the honest accounting: the stdio frame cap below is the TIGHTER
# bound today. One JSON-RPC frame is at most _FRAME_MAX (4e6) characters and
# base64 spends 4 characters per 3 bytes, so no artifact arriving over this
# transport can exceed ~2.9 MB — a 1024x1024 ComfyUI PNG (~1.5 MB) fits
# comfortably and a 4K one does not fit at all, cap or no cap. This is therefore
# a SECOND bound rather than the first: it survives raising _FRAME_MAX, and it
# states a number for "how much disk may one server's answer cost" instead of
# leaving the frame cap to imply one. It is checked BEFORE the write, and an
# over-cap payload is reported and discarded, never truncated onto disk — half a
# PNG at a content-addressed name would be read as a real image forever after.
#
# Nothing prunes mcp_out. Content-addressing makes a repeat of the same picture
# free, but a session that generates a thousand DISTINCT ones leaves a thousand
# files. The alternative is deleting a path the model may still hold from an
# earlier turn, which trades a disk-space problem for a correctness one; so the
# cap is per artifact, deliberately, and the directory is the user's to clear.
_ARTIFACT_MAX = 4 * 1024 * 1024

# Extension for the on-disk artifact, by media type. Unknown types get `.bin`:
# guessing is worse than being unhelpful, because the model reads the extension
# as a claim about the bytes.
_EXT_BY_MIME = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
    "image/tiff": ".tiff", "image/svg+xml": ".svg",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/webm": ".weba",
    "application/pdf": ".pdf", "application/json": ".json",
    "text/plain": ".txt", "application/octet-stream": ".bin",
}


def config_path():
    return rigma_home() / "mcp.json"


def artifact_dir() -> Path:
    """Where non-text MCP results are decoded to, so the model can read them.

    Inside $RIGMA_HOME rather than a temp dir on purpose: the model is given
    this path and may use it several turns later, and the OS temp cleaner is
    entitled to delete anything in %TEMP% whenever it likes.
    """
    return rigma_home() / "mcp_out"


def _spec_fingerprint(spec: dict) -> str:
    """A stable string answering "is this the same server spec?".

    Compared per server in `_ensure` so that a config edit restarts exactly the
    server it edited. `sort_keys` because the order of `args` is meaningful but
    the order of `env` keys is not, and a reordered `env` must not look like a
    change worth destroying a running server's state for. `default=str` and the
    fallback because a fingerprint must never be the thing that raises: the spec
    is whatever the user wrote in a file we do not control.
    """
    try:
        return json.dumps(spec, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(spec)


def _resolve_command(name: str, path: str | None = None) -> str:
    """The launcher `name` actually names, or `name` unchanged.

    Measured on this machine (Windows, Python 3.12):

        subprocess.run(["npx", "--version"])      -> FileNotFoundError [WinError 2]
        subprocess.run(["npx.cmd", "--version"])  -> rc=0  11.12.1
        subprocess.run(["uvx", "--version"])      -> rc=0  uvx 0.10.10

    `uvx` works because it is a real `.exe`; `npx` is a `.cmd` shim, and
    `Get-Command npx` finds `npx.ps1`. CreateProcess — what `subprocess` uses on
    Windows unless a shell is requested — resolves neither, so the config block
    every MCP README and Claude Desktop writes, `"command": "npx"`, failed in
    Rigma while working verbatim in every Node-based client. `shutil.which`
    consults PATHEXT, so it finds `npx.CMD` and the same config works here too.

    `path` is the PATH the CHILD will run with, when the spec sets one, and it
    matters for the same CreateProcess reason: the search happens in the calling
    process, against the calling process's PATH, and the `env` block handed to
    the child is not consulted. A spec that uses `env.PATH` to reach a privately
    installed launcher would otherwise be resolved against the wrong directories
    and then launched from somewhere it does not exist.

    Deliberately NOT a config rewrite and NOT a requirement to write `npx.cmd`:
    the user's file is the de-facto standard format and stays valid and portable.
    A name that resolves to nothing is returned unchanged, so the failure still
    names exactly what the user wrote — see `start`, which turns the resulting
    FileNotFoundError into a message about PATH.
    """
    try:
        return shutil.which(name, path=path) or name
    except Exception:
        # which() stats the filesystem; a permission error on one PATH entry is
        # not a reason to refuse to start a server that might launch fine.
        return name


def _save_artifact(b64: str, mime: str, label: str) -> str:
    """Decode one base64 payload to `$RIGMA_HOME/mcp_out/` and describe it.

    Content-addressed by the sha256 of the DECODED bytes, which buys two things
    a counter or a timestamp would not. A server that returns the same picture
    on every call — the normal case for a "render this prompt" tool whose prompt
    did not change — writes ONE file instead of one per turn, and the path the
    model was handed on turn 1 is still the correct path on turn 50, so a note
    the model wrote down does not rot.

    Written through `atomicio` because a truncated PNG at a stable name is worse
    than no file at all: the model would read it, see garbage, and have no way to
    tell that the server was fine and the disk write was not.

    Never raises. The contract of `run_tool` is that a tool reports failure as
    text the model can react to, and that applies here — a call that saved
    nothing must still say so, and must not lose the text blocks beside it.
    """
    try:
        # Some servers pretty-print their base64 across lines. Whitespace is not
        # a corruption signal, so it is stripped before the strict decode; a
        # genuinely non-base64 byte still fails and is reported.
        raw = base64.b64decode("".join(str(b64).split()), validate=True)
    except (ValueError, TypeError) as e:
        # binascii.Error is a ValueError, so this covers a bad alphabet and bad
        # padding as well as a non-str argument.
        return f"[{label}: could not be decoded as base64 ({e})]"
    if len(raw) > _ARTIFACT_MAX:
        return (f"[{label}: {len(raw)} bytes is over the {_ARTIFACT_MAX}-byte "
                "cap Rigma will write to disk, so it was not saved]")
    ext = _EXT_BY_MIME.get(str(mime or "").split(";")[0].strip().lower(), ".bin")
    path = artifact_dir() / (hashlib.sha256(raw).hexdigest() + ext)
    try:
        atomic_write_bytes(path, raw)
    except Exception as e:
        # OSError in practice — a full disk, a denied $RIGMA_HOME, a Windows
        # sharing violation that outlived atomicio's retries — but the "never
        # raises" promise above is worth more than the precision of the catch.
        return f"[{label}: could not be written to {path} ({e})]"
    return (f"{label} saved to {path} ({len(raw)} bytes, "
            f"{mime or 'unknown media type'}) — it is a file on disk that you "
            "can read or inspect.")


def _render_content(c: dict) -> str:
    """One MCP content block as text the model can actually use.

    `tools/call` may answer with `text`, `image`, `audio`, `resource` or
    `resource_link`. The old renderer kept `text` and collapsed everything else
    to the literal marker `[image]`, so a ComfyUI-style server that generates a
    picture handed Rigma's model the six characters `[image]` and nothing else.
    The failure is invisible from both ends: the server worked, the transport
    worked, and the model still could not do anything with the answer.

    Text stays text. Anything carrying base64 `data` (image and audio have the
    same shape, so they share one path) is decoded to a file and replaced by its
    PATH, because the model can read a file and cannot read a marker. A
    `resource` with a `text` body is inlined with its URI; with a `blob` body it
    takes the same disk path as an image. A `resource_link` is only a URI — the
    host has no `resources/read` yet — so it is passed through as text, which is
    at least actionable, rather than dropped.
    """
    kind = str(c.get("type") or "content")
    if kind == "text":
        return str(c.get("text", ""))
    if kind in ("image", "audio"):
        mime = str(c.get("mimeType") or "")
        return _save_artifact(str(c.get("data") or ""), mime, kind)
    if kind == "resource":
        res = c.get("resource")
        if not isinstance(res, dict):
            return "[resource]"
        uri = str(res.get("uri") or "")
        if isinstance(res.get("text"), str):
            return f"resource {uri}:\n{res['text']}" if uri else res["text"]
        if res.get("blob") is not None:
            return _save_artifact(str(res["blob"]),
                                  str(res.get("mimeType") or ""),
                                  f"resource {uri or '(no uri)'}")
        return f"[resource {uri or '(no uri)'}: no text or blob body]"
    if kind == "resource_link":
        return f"[resource_link {c.get('uri') or c.get('name') or '(no uri)'}]"
    return f"[{kind}]"


def load_config() -> dict:
    try:
        raw = json.loads(config_path().read_text(encoding="utf-8"))
        # A JSON array/string/null is not a config: `.get` raised AttributeError,
        # which is not in the caught tuple, so every MCP tool vanished with no
        # explanation and /api/mcp showed the Python error (09-8).
        if not isinstance(raw, dict):
            return {}
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
        # The spec this process was STARTED under, frozen here rather than
        # re-read later: `_ensure` compares it against the current file to decide
        # whether this process is still the one the user asked for. Captured at
        # construction because that is the moment the config was read.
        self.fingerprint = _spec_fingerprint(spec)
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
        # Consecutive call timeouts, and the flag they set. A live-but-mute
        # server is the same worker-thread tax as a dead reader, so it has to
        # become visible to `_ensure` the same way (09-5).
        self.timeouts = 0
        self.wedged = False

    # -- plumbing --------------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except ValueError:
            return
        # json.loads may return a list, string, number or null. `.get` on one of
        # those raised inside `_reader`'s try, which ended the reader for good —
        # and `_ensure` then tore down and rebuilt EVERY configured server for a
        # line that carries no protocol meaning (09-2).
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        try:
            q = self._replies.get(mid) if mid is not None else None
        except TypeError:
            return          # an unhashable JSON-RPC id is not a reply we sent
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
            self.timeouts += 1
            if self.timeouts >= _WEDGE_AFTER:
                self.wedged = True
                if not self.reader_error:
                    self.reader_error = (f"'{self.name}' timed out "
                                         f"{self.timeouts} times in a row")
            raise McpError(f"'{self.name}' timed out on {method}") from None
        else:
            self.timeouts = 0
        finally:
            self._replies.pop(mid, None)
        if "error" in reply:
            e = reply["error"]
            raise McpError(str(e.get("message", e)) if isinstance(e, dict)
                           else str(e))
        return reply.get("result")

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        # A string `args` is iterable, so `[str(a) for a in args]` launched the
        # server with one argv element per character; anything else non-list was
        # a TypeError that never named the field (09-9).
        args = self.spec.get("args") or []
        if isinstance(args, str):
            args = [args]
        elif not isinstance(args, (list, tuple)):
            raise McpError(f"'{self.name}': args must be a list of strings")
        cmd = [str(self.spec.get("command", ""))] + [str(a) for a in args]
        if not cmd[0]:
            raise McpError(f"'{self.name}': no command configured")
        env = {**os.environ, **{str(k): str(v) for k, v in
                                (self.spec.get("env") or {}).items()}}
        # Resolve BEFORE spawning, because CreateProcess does not consult
        # PATHEXT: `"command": "npx"` — the block in every MCP README — is a
        # FileNotFoundError on Windows even though `npx.cmd` is right there on
        # PATH. The user's config is not rewritten and `npx.cmd` is not
        # required; the bare name is made to work, which is what makes the same
        # mcp.json portable between Claude and Rigma. The spec's own PATH is
        # passed in because that is the PATH the child will run with, and
        # CreateProcess searches the parent's — not the child's.
        cmd[0] = _resolve_command(cmd[0], env.get("PATH"))
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                env=env, **kw)
        except FileNotFoundError as e:
            # The raw FileNotFoundError says "The system cannot find the file
            # specified" and never names the config field, so the one thing the
            # user needs — which `command` was wrong, and that PATH is where
            # Rigma looked — was the one thing it did not say.
            raise McpError(
                f"'{self.name}': command {cmd[0]!r} could not be launched. "
                "Rigma resolves the command through PATH (and PATHEXT on "
                "Windows) before spawning, so this name is not on PATH either. "
                "Use an absolute path, or install it."
            ) from e
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
        # A `result` that is not an object is the same trap as a non-object
        # JSON-RPC line (09-2), one level down: `.get` on a string or a list
        # raises, and the caller sees "mcp fake/shout: 'str' object has no
        # attribute 'get'" instead of anything about the server.
        if not isinstance(result, dict):
            return f"error: '{self.name}' answered tools/call with a non-object result"
        content = result.get("content")
        parts = []
        for c in content if isinstance(content, list) else []:
            if isinstance(c, dict):
                parts.append(_render_content(c))
        text = "\n".join(p for p in parts if p).strip() or "(no content)"
        if result.get("isError"):
            text = "error: " + text
        # The 8000-char display cap still applies to the WHOLE result, and it is
        # applied AFTER the artifacts are on disk — so a huge text block can
        # truncate the path of an image beside it. That is the existing
        # philosophy (a bounded result the model can see) and it is left alone;
        # the file exists either way, and a model that was told about one is not
        # worse off than one that was told `[image]`.
        return text[:_RESULT_MAX] + ("\n…(truncated)"
                                     if len(text) > _RESULT_MAX else "")


class McpManager:
    """Lazy singleton over the configured servers. First use starts them;
    a server that fails to boot is remembered as dead and skipped (one loud
    line in its place, not a crash).

    Reconciliation is per SERVER, not per file. `mcp.json` is re-read on every
    call and compared against what is running, so an edit takes effect on the
    next turn — but only the server that was edited is restarted, because a
    restart is destructive (see `_ensure`).
    """

    def __init__(self):
        self._servers: dict[str, McpServer] = {}
        self._failed: dict[str, str] = {}
        # The whole-file config key each failure was recorded under. A boot
        # failure is only evidence about the config it happened under: retrying
        # it every turn spawns a doomed process forever, and never retrying it
        # leaves a server the user just fixed permanently dead. Per NAME rather
        # than one key for the manager, so two servers that died under different
        # configs are each retried on their own evidence.
        self._failed_under: dict[str, str] = {}
        self._lock = threading.Lock()

    def running(self) -> bool:
        """Whether any server is up.

        `tools.tool_specs` skips MCP entirely when `load_config()` is empty, for
        zero overhead — and that skip also skipped `_ensure()`, so deleting the
        LAST server from `mcp.json` left its process alive until Rigma restarted.
        The guard consults this so the empty-config case still reconciles.
        """
        return bool(self._servers)

    def _forget(self, name: str) -> None:
        """Drop a recorded boot failure, for a name that is now up or gone."""
        self._failed.pop(name, None)
        self._failed_under.pop(name, None)

    def _ensure(self) -> None:
        cfg = load_config()
        key = json.dumps(cfg, sort_keys=True)
        with self._lock:
            # 1. Servers that are running but are no longer the ones the config
            #    asks for. A restart DISCARDS the server's in-process state — an
            #    indexed corpus, a loaded model, a browser session, a warm cache
            #    — so it is spent only on a server whose spec genuinely differs,
            #    never on every turn and never on a server that did not change.
            #    The old code compared one key for the WHOLE file and, on any
            #    difference, stopped and restarted every server: adding a second
            #    server threw away the first one's state for an edit that never
            #    mentioned it.
            for name in list(self._servers):
                srv = self._servers[name]
                spec = cfg.get(name)
                if isinstance(spec, dict) and srv.fingerprint == _spec_fingerprint(spec):
                    continue
                # Either the spec changed, or the name is gone from the file. A
                # removed server is stopped here and not at some later restart:
                # nothing else will ever stop it, and the model has no way to
                # discover that the server it is still calling is no longer
                # configured.
                del self._servers[name]
                self._forget(name)
                srv.stop()
            # 2. A server whose READER died is dead even though its process is
            #    alive: it can never answer again, so it must not be left in
            #    `_servers` to block a worker thread per call (AUDIT F55). The
            #    same is true of one that accepted calls and went mute (09-5).
            for name in [n for n, s in self._servers.items() if s.reader_error]:
                srv = self._servers.pop(name)
                self._failed[name] = srv.reader_error
                self._failed_under[name] = key
                srv.stop()
            # 3. Anything configured that is not up yet: newly added servers,
            #    servers whose spec just changed, and servers whose last failure
            #    was recorded under a different config (a config edit may have
            #    fixed it, and only the file changing is evidence of that).
            for name, spec in cfg.items():
                name = str(name)
                if not isinstance(spec, dict) or name in self._servers:
                    continue
                if self._failed_under.get(name) == key:
                    continue        # already known dead under exactly this config
                srv = McpServer(name, spec)
                try:
                    srv.start()
                    self._servers[name] = srv
                    self._forget(name)
                except Exception as e:
                    srv.stop()
                    self._failed[name] = str(e)[:200]
                    self._failed_under[name] = key

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

    def _route(self, namespaced: str) -> tuple[str, str] | None:
        """Resolve `mcp__<server>__<tool>` to (server, tool).

        Names are matched against the known server names LONGEST FIRST, because
        `split("__", 2)` assumes a server name contains no `__` — and a server
        named `my__server` then routed every call to `my` (09-10). An unknown
        name falls back to the first `__` so the error can name it.
        """
        if not namespaced.startswith("mcp__"):
            return None
        rest = namespaced[len("mcp__"):]
        for name in sorted(set(self._servers) | set(self._failed),
                           key=len, reverse=True):
            prefix = name + "__"
            if rest.startswith(prefix):
                return name, rest[len(prefix):]
        if "__" in rest:
            server, tool = rest.split("__", 1)
            return server, tool
        return None

    def call(self, namespaced: str, args: dict) -> str:
        self._ensure()
        routed = self._route(namespaced)
        if routed is None:
            return f"error: malformed mcp tool name '{namespaced}'"
        server, tool = routed
        srv = self._servers.get(server)
        if srv is None:
            why = self._failed.get(server, "not configured")
            return f"error: mcp server '{server}' is unavailable ({why})"
        if srv.reader_error:
            return (f"error: mcp server '{server}' stopped responding "
                    f"({srv.reader_error}) — it is restarted when mcp.json "
                    "changes")
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
                "wedged": {n: s.timeouts for n, s in self._servers.items()
                           if s.wedged},
                "oversize_frames": {n: s.oversize_frames
                                    for n, s in self._servers.items()
                                    if s.oversize_frames},
                "tools": [t["function"]["name"] for t in self.tool_specs()]}

    def stop_all(self) -> None:
        with self._lock:
            for s in self._servers.values():
                s.stop()
            self._servers = {}
            # Recorded failures go too: this is a deliberate teardown (shutdown,
            # or a test wanting a clean slate), and a manager that kept claiming
            # a server was dead would refuse to start it again on the next turn.
            self._failed, self._failed_under = {}, {}


_manager: McpManager | None = None


def manager() -> McpManager:
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager
