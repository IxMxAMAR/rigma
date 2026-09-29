"""An Agent Client Protocol client for `mcode acp`.

WHY THIS EXISTS. Rigma drives mcode through `mcode exec --output-format
stream-json`, one subprocess per turn. That transport is a PROJECTION: mcode
decides what a turn looked like and prints it. Everything the runtime can do that
is not a turn — the goal control plane, delegation, steering, a queue, plan mode,
switching model or permission policy mid-session, and ANSWERING a permission
prompt — has no `exec` representation at all. ACP is a real protocol over the
child's stdio, and the same runtime exposes all of it there.

This was recorded as "blocked on `mcode login`" for a round. It is not: ACP needs
no credentials, because mcode reaches a model through the BYOK provider Rigma
already registers. That claim came from a probe whose stdin was a file, so mcode
read one request, hit EOF, and exited before answering — and an unanswered request
was misread as a refusal. See `tools/mcode_acp_probe.py`, which now says so.

WHAT MAKES THIS DIFFERENT FROM `exec`, AND FROM Rigma's MCP client:

  1. It is BIDIRECTIONAL. The server sends `session/request_permission` and
     `elicitation/create` and WAITS for our reply. `mcp_client.McpServer` is
     client-to-server only, so its shape cannot be reused; this needs a reader
     that dispatches inbound REQUESTS, not just responses and notifications.
  2. stdin must stay OPEN for the life of the session. A file, or a pipe we close
     early, makes mcode exit — which is exactly the bug above.
  3. Capabilities are DECLARED, not discovered. mcode offers the questionnaire
     only if we send `clientCapabilities.elicitation.form`, and plan review only
     if we send `clientCapabilities.plan`. Declaring them is what turns a silent
     fallback into an interactive turn.

DELIBERATELY NOT DONE HERE: nothing in this module starts a model. It speaks the
protocol and maps what comes back. The transport is exercised against a scripted
child in the tests, which is the only way to test it under the user's standing
order that no model be loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any, Callable

# One JSON-RPC frame is one line. The bound exists for the same reason
# `mcp_server._FRAME_MAX` does: a reader that holds an arbitrarily long line whole
# can be made to allocate without limit by a peer that never sends a newline.
_FRAME_MAX = 8_000_000

# The protocol version this client speaks. Measured: mcode answers `initialize`
# with `protocolVersion: 1` and accepts 1.
PROTOCOL_VERSION = 1

# mcode's own extension surface, as advertised in
# `_meta["minimax-code/extensions"]` on `initialize`. Recorded rather than
# hardcoded into calls so a version that drops or renames one is visible instead
# of producing a confusing "Method not found" at the point of use.
# The standard ACP session methods Rigma uses, kept beside the extension list so the
# drift guard in `tests/test_acp_control.py` can cover both. (No such guard existed when
# this sentence was written — it described an intention, not a fact. It exists now, and it
# fails when a table and the code that uses it disagree.)
#
# `session/load` and `session/resume` take
# `{sessionId}` and answer `{sessionId}`, measured from the handlers:
#   onRequest(ee.agent.session.resume, async z => s(z.params.sessionId, ...))
#   onRequest(ee.agent.session.load,   async z => s(z.params.sessionId, ...))
STANDARD_METHODS = (
    "session/new",
    "session/load",
    "session/resume",
    "session/list",
    "session/set_mode",
    "session/set_config_option",
    "session/prompt",
)

EXTENSION_METHODS = (
    "session/activate",
    "mcode/session/activate",
    "mcode/session/steer",
    "mcode/session/queue/list",
    "mcode/session/queue/enqueue",
    "mcode/session/queue/update",
    "mcode/session/queue/delete",
    "mcode/session/queue/steer",
    "mcode/session/goal/get",
    "mcode/session/goal/create",
    "mcode/session/goal/patch",
    "mcode/session/goal/clear",
    "mcode/session/delegation/get",
    "mcode/session/delegation/stop",
)

EXTENSION_NOTIFICATIONS = (
    "mcode/session/current_session_update",
    "mcode/session/queue_update",
    "mcode/session/goal_update",
    "mcode/session/delegation_update",
)

# The two vocabularies are NOT the same set and share no member. `exec` takes
# `smart|full|off`; ACP's `permissionMode` configOption takes these. Passing one to
# the other is rejected outright — measured: `Invalid params: Unsupported
# permission mode: full` — so a translation table is required rather than a
# pass-through.
PERMISSION_MODES = ("default", "auto", "bypassPermissions")


class AcpError(RuntimeError):
    """A JSON-RPC error the server returned, or a transport failure."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class AcpUnavailable(AcpError):
    """The server did not answer at all.

    Kept distinct from `AcpError` because these are DIFFERENT FACTS: "refused" and
    "never asked" must never be conflated again. The false `mcode login` blocker
    was exactly that conflation.
    """


def _noop(_event: dict) -> None:
    return None


class AcpClient:
    """A live `mcode acp` child, spoken to over newline-delimited JSON-RPC.

    Threading model, and why it is this shape: one reader thread owns stdout and
    is the ONLY thing that reads it. It dispatches each frame by kind —
    a response resolves a pending `Future`-like slot, a server REQUEST is handed
    to a callback that must return a result, and a notification goes to the event
    sink. `send` may be called from any thread; a lock serialises writes so two
    frames can never interleave.
    """

    def __init__(self, argv: list[str], *, cwd: str = "", env: dict | None = None,
                 on_event: Callable[[dict], None] | None = None,
                 on_request: Callable[[str, dict], Any] | None = None,
                 default_timeout: float = 120.0):
        self.argv = list(argv)
        self.cwd = cwd or None
        self._env = env
        self._on_event = on_event or _noop
        self._on_request = on_request
        self.default_timeout = default_timeout

        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self.stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None

        # Filled in by `initialize` / `session_new`.
        self.server_info: dict = {}
        self.capabilities: dict = {}
        self.extensions: dict = {}
        self.session_id: str = ""
        self.modes: dict = {}
        self.config_options: list[dict] = []

    # ---- process lifecycle -------------------------------------------------

    def start(self) -> None:
        """Spawn the child. stdin is a PIPE and is NEVER closed until `stop`.

        Closing it early — or pointing it at a file — makes mcode read what it can,
        see EOF, and exit before answering. That is the failure this module exists
        partly to not repeat, so it is stated where the pipe is created.
        """
        env = dict(os.environ if self._env is None else self._env)
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        self._reader = threading.Thread(target=self._read_loop, name="acp-reader",
                                        daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop,
                                               name="acp-stderr", daemon=True)
        self._stderr_thread.start()

    def _fail_all_pending(self, message: str) -> None:
        """Release every waiter with an error.

        THE EVENT MUST BE SIGNALLED, not just the slot filled. An earlier version
        set `slot["error"]` and dropped the slot without setting the event, so a
        caller blocked in `request()` never woke: it held its slot, saw no event,
        and waited out its FULL timeout — or forever, if the timeout was long. The
        thread also stayed alive, which is how this was caught.
        """
        with self._pending_lock:
            waiters = list(self._pending.values())
            self._pending.clear()
        for slot in waiters:
            slot["error"] = AcpUnavailable(message)
            slot["event"].set()

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the child. Safe to call twice."""
        self._closed.set()
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            # CLOSING STDIN IS THE GRACEFUL SHUTDOWN, and it has to be given a moment
            # before the kill. mcode reads EOF, finishes, and exits — and anything it
            # does on the way out (flushing a session store, running its own atexit
            # hooks) only happens if it is allowed to. Terminating in the same breath
            # makes that a race the child usually loses: the signal arrives first and
            # the flush never runs.
            #
            # This is not hypothetical. The test double persists its session through
            # `atexit`, so the race decides whether a control operation's result is
            # visible to the next process — and it made a `--record` assertion vacuous
            # in the same way. A short grace period costs nothing when the child exits
            # promptly, which is the normal case.
            try:
                proc.wait(timeout=min(2.0, timeout))
            except Exception:
                proc.terminate()
                proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # Any waiter still blocked must learn the transport is gone rather than
        # wait out its full timeout.
        self._fail_all_pending("the mcode acp process ended")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---- transport ---------------------------------------------------------

    def _stderr_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                # Bounded: a chatty child must not grow this without limit.
                self.stderr_tail.append(line.rstrip("\n"))
                if len(self.stderr_tail) > 200:
                    del self.stderr_tail[:100]
        except Exception:
            pass

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if self._closed.is_set():
                    break
                if len(line) > _FRAME_MAX:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except ValueError:
                    # Not JSON: a diagnostic on the wrong stream. Keep it, since it
                    # is the only evidence if the child then dies.
                    self.stderr_tail.append(line[:400])
                    continue
                try:
                    self._dispatch(frame)
                except Exception:
                    pass        # one bad frame must not kill the reader
        except Exception:
            pass
        finally:
            # The child is gone. Fail everything still waiting, rather than let
            # each caller discover it by timing out.
            self._fail_all_pending("the mcode acp process closed its output")

    def _dispatch(self, frame: dict) -> None:
        has_id = "id" in frame and frame.get("id") is not None
        method = frame.get("method")

        if method is not None and has_id:
            self._handle_request(frame)
            return
        if method is not None:
            self._handle_notification(frame)
            return
        if has_id:
            self._handle_response(frame)

    def _handle_response(self, frame: dict) -> None:
        with self._pending_lock:
            slot = self._pending.pop(frame.get("id"), None)
        if slot is None:
            return
        if "error" in frame:
            err = frame.get("error") or {}
            slot["error"] = AcpError(
                str(err.get("message") or "mcode acp returned an error"),
                code=err.get("code"), data=err.get("data"))
        else:
            slot["result"] = frame.get("result")
        slot["event"].set()

    def _handle_notification(self, frame: dict) -> None:
        try:
            self._on_event({"method": frame.get("method"),
                            "params": frame.get("params") or {}})
        except Exception:
            pass

    def _decline(self, method: str, params: dict) -> Any:
        """The protocol's own refusal for a request this client cannot serve.

        WHY `None` WAS WRONG. A JSON-RPC `result` of `null` is a valid envelope but not a
        valid ANSWER. `session/request_permission` expects an `outcome` envelope and
        `elicitation/create` expects an `action`; mcode reads an unrecognised shape as a
        malformed reply rather than as a refusal, and the failure surfaces as a generic
        502 on a short-lived process. The module already knew the correct shapes —
        `answer_permission(allow=None)` and `answer_elicitation(accepted=False)` — they
        were simply not wired to this path.
        """
        if method == "session/request_permission":
            return answer_permission(params, allow=None)
        if method == "elicitation/create":
            return answer_elicitation(params, accepted=False)
        return None

    def _handle_request(self, frame: dict) -> None:
        """A server-initiated request. We MUST answer, or the turn stalls."""
        method = str(frame.get("method") or "")
        params = frame.get("params") or {}
        result: Any = None
        error: dict | None = None
        if self._on_request is None:
            # No handler is still a REAL answer, not silence: an unanswered request would
            # leave mcode waiting forever. It is now the SHAPE the protocol defines for a
            # refusal rather than a bare `null`, which mcode could only read as malformed.
            result = self._decline(method, params)
        else:
            try:
                result = self._on_request(method, params)
            except Exception as exc:
                error = {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}
        reply: dict = {"jsonrpc": "2.0", "id": frame.get("id")}
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        try:
            self._write(reply)
        except Exception:
            pass

    def _write(self, obj: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise AcpUnavailable("the mcode acp process is not running")
        body = json.dumps(obj, separators=(",", ":"))
        # THE OUTBOUND HALF OF `_FRAME_MAX`. That constant bounds what the server may send
        # US, and nothing bounded what we send it — so a request carrying a very large
        # `text` went to the child's pipe whole. The UI cannot produce such a value, but
        # the route accepts JSON from any client, and an unbounded write to a child's
        # stdin is the kind of thing that surfaces as a hang rather than as an error.
        if len(body) > _FRAME_MAX:
            raise AcpError(
                f"the request is too large to send ({len(body)} bytes, limit "
                f"{_FRAME_MAX})")
        with self._write_lock:
            proc.stdin.write(body + "\n")
            proc.stdin.flush()

    def _alloc_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def request(self, method: str, params: dict | None = None,
                timeout: float | None = None) -> Any:
        """Send a request and wait for its reply.

        Raises `AcpUnavailable` when no reply arrives at all, and `AcpError` when
        the server answered with an error. The distinction is load-bearing: the
        false `mcode login` blocker was a timeout read as a refusal.
        """
        if self._closed.is_set():
            raise AcpUnavailable("the mcode acp client is stopped")
        rid = self._alloc_id()
        slot = {"event": threading.Event(), "result": None, "error": None}
        with self._pending_lock:
            self._pending[rid] = slot
        frame = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            frame["params"] = params
        try:
            self._write(frame)
        except Exception:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise
        if not slot["event"].wait(timeout if timeout is not None else self.default_timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise AcpUnavailable(f"no reply to {method!r} within the timeout")
        if slot["error"] is not None:
            raise slot["error"]
        return slot["result"]

    def notify(self, method: str, params: dict | None = None) -> None:
        frame = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self._write(frame)

    # ---- protocol: the parts Rigma uses -----------------------------------

    def initialize(self, *, declare_elicitation: bool = True,
                   declare_plan: bool = True, declare_extensions: bool = True,
                   timeout: float | None = None) -> dict:
        """Handshake, DECLARING the capabilities that unlock interaction.

        This is the whole reason a DSH-shaped client cannot do this. mcode offers
        the questionnaire only when we say we can render one, and the plan review
        only when we say we understand plans. Declaring a capability we cannot
        actually serve would be worse than not declaring it, so both flags are
        parameters rather than constants — the caller says what Rigma can really do.

        THE THIRD DECLARATION IS EASY TO MISS AND SILENTLY LOSES EVERYTHING.
        mcode gates ALL FOUR of its extension notifications — `goal_update`,
        `queue_update`, `delegation_update` and `current_session_update` — on the
        CLIENT declaring

            clientCapabilities._meta["minimax-code/extensions"]
                === true, or {version: 1, notifications: true}

        (`po()` in `run-acp-command-JPZMIXGP.js`; `Rd` is 1). It is not symmetric:
        the server ADVERTISES its extension list in the `initialize` RESPONSE, so a
        client can read that list, call every extension method successfully, and
        still never receive a single update — the calls all work and the UI never
        moves. A goal created by `goal/create` answers correctly and its
        `goal_update` is dropped, which looks like "mcode does not notify" rather
        than "we never said we could hear it".
        """
        # NO `fs`. This used to declare `{"readTextFile": True, "writeTextFile": True}`
        # while nothing in this package handled `fs/read_text_file` or
        # `fs/write_text_file` — so Rigma told mcode it could read and write files on
        # request and then refused every request. mcode PLANS around a declared
        # capability, so the claim was worse than silence.
        #
        # Left unimplemented on purpose: it would give the agent file access through a
        # second path beside the sandbox policy that already governs its tools.
        client_capabilities: dict = {
            "terminal": False,
        }
        if declare_elicitation:
            # `form` is the shape mcode checks for, by name.
            client_capabilities["elicitation"] = {"form": {}}
        if declare_plan:
            # An OBJECT in the schema, tested for truthiness — so `{}` is right.
            client_capabilities["plan"] = {}
        if declare_extensions:
            client_capabilities["_meta"] = {
                "minimax-code/extensions": {"version": 1, "notifications": True}
            }
        res = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": client_capabilities,
        }, timeout=timeout)
        res = res or {}
        self.server_info = res.get("agentInfo") or {}
        self.capabilities = res.get("agentCapabilities") or {}
        meta = res.get("_meta") or {}
        self.extensions = meta.get("minimax-code/extensions") or {}
        return res

    def session_new(self, cwd: str, mcp_servers: list | None = None,
                    timeout: float | None = None) -> dict:
        """Open a session and record what the server says it can do.

        `modes` and `configOptions` come back HERE, not from `initialize`, so a
        client that only handshakes cannot know that plan mode exists or what the
        permission vocabulary is.
        """
        res = self.request("session/new",
                           {"cwd": cwd, "mcpServers": mcp_servers or []},
                           timeout=timeout) or {}
        self.session_id = str(res.get("sessionId") or "")
        self.modes = res.get("modes") or {}
        self.config_options = res.get("configOptions") or []
        return res

    def session_resume(self, session_id: str, cwd: str = "",
                       mcp_servers: list | None = None,
                       timeout: float | None = None) -> dict:
        """Reattach to a session mcode already has, so a turn CONTINUES it.

        This is what makes the chat a conversation rather than a series of unrelated
        questions, and it is the ACP equivalent of `exec`'s `--session <id>`. The
        handler is `onRequest(ee.agent.session.resume, async z =>
        s(z.params.sessionId, ...))`, so the parameter is `sessionId` and the answer
        carries `sessionId` back.

        `session/load` takes the same parameter and is registered to a near-identical
        handler; `resume` is used because it is the one whose name says what is
        wanted here. `session_load` is kept as an alias for a version that drops one.
        """
        body: dict = {"sessionId": session_id}
        if cwd:
            body["cwd"] = cwd
        if mcp_servers is not None:
            body["mcpServers"] = mcp_servers
        res = self.request("session/resume", body, timeout=timeout) or {}
        # The server may answer with a DIFFERENT id than the one asked for. Record
        # what it said, not what was requested — the same rule `exec`'s `--model`
        # follows, where Rigma uses "the id mcode actually gave us".
        self.session_id = str(res.get("sessionId") or session_id)
        self.modes = res.get("modes") or self.modes
        self.config_options = res.get("configOptions") or self.config_options
        return res

    def session_load(self, session_id: str, cwd: str = "",
                     mcp_servers: list | None = None,
                     timeout: float | None = None) -> dict:
        """The other spelling of reattaching. Same parameter, same handler shape."""
        body: dict = {"sessionId": session_id}
        if cwd:
            body["cwd"] = cwd
        if mcp_servers is not None:
            body["mcpServers"] = mcp_servers
        res = self.request("session/load", body, timeout=timeout) or {}
        self.session_id = str(res.get("sessionId") or session_id)
        self.modes = res.get("modes") or self.modes
        self.config_options = res.get("configOptions") or self.config_options
        return res

    def session_list(self, timeout: float | None = None) -> Any:
        """Every session mcode is holding, so Rigma can offer to continue one."""
        return self.request("session/list", {}, timeout=timeout)

    def available_modes(self) -> list[dict]:
        return list(self.modes.get("availableModes") or [])

    def current_mode(self) -> str:
        return str(self.modes.get("currentModeId") or "")

    def set_mode(self, mode_id: str, timeout: float | None = None) -> Any:
        return self.request("session/set_mode",
                            {"sessionId": self.session_id, "modeId": mode_id},
                            timeout=timeout)

    def config_option(self, option_id: str) -> dict | None:
        for opt in self.config_options:
            if opt.get("id") == option_id:
                return opt
        return None

    def set_config_option(self, option_id: str, value: Any,
                          timeout: float | None = None) -> Any:
        """Set a configOption — this is how model and permission policy change.

        `session/set_config_option` is the protocol's own operation for it, which is
        why an ACP client can switch models mid-session and an `exec` driver cannot.
        """
        return self.request("session/set_config_option", {
            "sessionId": self.session_id,
            "configId": option_id,
            "value": value,
        }, timeout=timeout)

    def prompt(self, text: str, timeout: float | None = None) -> Any:
        """Send one prompt. Streaming arrives through `on_event` as it happens."""
        return self.request("session/prompt", {
            "sessionId": self.session_id,
            "prompt": [{"type": "text", "text": text}],
        }, timeout=timeout)

    def cancel(self) -> None:
        """Ask the server to stop the current turn.

        A NOTIFICATION, not a request: there is nothing to wait for, and `exec`
        has no equivalent at all — its only stop is killing the process.
        """
        self.notify("session/cancel", {"sessionId": self.session_id})

    # ---- protocol: mcode's extension surface ------------------------------

    def goal_get(self, timeout: float | None = None) -> Any:
        return self.request("mcode/session/goal/get",
                            {"sessionId": self.session_id}, timeout=timeout)

    def goal_create(self, params: dict, timeout: float | None = None) -> Any:
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/goal/create", body, timeout=timeout)

    def goal_patch(self, params: dict, timeout: float | None = None) -> Any:
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/goal/patch", body, timeout=timeout)

    def goal_clear(self, timeout: float | None = None) -> Any:
        return self.request("mcode/session/goal/clear",
                            {"sessionId": self.session_id}, timeout=timeout)

    def queue_list(self, timeout: float | None = None) -> Any:
        return self.request("mcode/session/queue/list",
                            {"sessionId": self.session_id}, timeout=timeout)

    def queue_enqueue(self, params: dict, timeout: float | None = None) -> Any:
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/queue/enqueue", body, timeout=timeout)

    def queue_update(self, params: dict, timeout: float | None = None) -> Any:
        """Edit a queued message that has not run yet. Takes `{itemId, text}`."""
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/queue/update", body, timeout=timeout)

    def queue_delete(self, params: dict, timeout: float | None = None) -> Any:
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/queue/delete", body, timeout=timeout)

    def queue_steer(self, params: dict, timeout: float | None = None) -> Any:
        """Promote an ALREADY-QUEUED item into the active turn.

        A third thing, distinct from both neighbours: `enqueue` appends for a later
        turn, `steer` injects new text into the running turn, and this moves an
        existing queued item into it. Takes `{itemId}`; answers `{queueItemId, turnId}`.
        """
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/queue/steer", body, timeout=timeout)

    def activate(self, timeout: float | None = None) -> Any:
        """Mark this session as the active one.

        `session/activate` and `mcode/session/activate` are registered to the SAME
        closure with the same validator and the same response, so the two names are
        interchangeable; the only observable effect is a
        `mcode/session/current_session_update` notification. Called through the
        namespaced spelling, and `session_activate` is kept as an alias so a
        version that drops one name still works.
        """
        return self.request("mcode/session/activate",
                            {"sessionId": self.session_id}, timeout=timeout)

    def session_activate(self, timeout: float | None = None) -> Any:
        """The un-namespaced spelling of `activate`. Same handler, same result."""
        return self.request("session/activate",
                            {"sessionId": self.session_id}, timeout=timeout)

    def steer(self, params: dict, timeout: float | None = None) -> Any:
        """Inject text into the Turn that is ALREADY RUNNING.

        Preconditions the server enforces, which are worth knowing before calling:
        an admitted, active prompt turn must exist, and the turn id must match — so
        this cannot be used to start work, only to redirect it. Answers
        `{turnId, mode}` with mode `steered` (or `duplicate` if it deduplicated).
        """
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/steer", body, timeout=timeout)

    def delegation_get(self, timeout: float | None = None) -> Any:
        return self.request("mcode/session/delegation/get",
                            {"sessionId": self.session_id}, timeout=timeout)

    def delegation_stop(self, params: dict, timeout: float | None = None) -> Any:
        body = {"sessionId": self.session_id}
        body.update(params or {})
        return self.request("mcode/session/delegation/stop", body, timeout=timeout)

    def supports(self, method: str) -> bool:
        """Did the server ADVERTISE this method?

        Asked before use, so a version that drops one produces a sentence rather
        than a raw "Method not found" from the middle of a turn.

        NO FALLBACK TO THE CONSTANT. An earlier version fell back to
        `EXTENSION_METHODS` when the server advertised nothing, which made this
        answer True for a server that had advertised nothing at all — the same
        shape of error as the false blocker: a claim about the server derived from
        our own file rather than from what the server said. `EXTENSION_METHODS`
        documents the version this client was written against; it is not evidence
        about a server that stayed silent.
        """
        if not self.extensions:
            return False
        advertised = self.extensions.get("methods")
        if not isinstance(advertised, list):
            return False
        return method in advertised


def answer_permission(params: dict, *, allow: bool | None,
                      allow_always: bool = False) -> dict:
    """Build the reply to `session/request_permission`.

    THE SHAPE MATTERS. mcode sends a list of `options` and expects the chosen
    `optionId` back inside an `outcome` envelope; `selected` with an optionId, or
    `cancelled`. Measured option ids are `allow-once`, `allow-always` and `deny`
    (mcode) — Rigma must echo back one the SERVER offered rather than invent one,
    because an unrecognised id is read as a refusal.

    `allow=None` means "decline to answer", which is the honest reply when Rigma
    has no policy for this request: it must not be reported as a denial.
    """
    if allow is None:
        return {"outcome": {"outcome": "cancelled"}}
    offered = {str(o.get("optionId")) for o in (params.get("options") or [])
               if isinstance(o, dict)}
    if allow:
        wanted = "allow-always" if allow_always else "allow-once"
    else:
        wanted = "deny"
    if offered and wanted not in offered:
        # Fall back to whatever the server actually offered rather than sending an
        # id it does not know.
        for alt in (("allow-once", "allow-always") if allow else ("deny", "reject-once")):
            if alt in offered:
                wanted = alt
                break
        else:
            return {"outcome": {"outcome": "cancelled"}}
    return {"outcome": {"outcome": "selected", "optionId": wanted}}


def answer_elicitation(params: dict, *, accepted: bool,
                       content: dict | None = None) -> dict:
    """Build the reply to `elicitation/create`.

    This is the channel that makes `ask_user_question` possible at all. `accepted`
    with `content` answers the question; declining is a REAL answer that makes
    mcode take its non-interactive fallback, which is what Rigma gets today by
    declaring nothing.
    """
    if not accepted:
        return {"action": "decline"}
    return {"action": "accept", "content": dict(content or {})}


# --- translating ACP into Rigma's own vocabulary -----------------------------
#
# ACP says the same things in a different language. These maps are the whole
# translation, and each is a place where guessing would be expensive.
#
# TOOL STATUS is a three-valued fact here, matching `exec`'s treatment: `pending`
# and `in_progress` are IN FLIGHT, and in-flight is not success. Returning True for
# them would put a tick on a call that has not finished — the same error the
# `_ok_of` numeric enum was carefully kept away from.
_ACP_TOOL_OK = {
    "completed": True,
    "failed": False,
    "pending": None,
    "in_progress": None,
}

# The plan-mode ids. `current_mode_update` carries an ID, not a boolean, and the UI
# wants a boolean — so the id has to be interpreted, and interpreting it wrongly
# would light the plan indicator for a session that is not planning.
_ACP_PLAN_MODES = ("plan",)


def acp_tool_ok(status: object) -> bool | None:
    """Whether an ACP tool call succeeded, or None while it is still in flight."""
    if status is None:
        return None
    return _ACP_TOOL_OK.get(str(status).strip().lower())


def _acp_content_text(content: object) -> str:
    """Flatten an ACP content block or block list to text.

    ACP content is a UNION — a block may be text, an image, a resource link or an
    embedded resource — so a reader that assumes a string silently drops every
    non-text block. That is the same class of defect as the `exec` projector's
    unnamed blocks, and it is named here rather than dropped: a block with no text
    becomes a short description of what it was instead of nothing.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        ctype = str(content.get("type") or "")
        if ctype == "image":
            return "[image]"
        if ctype == "audio":
            return "[audio]"
        if ctype == "resource_link":
            return f"[link {content.get('name') or content.get('uri') or ''}]"
        if ctype == "resource":
            return "[embedded resource]"
        if "text" in content:
            return str(content.get("text") or "")
        return f"[{ctype or 'content'}]"
    if isinstance(content, (list, tuple)):
        return "".join(p for p in (_acp_content_text(x) for x in content) if p)
    return str(content)


def map_acp_update(notification: dict) -> list:
    """Translate one ACP notification into Rigma `TurnEvent`s.

    Returns a LIST because one ACP frame can be several Rigma facts: a `tool_call`
    whose status is already terminal carries both the call and its outcome, and
    splitting it into two events is what lets the transcript render a chip and then
    its result the way every other backend's do.

    Unknown `sessionUpdate` variants return an EMPTY list rather than a notice. The
    protocol's union has 15 members and will grow; a future variant must not become a
    wrong row, and it must not become noise either — the honest place for "this build
    does not render that yet" is the capability menu, not every turn.

    (This said 13. VERIFIED against `@agentclientprotocol/sdk@1.4.0` as resolved in the
    DSH checkout: the union declares 15, adding `compaction_update` and
    `compaction_summary_chunk`. The count is a wire fact that moves with the SDK, so it
    is stated as of a version rather than as a constant, and
    `test_the_docstring_does_not_claim_a_union_size_it_cannot_know` fails if a bare
    count is reintroduced.)
    """
    from .harness import TurnEvent

    method = str(notification.get("method") or "")
    params = notification.get("params") or {}
    out: list = []

    # ---- mcode's extension notifications ---------------------------------
    if method == "mcode/session/goal_update":
        goal = params.get("goal")
        if goal is None:
            # The clear tombstone. `chat/goal.ts` reads `{cleared: true}` as an
            # instruction to blank the panel, so this must not be sent as
            # "unrecognised payload" — that would leave a cleared goal on screen.
            out.append(TurnEvent("state", event="goal",
                                 data={"cleared": True,
                                       "goalId": params.get("goalId")}))
        else:
            out.append(TurnEvent("state", event="goal", data=goal))
        return out
    if method == "mcode/session/queue_update":
        items = params.get("items")
        out.append(TurnEvent("state", event="acp_queue",
                             data={"items": items if isinstance(items, list) else []}))
        return out
    if method == "mcode/session/delegation_update":
        snap = params.get("snapshot")
        out.append(TurnEvent("state", event="acp_delegation",
                             data=snap if isinstance(snap, dict) else {}))
        return out
    if method == "mcode/session/current_session_update":
        out.append(TurnEvent("state", event="acp_current_session",
                             data={"sessionId": params.get("sessionId")}))
        return out
    if method != "session/update":
        return out

    # ---- the standard session/update union --------------------------------
    update = params.get("update") or {}
    kind = str(update.get("sessionUpdate") or "")

    if kind in ("compaction_update", "compaction_summary_chunk"):
        # A STANDARD variant as of `@agentclientprotocol/sdk@1.4.0` (VERIFIED in the DSH
        # checkout's resolved copy: the union declares 15 members, and these are the two
        # that were missing from the 13 this mapper used to know).
        #
        # Handled rather than dropped because the OTHER adapter already renders this fact:
        # the DSH path maps its `compaction/` lifecycle onto one `compaction` SSE event,
        # so the identical event would have rendered on one backend and vanished on the
        # other. The payload is passed through WHOLE, matching the DSH arm's reason —
        # `summary` is a ContentBlock list and reshaping it here would stop the UI folding
        # it.
        #
        # VERIFIED: mcode 0.5.4's own bundle mentions neither string, so this is
        # forward-compatibility rather than a path a turn takes today.
        out.append(TurnEvent("state", event="compaction/" + kind, data=update))
        return out

    if kind in ("agent_message_chunk", "user_message_chunk"):
        text = _acp_content_text(update.get("content"))
        if text:
            out.append(TurnEvent("text", text))
        return out

    if kind == "agent_thought_chunk":
        text = _acp_content_text(update.get("content"))
        if text:
            out.append(TurnEvent("thinking", text))
        return out

    if kind in ("tool_call", "tool_call_update"):
        call_id = str(update.get("toolCallId") or "")
        name = str(update.get("name") or update.get("title") or "tool")
        ok = acp_tool_ok(update.get("status"))
        args = update.get("rawInput")
        if not isinstance(args, dict):
            args = {}
        text = _acp_content_text(update.get("content"))
        raw_out = update.get("rawOutput")
        if raw_out is not None and not text:
            text = _acp_content_text(raw_out)
        if ok is None:
            # Still running. Announce the chip once there is anything to announce:
            # an id alone is not worth drawing.
            if kind == "tool_call" and (args or text):
                out.append(TurnEvent("tool", name=name, args=args, data={"id": call_id}))
            elif kind == "tool_call_update" and (args or text):
                out.append(TurnEvent("tool", name=name, args=args, data={"id": call_id}))
            return out
        # A terminal status is the RESULT. If nothing was announced yet, the call
        # goes first so the transcript reads call-then-outcome.
        out.append(TurnEvent("tool", name=name, args=args, data={"id": call_id}))
        out.append(TurnEvent("tool_result", text=text, name=name, ok=ok,
                             data={"id": call_id}))
        return out

    if kind == "plan_update":
        plan = update.get("plan") or {}
        out.append(TurnEvent("state", event="acp_plan", data=plan))
        return out
    if kind == "plan_removed":
        out.append(TurnEvent("state", event="acp_plan",
                             data={"planId": update.get("planId"), "removed": True}))
        return out

    if kind == "current_mode_update":
        mode = str(update.get("currentModeId") or "")
        # Reported through the SAME name DSH uses, because it is the same fact and
        # the store already has one arm for it. `active` is derived from the id:
        # the UI wants a boolean and the wire carries an id.
        out.append(TurnEvent("state", event="plan/mode",
                             data={"active": mode in _ACP_PLAN_MODES,
                                   "modeId": mode}))
        return out

    if kind == "config_option_update":
        opts = update.get("configOptions")
        out.append(TurnEvent("state", event="acp_config",
                             data={"configOptions": opts if isinstance(opts, list) else []}))
        return out

    if kind == "available_commands_update":
        cmds = update.get("availableCommands")
        out.append(TurnEvent("state", event="acp_commands",
                             data={"commands": cmds if isinstance(cmds, list) else []}))
        return out

    if kind == "session_info_update":
        data: dict = {}
        if update.get("title") is not None:
            data["title"] = str(update.get("title"))
        if update.get("updatedAt") is not None:
            data["updatedAt"] = update.get("updatedAt")
        if data:
            # `session/title` is the BACKEND-side name `serve.py` already routes
            # (`_ev == "session/title"` -> SSE `session_title`). An earlier version
            # of this mapper emitted `session_title` — the SSE name, one hop too
            # early — and the event was dropped silently. The cross-layer drift
            # guard in tests/test_dsh_capabilities.py caught it, which is exactly
            # why that guard reads the names from source instead of restating them.
            out.append(TurnEvent("state", event="session/title", data=data))
        return out

    if kind == "usage_update":
        # `used` and `size` are ACP's names for what Rigma's usage line already
        # renders from `exec`. Translated rather than passed through, because the
        # renderer keys on its own names and a second vocabulary would show nothing.
        data = {}
        if isinstance(update.get("used"), (int, float)):
            data["usedTokens"] = int(update["used"])
        if isinstance(update.get("size"), (int, float)):
            data["contextWindowTokens"] = int(update["size"])
        cost = update.get("cost")
        if isinstance(cost, dict) and cost.get("amount") is not None:
            data["cost"] = cost
        if data:
            out.append(TurnEvent("state", event="usage", data=data))
        return out

    if kind == "plan":
        entries = update.get("entries")
        if isinstance(entries, list):
            # ACP's `plan` variant is the same fact as a todo list: a set of items
            # with a status each. Rendered through the todos channel so it lands in
            # the panel that already exists rather than needing a second one.
            out.append(TurnEvent("state", event="todos", data={"todos": [
                {"content": _acp_content_text(e.get("content")),
                 "status": str(e.get("status") or "pending")}
                for e in entries if isinstance(e, dict)]}))
        return out

    return out


def probe_surface(argv: list[str], *, cwd: str = "", env: dict | None = None,
                  timeout: float = 30.0) -> dict:
    """Handshake, open a session, and report what the server offers.

    Used by the capability inventory so the UI can state mcode's ACP surface from
    MEASUREMENT rather than from a constant in this file. Loads no model: it
    performs the handshake and reads `session/new`'s advertised modes and
    configOptions, then stops.

    Returns `{"ok": bool, "server": {...}, "modes": [...], "configOptions": [...],
    "extensions": {...}, "error": str}` — never raises, because a probe that
    raises takes down whatever is drawing the menu.
    """
    out: dict = {"ok": False, "server": {}, "modes": [], "configOptions": [],
                 "extensions": {}, "error": ""}
    client = AcpClient(argv, cwd=cwd, env=env, default_timeout=timeout)
    try:
        client.start()
        client.initialize(timeout=timeout)
        out["server"] = dict(client.server_info)
        out["extensions"] = dict(client.extensions)
        res = client.session_new(cwd or os.getcwd(), timeout=timeout)
        out["modes"] = list((res.get("modes") or {}).get("availableModes") or [])
        out["configOptions"] = [
            {"id": o.get("id"), "currentValue": o.get("currentValue"),
             "options": [x.get("value") for x in (o.get("options") or [])
                         if isinstance(x, dict)]}
            for o in (res.get("configOptions") or []) if isinstance(o, dict)
        ]
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        client.stop()
    return out


# --- driving a chat turn over ACP ---------------------------------------------
#
# WHY THIS IS THE POINT OF THE WHOLE MODULE. Everything above is transport. A
# control plane nobody can reach during a turn is still unreachable, and the
# features that motivated the client — answering a permission prompt, plan mode,
# steering, the queue — only become LIVE when a chat turn runs here instead of
# through `mcode exec`.
#
# WHAT IT DELIBERATELY IS NOT. It is not a replacement for `drive_turn`. Switching
# the default transport is a separate decision with its own risk, so this is a
# parallel path the caller opts into. `exec` stays the default until this has been
# exercised for real, which the standing order currently forbids.
#
# THE PERMISSION POLICY IS THE INTERESTING PART. `exec` has no way to ask, so
# `smart` could dead-end a chat permanently (R5-MCODE-DEADEND). Here the server
# asks and WAITS, so the answer has to come from somewhere. It comes from the mode
# Rigma already has, translated — and the translation is NOT a pass-through, because
# the two vocabularies share no member (`exec`: smart|full|off; ACP:
# default|auto|bypassPermissions). Every decision is reported as a `state` event so
# the transcript shows what was granted rather than silently proceeding, which is the
# same discipline the governance panel already follows.

# `exec`'s mode -> ACP's mode. The two sets share NO member, so this is a
# translation rather than a rename. Measured: passing `full` to ACP is rejected with
# `Unsupported permission mode: full`.
_EXEC_TO_ACP_PERMISSION = {
    "full": "bypassPermissions",
    "smart": "auto",
    "off": "default",
}


def _acp_approval_outcome(outcome: dict, option_id: str) -> str:
    """Translate an ACP permission outcome into the vocabulary Rigma already reads.

    COPIED FROM DSH'S OWN BRIDGE, deliberately, because the two transports must
    produce the SAME words or the governance panel renders one of them as a bare
    identifier. `packages/acp/acp/src/index.ts:171` is the whole rule:

        if (outcome.outcome === 'cancelled') return 'cancelled'
        return outcome.optionId === 'allow-once' ? 'allowed-once' : 'rejected'

    `ApprovalOutcome` there is `'allowed-once' | 'rejected' | 'cancelled' |
    'unavailable'`, and `outcomeTone`/`outcomeLabel` in `chat/governance.ts` know
    exactly those four. An earlier version of this function returned
    "allowed"/"denied", which is a THIRD vocabulary: `outcomeTone` would have
    coloured a grant grey and `outcomeLabel` would have shown the raw word, so a
    granted permission would have looked like an unexplained neutral event.

    Note the strictness: only `allow-once` is a grant. `allow-always` is NOT, which
    matches DSH — a durable grant is not something either transport infers from a
    one-shot answer.
    """
    if str(outcome.get("outcome") or "") == "cancelled":
        return "cancelled"
    return "allowed-once" if option_id == "allow-once" else "rejected"


def acp_permission_mode(permission: str) -> str:
    """Translate one of `exec`'s permission words into ACP's vocabulary.

    Falls back to `auto` rather than to `bypassPermissions`: an unrecognised mode
    must not silently become the most permissive one. `auto` is the middle — it
    answers, and every answer is reported.
    """
    return _EXEC_TO_ACP_PERMISSION.get(str(permission or "").strip().lower(), "auto")



# --- driving the control plane OUTSIDE a turn ---------------------------------
#
# WHY THIS EXISTS. Every method below was defined in `AcpClient` and NONE of them was
# called by anything: the control plane was readable and not invocable. The queue, the
# goal, steering and the delegation tree all arrived as NOTIFICATIONS, so the UI could
# draw them and a user could not touch them. A control plane you can only watch is not a
# control plane, and "carry the goal control plane" was not done until a caller existed.
#
# WHAT IT IS. A short-lived client that RESUMES an existing session, performs ONE
# operation and stops. Verified against `tests/fake_acp_server.py`, which is a real
# subprocess over real pipes: `session/resume` answers the same sessionId, and the goal
# operations mutate state that a later `goal/get` reads back.
#
# WHY NOT DO IT ON THE TURN'S CONNECTION. That would be better — no second process, no
# question about who owns the session — but the turn's connection lives inside
# `drive_turn_acp`'s worker thread, and reaching into it from an HTTP route would need a
# new cross-thread control channel. That is a larger change than this, and it is the
# reason this path opens its own client. The cost is honest and bounded: one extra mcode
# process per control action, which is what `probe_surface` already does for the
# capability menu.
#
# THE OPERATION NAMES ARE AN ALLOWLIST, and that is the point. An HTTP body supplies the
# name, so mapping it through a dict of literals is what stops a request from naming any
# method it likes on the protocol — including `session/prompt`, which would be a model
# turn smuggled in through a control route.
_CONTROL_OPS: dict[str, tuple[str, tuple[str, ...]]] = {
    # name: (what it does, the params it requires)
    "goal_get":       ("read the goal", ()),
    "goal_create":    ("create a goal", ("objective",)),
    "goal_patch":     ("change a goal", ()),
    "goal_clear":     ("clear the goal", ()),
    "queue_list":     ("read the queue", ()),
    "queue_enqueue":  ("add to the queue", ("text",)),
    "queue_update":   ("edit a queued message", ("itemId", "text")),
    "queue_delete":   ("drop a queued message", ("itemId",)),
    "queue_steer":    ("promote a queued message into the running turn", ("itemId",)),
    "steer":          ("inject text into the running turn", ("text",)),
    "activate":       ("make this the active session", ()),
    "delegation_get": ("read the delegation tree", ()),
    # NO PARAMS, and that is the correction. The server resolves `params.sessionId` as the
    # ROOT session and stops the whole delegation for it; there is no member id in this
    # call. The old `("sessionId",)` was the root id under a name that collided with the
    # context key, which is what let a request body redirect the operation to another
    # session. The session id comes from the chat row, so nothing is required here.
    "delegation_stop": ("stop this session's delegated work", ()),
    # REACHABLE OVER HTTP, NO UI CONTROL YET, and the reason is a missing input rather
    # than an omission: the valid `modeId`s arrive in `session/new`'s response
    # (`availableModes`), which Rigma does not store. `config_set` can be a dropdown
    # because the server re-sends its option list on every `config_option_update`; this
    # one has nothing to read. Hardcoding `plan`/`default` was rejected — it would be a
    # control that silently stops matching the server on the next mcode version.
    "mode_set":       ("change the mode (plan/default)", ("modeId",)),
    "config_set":     ("change a configOption (model, permissionMode)",
                       ("optionId", "value")),
}

# The names the UI may offer, in the order it should offer them. A separate tuple rather
# than `list(_CONTROL_OPS)`, because dict order is an implementation detail of how the
# table happens to be written and this is a presentation order.
CONTROL_OPS = tuple(_CONTROL_OPS)


def control_op_error(op: str, params: dict | None) -> str:
    """Why this operation cannot be attempted, or "" if it can.

    Separate from `drive_control` so the ROUTE can refuse a bad request with a 400
    BEFORE opening a process — an unknown operation or a missing parameter is a client
    mistake, not a transport failure, and paying for an mcode launch to discover it
    would also report it as the wrong kind of error.
    """
    if op not in _CONTROL_OPS:
        return (f"unknown operation {op!r}; known: "
                + ", ".join(sorted(_CONTROL_OPS)))
    missing = [k for k in _CONTROL_OPS[op][1] if not (params or {}).get(k)]
    if missing:
        return f"{op} requires {', '.join(missing)}"
    return ""


def _control_argv(exe: "str | list[str]") -> list[str]:
    """The argv for a control client, from a path, an argv, or a test double.

    THE SPACE IS THE WHOLE POINT. `bin_path()` returns an UNQUOTED path, and mcode very
    often lives under one containing a space — the per-user npm directory is the
    ordinary Windows install. `shlex.split(posix=False)` splits a `Program Files` path
    into its first two words,
    so argv[0] became a directory that does not exist and every control operation failed
    with a 502 while the `exec` turn path — which never splits — worked fine on the same
    value.

    So a real path is taken as ONE token. A string is split only when it is not an
    existing file, which is the test-double case (an interpreter plus a script) and
    nothing else; that keeps one parameter serving both without guessing.

    `acp` is appended unless it is already the last argument, so a caller that passed a
    complete command line is not given it twice. Checking the LAST argument rather than
    "anywhere" matters: a directory merely named `acp` earlier in a path would otherwise
    suppress the subcommand entirely.
    """
    if isinstance(exe, (list, tuple)):
        argv = [str(a) for a in exe]
    else:
        raw = str(exe)
        if os.path.isfile(raw):
            argv = [raw]
        else:
            try:
                import shlex as _shlex

                argv = _shlex.split(raw, posix=(os.name != "nt"))
            except Exception:
                argv = [raw]
            argv = [a.strip('"') for a in argv] or [raw]
    if not argv:
        return ["acp"]
    if argv[-1] != "acp":
        argv.append("acp")
    return argv


def drive_control(op: str, params: dict | None = None, *,
                  exe: "str | list[str]" = "",
                  session_id: str, cwd: str = "", timeout: float = 60.0) -> dict:
    """Perform ONE control-plane operation on an existing session.

    Returns `{"ok": bool, "op": str, "result": Any, "error": str}` and never raises,
    for the same reason `probe_surface` never raises: this is called from a route that
    is drawing a panel, and an exception would take down the panel rather than report
    one failure.

    A SESSION IS REQUIRED. Without one there is nothing to control, and creating a
    session to control would produce a goal on a session the chat is not using — a
    success that changed nothing the user can see.
    """
    out: dict = {"ok": False, "op": op, "result": None, "error": ""}
    why = control_op_error(op, params)
    if why:
        out["error"] = why
        return out
    if not session_id:
        out["error"] = ("this chat has no mcode session yet; the control plane needs "
                        "one, so it becomes available after the first turn")
        return out
    # The id goes into a protocol frame verbatim, so it is bounded rather than trusted.
    # A real one is short; this catches a mangled row or a caller passing something else.
    if len(session_id) > 256 or any(c in session_id for c in "\r\n\x00"):
        out["error"] = ("this chat's mcode session id is not usable "
                        "(it is empty, over-long, or contains control characters)")
        return out
    if not exe:
        exe = _default_exe()
    if not exe:
        out["error"] = "mcode is not on PATH"
        return out

    # THE SESSION ID IS NOT A PARAMETER. It was one: `params` was passed through to
    # `body.update(params)`, and every client method builds `{"sessionId":
    # self.session_id}` first — so a request body carrying `sessionId` drove a DIFFERENT
    # mcode session than the chat it was sent to. `delegation_stop` requires a
    # `sessionId`, which made the override look like a normal argument.
    #
    # Dropped here rather than at the eight call sites because the resumed session is the
    # operation's CONTEXT, not its input: no operation in `_CONTROL_OPS` has any business
    # naming a session, and the route already decided which one by looking it up on the
    # chat row. A caller who needs a different session must send to a different chat.
    body = dict(params or {})
    body.pop("sessionId", None)
    # THE BUDGET IS FOR THE WHOLE OPERATION, NOT FOR EACH REQUEST. `timeout` is what the
    # route advertises, and this function makes three requests: initialize, resume, and the
    # operation itself. Giving each the full budget made the real worst case about three
    # times the advertised one, inside a single `asyncio.to_thread` — and that thread comes
    # from the default executor, so concurrent control requests could starve unrelated
    # routes that use it too. A third each, with a floor so a small budget cannot produce
    # an instant timeout on work that would have succeeded.
    each = max(5.0, float(timeout) / 3.0)
    client = AcpClient(_control_argv(exe), cwd=cwd or None, default_timeout=each)
    try:
        client.start()
        # `declare_extensions` matters even here: without it the server still ANSWERS
        # every call correctly and sends no `goal_update`, so the panel would not
        # refresh after the user changed something. The answer and the notification are
        # separately gated, which is the asymmetry this module documents.
        client.initialize(timeout=each)
        client.session_resume(session_id, cwd=cwd or None, timeout=each)
        if op == "goal_get":
            res = client.goal_get(timeout=each)
        elif op == "goal_create":
            res = client.goal_create(body, timeout=each)
        elif op == "goal_patch":
            res = client.goal_patch(body, timeout=each)
        elif op == "goal_clear":
            res = client.goal_clear(timeout=each)
        elif op == "queue_list":
            res = client.queue_list(timeout=each)
        elif op == "queue_enqueue":
            res = client.queue_enqueue(body, timeout=each)
        elif op == "queue_update":
            res = client.queue_update(body, timeout=each)
        elif op == "queue_delete":
            res = client.queue_delete(body, timeout=each)
        elif op == "queue_steer":
            res = client.queue_steer(body, timeout=each)
        elif op == "steer":
            res = client.steer(body, timeout=each)
        elif op == "activate":
            res = client.activate(timeout=each)
        elif op == "delegation_get":
            res = client.delegation_get(timeout=each)
        elif op == "delegation_stop":
            res = client.delegation_stop(body, timeout=each)
        elif op == "mode_set":
            res = client.set_mode(str(body["modeId"]), timeout=each)
        elif op == "config_set":
            res = client.set_config_option(str(body["optionId"]), body["value"],
                                          timeout=each)
        else:  # pragma: no cover - `control_op_error` already refused these
            raise AcpError(f"unhandled operation {op!r}")
        out["result"] = res
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.stop()
        except Exception:
            pass
    return out


def drive_turn_acp(prompt: str, *, exe: str = "", base_url: str = "",
                   model: str = "", system_prompt: str = "", session_id: str = "",
                   cwd: str = "", max_tokens: int = 4096,
                   context_window: int = 32768, timeout: float = 1800.0,
                   state: dict | None = None, cancel=None,
                   permission: str = "full", on_permission=None,
                   on_question=None):
    """Run one mcode turn over ACP and yield what happened, in Rigma's vocabulary.

    A generator, matching `drive_turn`'s contract: the caller owns the thread, so a
    stalled mcode stalls a worker rather than the event loop streaming a chat.

    `session_id` (or `state["session_id"]`) CONTINUES a session rather than starting
    one. The session this turn actually ran in is written back to `state`, so the
    next turn continues the same conversation — the same contract `drive_turn` has.

    `on_permission(method, params) -> bool | None` decides a permission request.
    Returning None means "no policy for this", which is reported as a decline and
    never as a denial — the server is told `cancelled`, which is its own word for
    "no answer". `on_question(method, params) -> dict | None` does the same for an
    elicitation. Neither has a default that answers, because a default that granted
    permission would be Rigma deciding something the user did not.
    """
    import queue as _queue
    import threading as _threading

    from .harness import TurnEvent

    mode = acp_permission_mode(permission)
    exe = exe or _default_exe()
    if not exe:
        yield TurnEvent("error", "mcode is not on PATH")
        return

    events: "_queue.Queue" = _queue.Queue()
    stopped = _threading.Event()
    # Set once the prompt request has returned or failed, so the drain loop knows
    # there is nothing more coming and does not wait out its timeout.
    finished = _threading.Event()
    outcome: dict = {}

    def _on_event(notification: dict) -> None:
        for ev in map_acp_update(notification):
            events.put(ev)

    def _on_request(method: str, params: dict):
        """Answer a server-initiated request.

        THE ANSWER IS A REAL ANSWER. An unanswered request leaves mcode waiting
        forever, which is worse than a decline — so every branch returns something.
        """
        if method == "session/request_permission":
            allow = None
            if on_permission is not None:
                try:
                    allow = on_permission(method, params)
                except Exception:
                    allow = None
            if allow is None:
                # No policy: decline in the server's own words. `allow-once` is used
                # when allowing because a one-shot grant is the narrow one, and
                # `allow-always` would be Rigma inventing a durable policy the user
                # never set.
                allow = mode == "bypassPermissions"
            reply = answer_permission(params, allow=allow)
            # REPORTED THROUGH THE CHANNEL THAT ALREADY EXISTS. DSH reports its
            # approvals as `approval/asked` and `approval/decided`, and
            # `chat/governance.ts` folds that pair into the governance trail — a
            # panel with a UI, tested, and already on screen. A second vocabulary
            # here would mean a second panel for the same concept and one more
            # place for the two to disagree, so the ACP decision is translated into
            # it instead. `asked` and `decided` share an `id`, which is what lets
            # the fold pair a question with its answer.
            tool_call = params.get("toolCall") or {}
            call_id = str(tool_call.get("toolCallId") or "")
            outcome = reply.get("outcome", {})
            option_id = str(outcome.get("optionId") or "")
            events.put(TurnEvent("state", event="approval/asked", data={
                "id": call_id,
                "toolName": str(tool_call.get("title") or ""),
                # The ACP surface has no `reason` field, so the reason is derived
                # from what the transport actually carries rather than invented: an
                # automatic decision is explained by the mode that made it.
                "reason": ("decided automatically by the session's permission mode"
                           if on_permission is None else "decided by Rigma's policy"),
                "auto": on_permission is None,
            }))
            events.put(TurnEvent("state", event="approval/decided", data={
                "id": call_id,
                "outcome": _acp_approval_outcome(outcome, option_id),
                "optionId": option_id,
                "policy": mode,
            }))
            return reply
        if method == "elicitation/create":
            content = None
            if on_question is not None:
                try:
                    content = on_question(method, params)
                except Exception:
                    content = None
            if content is None:
                # Declining is what makes mcode take its non-interactive fallback —
                # the same place it already lands today, so this changes nothing
                # except that the server is told rather than left waiting.
                events.put(TurnEvent("notice", text=(
                    "mcode asked a question and no answer was available, so it "
                    "continued without one")))
                return answer_elicitation(params, accepted=False)
            return answer_elicitation(params, accepted=True, content=content)
        # Anything else the server asks for: answer `None`, which is the protocol's
        # "declined", rather than leaving it unanswered.
        return None

    client = AcpClient([exe, "acp"], cwd=cwd or "", on_event=_on_event,
                       on_request=_on_request, default_timeout=timeout)

    def _run() -> None:
        try:
            client.start()
            client.initialize(timeout=min(timeout, 60.0))
            resume = str(session_id or (state or {}).get("session_id") or "").strip()
            if resume:
                try:
                    client.session_resume(resume, cwd=cwd, timeout=min(timeout, 60.0))
                except AcpError as exc:
                    # A session mcode no longer has is not a failed turn: start a new
                    # one and SAY SO, rather than continuing a conversation whose
                    # earlier half is gone without the user being told.
                    events.put(TurnEvent("notice", text=(
                        f"could not continue the previous mcode session "
                        f"({exc}); starting a new one")))
                    client.session_new(cwd or "", timeout=min(timeout, 60.0))
            else:
                client.session_new(cwd or "", timeout=min(timeout, 60.0))
            if state is not None:
                state["session_id"] = client.session_id
            # The permission policy is set through the protocol, which is how a
            # session changes it mid-flight — something `exec` cannot do at all.
            try:
                client.set_config_option("permissionMode", mode,
                                         timeout=min(timeout, 30.0))
            except AcpError:
                # Not fatal: the session keeps mcode's own default and the turn runs.
                pass
            if mode == "bypassPermissions":
                events.put(TurnEvent("notice", text=(
                    "mcode is running with permission prompts bypassed for this "
                    "turn; a workspace-scoped policy still bounds writes and "
                    "deletes")))
            res = client.prompt(prompt, timeout=timeout)
            outcome["stop"] = (res or {}).get("stopReason")
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            finished.set()

    worker = _threading.Thread(target=_run, name="acp-turn", daemon=True)
    worker.start()

    try:
        while True:
            if cancel is not None and cancel.is_set() and not stopped.is_set():
                stopped.set()
                # A NOTIFICATION, not a kill: ACP can ask the server to stop the
                # turn without ending the session, which `exec` cannot do at all.
                try:
                    client.cancel()
                except Exception:
                    pass
            try:
                ev = events.get(timeout=0.2)
            except _queue.Empty:
                if finished.is_set() and events.empty():
                    break
                if stopped.is_set() and finished.is_set():
                    break
                continue
            yield ev
    finally:
        client.stop()
        worker.join(timeout=5.0)

    if stopped.is_set():
        # Nothing failed: the session id is already recorded, so the next turn
        # continues this conversation.
        yield TurnEvent("notice", text="stopped at your request")
    if outcome.get("error"):
        yield TurnEvent("error", outcome["error"])


def _default_exe() -> str:
    """The mcode binary, resolved by the SAME function the `exec` adapter uses.

    `bin_path()` honours `RIGMA_MCODE_BIN` and then `PATH`, so the two transports
    can never disagree about which mcode is running.
    """
    try:
        from .harness_mcode import bin_path

        return bin_path() or ""
    except Exception:
        return ""
