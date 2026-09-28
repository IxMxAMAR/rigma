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

    def _handle_request(self, frame: dict) -> None:
        """A server-initiated request. We MUST answer, or the turn stalls."""
        method = str(frame.get("method") or "")
        params = frame.get("params") or {}
        result: Any = None
        error: dict | None = None
        if self._on_request is None:
            # No handler is a REAL answer, not silence: an unanswered request would
            # leave mcode waiting forever. `None` is what the protocol uses for
            # "declined", and it is what makes the runtime fall back.
            result = None
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
        client_capabilities: dict = {
            "fs": {"readTextFile": True, "writeTextFile": True},
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
