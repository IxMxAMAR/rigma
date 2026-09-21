"""MCP server: Rigma's own tools, offered to an external agent.

The mirror of `mcp_client.py`. That module makes every MCP server a Rigma tool;
this one makes Rigma a tool PROVIDER for somebody else's agent.

Why this is the only shape that respects the ownership line: the arm's tool
roster stays the arm's. It decides whether to load this server and whether to
call what it offers. Rigma never edits a message the arm sends to its model —
which is what the four obvious alternatives (normalising roles, prepending
Rigma's prompt, shrinking the roster, re-running Rigma's tool-call repair on the
arm's calls) would all require.

The other half of the win is free: `tools/call` goes through `run_tool`, so the
arm inherits Rigma's weak-model repair layer — `resolve_tool_name` repairs a
near-miss tool name and `normalize_tool_args` repairs the arguments — without
Rigma touching the arm's protocol at all.

## What is offered, and what is deliberately not

Every offered tool is a schema on EVERY request, forever. mcode already sends 18
tool schemas per turn, so a second, worse copy of a tool the arm already owns is
pure context cost. The roster is therefore the tools the arm CANNOT do, and the
test that pins it makes adding one a deliberate act.

Each entry must be justified by its DEPENDENCIES, not its name — that is the
question that decides whether an in-process server can serve it at all:

  search_my_documents   rag_dir()/sidecar.json + the sidecar's HTTP port. A file
                        and a port. No session, no Rigma process. Eligible.
  remember, recall      rigma_home()/model_memory.json. A file, read and written
                        whole. No session, no Rigma process. Eligible — and worth
                        the two small schemas, because this is the one kind of
                        memory the arm does NOT have: its own session remembers
                        the conversation, but not the durable facts about the
                        user, and sharing this file means a fact learned in an
                        arm chat is there for the native loop and the other way
                        round. That sharing is the point, not a side effect. The
                        arm has exactly the power the native model already has
                        here — same file, same trust, no new surface.

NOT offered, and why — these are not oversights:

  undo_last_change      Its journal only records RIGMA's own write_file /
                        edit_file. The arm edits with its own tools, which Rigma
                        never sees, so it would answer "nothing to undo" every
                        single time — a tool that looks like it works and finds
                        nothing. Making it work needs a filesystem watcher,
                        which is a different feature.
  view_image            Returns Rigma's IMAGE_SENTINEL for RIGMA'S OWN LOOP to
                        read and inject as vision. An external agent's loop has
                        never heard of it, and injecting it into the arm's
                        request is exactly the line this seam does not cross.
  read_file, write_file, edit_file, grep, find_files, list_directory,
  run_shell, run_python, web_search, fetch_url, http_request
                        The arm has all of them, integrated with its own
                        permission model.

## Transport rules

stdio, newline-delimited JSON-RPC 2.0, UTF-8. **Nothing but JSON-RPC may ever
reach stdout** — one stray `print` corrupts the stream and the arm sees a server
that does not work. Diagnostics go to stderr.
"""
from __future__ import annotations

import json
import os
import sys

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "rigma"

# The curated roster. Adding a name here is a deliberate act, and
# `test_the_roster_is_exactly_what_was_justified` fails until it is justified in
# this file's docstring as well.
_ROSTER = ("search_my_documents", "remember", "recall")

# Tools whose result is not a failure even though it begins with "error". Empty
# today; it exists so the `isError` heuristic below is not the only word on it.
_NOT_AN_ERROR: set[str] = set()


def workspace() -> str:
    """The workspace this server was pointed at, from the environment.

    mcode launches the server with whatever env Rigma registered, so the
    workspace is passed rather than guessed — a tool that silently operated on
    the wrong directory would be worse than one that refused.
    """
    return str(os.environ.get("RIGMA_MCP_WORKSPACE") or "").strip()


def profile() -> str:
    """The run profile, so the same gates the native loop applies still apply.

    This server is a SECOND execution path into Rigma's tools, and a second path
    that skipped the profile gates would be a hole rather than a feature. Read
    from the environment for the same reason the workspace is.
    """
    return str(os.environ.get("RIGMA_MCP_PROFILE") or "all").strip() or "all"


def ctx() -> dict:
    """The context dict `run_tool` expects.

    Deliberately minimal and deliberately pessimistic: `allow_code` is False
    unless the environment says otherwise, so a tool that needs it is refused
    rather than silently permitted by a server nobody configured.
    """
    return {
        "workspace": workspace(),
        "profile": profile(),
        "allow_code": os.environ.get("RIGMA_MCP_ALLOW_CODE") == "1",
        "run_id": "",
    }


def offered() -> list[dict]:
    """The roster as MCP tool definitions.

    Built from Rigma's OWN registry rather than hand-written, so a tool whose
    arguments change cannot drift from the schema the arm was shown. Filtered
    through `tool_specs` so the same `needs` rules apply — which is why
    `search_my_documents` disappears entirely when no documents are indexed,
    instead of being offered and answering "nothing is indexed yet" on every
    turn.
    """
    from . import rag
    from . import tools as toolkit

    try:
        has_rag = bool(rag.recorded_sidecar_port())
    except Exception:
        has_rag = False
    try:
        specs = toolkit.tool_specs(has_rag=has_rag, workspace=workspace(),
                                   profile=profile())
    except Exception:
        return []

    out: list[dict] = []
    for s in specs:
        fn = s.get("function") if isinstance(s, dict) else None
        if not isinstance(fn, dict) or fn.get("name") not in _ROSTER:
            continue
        out.append({
            "name": fn["name"],
            "description": str(fn.get("description") or ""),
            "inputSchema": fn.get("parameters") or {"type": "object"},
        })
    return out


def _text(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": str(text)}],
            "isError": bool(is_error)}


def call(name: str, args: dict) -> dict:
    """Run one tool and wrap the result the way MCP wants it.

    A failure comes back as CONTENT with `isError`, never as a JSON-RPC error:
    the model is the party who has to react to it, and a protocol-level error is
    invisible to the model — the arm would just see a broken server.
    """
    from . import tools as toolkit

    if name not in _ROSTER:
        return _text(f"error: '{name}' is not offered by this server", True)
    try:
        out = toolkit.run_tool(name, args or {}, ctx())
    except Exception as e:                 # `run_tool` should not raise; if it
        return _text(f"error running {name}: {e}", True)   # does, say so
    text = str(out)
    bad = text.startswith("error") and name not in _NOT_AN_ERROR
    return _text(text, bad)


def _ok(mid, result) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": code, "message": message}}


def handle(msg) -> dict | None:
    """One message in, one reply out — or None for a notification.

    Pure, so the protocol can be tested without a subprocess. A notification has
    no `id` and MUST NOT be answered; replying to one is a protocol violation
    that some clients treat as fatal.
    """
    if not isinstance(msg, dict):
        return _err(None, -32600, "invalid request")

    mid = msg.get("id")
    method = str(msg.get("method") or "")
    if mid is None:
        return None                       # notification: nothing to reply to

    if method == "initialize":
        return _ok(mid, {"protocolVersion": PROTOCOL_VERSION,
                         "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": {"name": SERVER_NAME, "version": _version()}})
    if method == "tools/list":
        return _ok(mid, {"tools": offered()})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = str(params.get("name") or "")
        args = params.get("arguments")
        return _ok(mid, call(name, args if isinstance(args, dict) else {}))
    if method == "ping":
        return _ok(mid, {})
    return _err(mid, -32601, f"unknown method: {method}")


def _version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return "0"


def serve(stdin=None, stdout=None) -> None:
    """The loop. Reads until stdin closes, which is how a client stops us."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            replies = [_err(None, -32700, "parse error")]
        else:
            # MCP allows a batch. Handling it is three lines, and a client that
            # batches getting silence back would look like a hung server.
            batch = msg if isinstance(msg, list) else [msg]
            replies = [r for r in (handle(m) for m in batch) if r is not None]
        for reply in replies:
            stdout.write(json.dumps(reply) + "\n")
        stdout.flush()


def main() -> int:
    try:
        serve()
    except (BrokenPipeError, KeyboardInterrupt):
        pass            # the client went away; that is a normal way to end
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
