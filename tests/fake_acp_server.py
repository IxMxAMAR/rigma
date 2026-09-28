"""A scripted `mcode acp` server, for testing Rigma's ACP client.

WHY A FAKE AND NOT THE REAL THING. The real server answers `session/prompt` by
calling a model, and the user's standing order forbids loading one. So the client's
transport is exercised against this: a real child process, real pipes, real
newline-delimited JSON-RPC, real bidirectional requests — everything except the
model.

It deliberately reproduces the two behaviours that a naive client gets wrong:

  1. It sends SERVER-INITIATED REQUESTS (`session/request_permission` and
     `elicitation/create`) and WAITS for the reply before continuing. A client that
     only reads responses and notifications deadlocks here rather than failing
     loudly.
  2. It exits if stdin reaches EOF, like the real one — which is why a probe whose
     stdin was a file reported a false refusal.

Invoked as: python fake_acp_server.py [--modes a,b] [--no-extensions]
                                      [--ask-permission] [--ask-question]
                                      [--prompt-error MESSAGE]
"""

from __future__ import annotations

import atexit
import json
import pathlib
import sys

MODES = ["default", "plan"]

# The server's own extension list, as the real one advertises it in the
# `initialize` RESPONSE. Declared once here so the fake cannot drift from the
# client's table without a test noticing.
EXTENSION_METHODS = [
    "session/activate", "mcode/session/activate", "mcode/session/steer",
    "mcode/session/queue/list", "mcode/session/queue/enqueue",
    "mcode/session/queue/update", "mcode/session/queue/delete",
    "mcode/session/queue/steer",
    "mcode/session/goal/get", "mcode/session/goal/create",
    "mcode/session/goal/patch", "mcode/session/goal/clear",
    "mcode/session/delegation/get", "mcode/session/delegation/stop",
]


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    modes = list(MODES)
    no_extensions = False
    ask_permission = False
    ask_question = False
    silent = False
    prompt_error = ""
    resume_renames = False
    resume_unknown = False
    record_path = ""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--modes" and i + 1 < len(argv):
            modes = [m for m in argv[i + 1].split(",") if m]
            i += 2
            continue
        if a == "--no-extensions":
            no_extensions = True
        elif a == "--ask-permission":
            ask_permission = True
        elif a == "--ask-question":
            ask_question = True
        elif a == "--silent":
            # Read requests and answer NONE of them. This is the shape of the bug
            # that produced the false `mcode login` blocker: a client that reads an
            # unanswered request as a refusal. Only `initialize` is answered, so a
            # test can get a session and then observe true silence.
            silent = True
        elif a == "--record" and i + 1 < len(argv):
            record_path = argv[i + 1]
            i += 2
            continue
        elif a == "--resume-renames":
            resume_renames = True
        elif a == "--resume-unknown":
            resume_unknown = True
        elif a == "--prompt-error" and i + 1 < len(argv):
            prompt_error = argv[i + 1]
            i += 2
            continue
        i += 1

    state = {"sessionId": "", "mode": modes[0] if modes else "default",
             "permission": "auto", "model": "m:custom_provider%3Arigma:local-test:v:thinking",
             "goal": None, "asked": 0, "queue": [],
             # THE GATE. mcode gates ALL FOUR extension notifications on the client
             # declaring this; it is not symmetric with the extension list it
             # advertises in the initialize RESPONSE. Reproduced here so the client's
             # declaration is tested rather than assumed:
             #   function po(e){let t=e._meta?.["minimax-code/extensions"];
             #     return t===!0||oh(t)&&t.version===1&&t.notifications===!0}
             "extensions_enabled": False,
             # R6-ACP-TURN: the resume surface. `resume_renames` makes the server
             # answer with a DIFFERENT id than the one asked for, which the real
             # server may do and which a client trusting its own id would miss.
             # `resume_unknown` errors an id the server does not have, so the
             # driver's "start a new session and say so" path is reachable.
             "known_sessions": ["mvs_fake_session", "mvs_older_session"],
             "resumed": "", "renamed_session": "mvs_renamed_session",
             "resume_renames": resume_renames,
             "resume_unknown": resume_unknown,
             # Every config value the server REFUSED. Recorded because the driver
             # deliberately does not fail a turn over a refused option, so from the
             # client side "the server refused it" and "the driver never asked" are
             # otherwise indistinguishable.
             "rejected_config": []}
    server_id = 900

    def _dump_rejections() -> None:
        """Write the refused config values where a test can read them.

        WHY A FILE. The fake owns stdout for JSON-RPC, so it cannot print, and the
        rejections are the only evidence that a pass-through mode reached the server
        and was refused. `--record <path>` names the file.
        """
        if not record_path:
            return
        # Deliberately NOT wrapped in a bare except: a recorder that fails silently
        # makes a test read "nothing was refused" when the truth is "nothing was
        # written", which is the exact false-negative this whole round is about.
        pathlib.Path(record_path).write_text(
            json.dumps(state.get("rejected_config", [])), encoding="utf-8")

    atexit.register(_dump_rejections)

    def ask(method: str, params: dict):
        """Send a SERVER-INITIATED request and block for its reply."""
        nonlocal server_id
        server_id += 1
        emit({"jsonrpc": "2.0", "id": server_id, "method": method, "params": params})
        for line in sys.stdin:                       # blocks until the reply arrives
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            if frame.get("id") == server_id:
                return frame.get("result")
            # Anything else arriving while we wait is a request from the client;
            # answer it so a well-behaved client is never left hanging either.
            if frame.get("method") and "id" in frame:
                emit({"jsonrpc": "2.0", "id": frame["id"], "result": None})
        return None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")
        method = str(req.get("method") or "")
        params = req.get("params") or {}

        if silent and method not in ("initialize", "session/new"):
            # Deliberately answer nothing. A client must report this as "no reply",
            # never as a refusal.
            continue

        if method == "initialize":
            caps = {"loadSession": True, "sessionCapabilities":
                    {"list": {}, "fork": {}, "resume": {}, "close": {}}}
            cc = params.get("clientCapabilities") or {}
            ext = (cc.get("_meta") or {}).get("minimax-code/extensions")
            state["extensions_enabled"] = (
                ext is True or (isinstance(ext, dict)
                                and ext.get("version") == 1
                                and ext.get("notifications") is True))
            state["declared"] = {
                "elicitation": (cc.get("elicitation") or {}).get("form") is not None,
                "plan": bool(cc.get("plan")),
            }
            res = {"protocolVersion": 1,
                   "agentInfo": {"name": "minimax-code", "title": "MiniMax Code",
                                 "version": "0.5.4-fake"},
                   "agentCapabilities": caps}
            if not no_extensions:
                res["_meta"] = {"minimax-code/extensions": {
                    "methods": list(EXTENSION_METHODS),
                    "notifications": ["mcode/session/goal_update",
                                      "mcode/session/queue_update",
                                      "mcode/session/delegation_update"]}}
            emit({"jsonrpc": "2.0", "id": rid, "result": res})
            continue

        if method == "session/new":
            state["sessionId"] = "mvs_fake_session"
            emit({"jsonrpc": "2.0", "id": rid, "result": {
                "sessionId": state["sessionId"],
                "modes": {"currentModeId": state["mode"],
                          "availableModes": [{"id": m, "name": m.title()}
                                             for m in modes]},
                "configOptions": [
                    {"id": "permissionMode", "currentValue": state["permission"],
                     "options": [{"value": v} for v in
                                 ("default", "auto", "bypassPermissions")]},
                    {"id": "model", "currentValue": state["model"],
                     "options": [{"value": state["model"]},
                                 {"value": "m:minimax:MiniMax-M3:v:thinking"}]},
                ]}})
            continue

        if method in ("session/resume", "session/load"):
            # The real handlers are `onRequest(ee.agent.session.resume, async z =>
            # s(z.params.sessionId, ...))` and the same shape for `load`: the
            # parameter is `sessionId` and the answer carries `sessionId` back.
            # The server may answer with a DIFFERENT id than the one asked for, and
            # it does here when `--resume-renames` is set, so a client that assumes
            # its own id was accepted is caught.
            asked = str(params.get("sessionId") or "")
            if not asked:
                emit({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602,
                                "message": "Invalid params: sessionId is required"}})
                continue
            if state.get("resume_unknown") and asked not in state["known_sessions"]:
                emit({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602,
                                "message": f"Invalid params: unknown session: {asked}"}})
                continue
            state["sessionId"] = (state["renamed_session"]
                                  if state.get("resume_renames") else asked)
            state["resumed"] = asked
            emit({"jsonrpc": "2.0", "id": rid, "result": {
                "sessionId": state["sessionId"],
                "modes": {"currentModeId": state["mode"],
                          "availableModes": [{"id": m, "name": m.title()}
                                             for m in modes]},
                "configOptions": [
                    {"id": "permissionMode", "currentValue": state["permission"]},
                ]}})
            continue

        if method == "session/list":
            emit({"jsonrpc": "2.0", "id": rid, "result": {
                "sessions": [{"sessionId": s} for s in state["known_sessions"]]}})
            continue

        if method == "session/set_mode":
            want = str(params.get("modeId") or "")
            if want not in modes:
                emit({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602,
                                "message": f"Invalid params: Unsupported mode: {want}"}})
            else:
                state["mode"] = want
                emit({"jsonrpc": "2.0", "id": rid,
                      "result": {"_meta": {"minimax-code/transition": "next_prompt"}}})
            continue

        if method == "session/set_config_option":
            key = str(params.get("configId") or "")
            val = params.get("value")
            # Recorded FIRST, before the `continue` at the end of this branch. A
            # recorder placed after the branch is skipped by exactly the rejection
            # path it exists to capture, so the file reads `[]` and the test reads
            # "nothing was refused" from "nothing was recorded".
            if key == "permissionMode" and val not in ("default", "auto",
                                                       "bypassPermissions"):
                state["rejected_config"].append({"configId": key, "value": val})
            if key == "permissionMode" and val not in ("default", "auto",
                                                       "bypassPermissions"):
                # The REAL rejection, copied from the runtime's own guard:
                #   function nS(e){if(e==="default"||e==="auto"||
                #     e==="bypassPermissions")return e;
                #     throw j.invalidParams(void 0,`Unsupported permission mode: ${e}`)}
                # The two vocabularies share no member, so `exec`'s `full` lands here.
                emit({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602,
                                "message": f"Unsupported permission mode: {val}"}})
            else:
                state[key] = val
                emit({"jsonrpc": "2.0", "id": rid, "result": {}})
            continue

        if method == "mcode/session/goal/get":
            emit({"jsonrpc": "2.0", "id": rid, "result": {"goal": state["goal"]}})
            continue
        if method == "mcode/session/goal/create":
            state["goal"] = {"id": "goal_fake_1",
                             "objective": str(params.get("objective") or ""),
                             "status": "active"}
            emit({"jsonrpc": "2.0", "id": rid, "result": {"goal": state["goal"]}})
            # GATED, exactly as the real server gates it. An undeclared client gets
            # a correct ANSWER and no update — the failure mode that looks like
            # "mcode does not notify".
            if state["extensions_enabled"]:
                emit({"jsonrpc": "2.0", "method": "mcode/session/goal_update",
                      "params": {"sessionId": state["sessionId"], "goal": state["goal"]}})
            continue
        if method == "mcode/session/goal/clear":
            state["goal"] = None
            emit({"jsonrpc": "2.0", "id": rid, "result": {}})
            continue
        if method == "mcode/session/queue/list":
            emit({"jsonrpc": "2.0", "id": rid, "result": {"items": list(state["queue"])}})
            continue
        if method == "mcode/session/queue/enqueue":
            item = {"itemId": f"q{len(state['queue']) + 1}",
                    "sessionId": state["sessionId"], "status": "queued",
                    "content": str(params.get("text") or "")}
            state["queue"].append(item)
            emit({"jsonrpc": "2.0", "id": rid,
                  "result": {"itemId": item["itemId"], "status": "queued",
                             "position": len(state["queue"])}})
            continue
        if method == "mcode/session/queue/update":
            for it in state["queue"]:
                if it["itemId"] == params.get("itemId"):
                    it["content"] = str(params.get("text") or "")
                    emit({"jsonrpc": "2.0", "id": rid, "result": {"item": it}})
                    break
            else:
                emit({"jsonrpc": "2.0", "id": rid, "result": {"item": None}})
            continue
        if method == "mcode/session/queue/delete":
            before = len(state["queue"])
            state["queue"] = [i for i in state["queue"]
                              if i["itemId"] != params.get("itemId")]
            emit({"jsonrpc": "2.0", "id": rid,
                  "result": {"item": (state["queue"][0] if state["queue"]
                                      and len(state["queue"]) < before else None)}})
            continue
        if method == "mcode/session/queue/steer":
            emit({"jsonrpc": "2.0", "id": rid,
                  "result": {"queueItemId": params.get("itemId"), "turnId": "turn_1"}})
            continue
        if method in ("session/activate", "mcode/session/activate"):
            emit({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": state["sessionId"]}})
            if state["extensions_enabled"]:
                emit({"jsonrpc": "2.0",
                      "method": "mcode/session/current_session_update",
                      "params": {"sessionId": state["sessionId"]}})
            continue
        if method == "mcode/session/steer":
            emit({"jsonrpc": "2.0", "id": rid, "result": {"turnId": "turn_1", "mode": "steered"}})
            continue
        if method == "mcode/session/delegation/get":
            emit({"jsonrpc": "2.0", "id": rid, "result": {"delegations": []}})
            continue

        if method == "session/cancel":
            emit({"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": state["sessionId"],
                             "update": {"sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "[cancelled]"}}}})
            continue

        if method == "session/prompt":
            state["asked"] += 1
            if ask_permission:
                decision = ask("session/request_permission", {
                    "sessionId": state["sessionId"],
                    "toolCall": {"toolCallId": "call_1", "title": "write file"},
                    "options": [{"optionId": "allow-once", "name": "Allow once",
                                 "kind": "allow_once"},
                                {"optionId": "allow-always", "name": "Always",
                                 "kind": "allow_always"},
                                {"optionId": "deny", "name": "Deny",
                                 "kind": "reject_once"}]})
                emit({"jsonrpc": "2.0", "method": "session/update",
                      "params": {"sessionId": state["sessionId"],
                                 "update": {"sessionUpdate": "agent_message_chunk",
                                            "content": {"type": "text",
                                                        "text": f"perm={json.dumps(decision)}"}}}})
            if ask_question:
                answer = ask("elicitation/create", {
                    "sessionId": state["sessionId"],
                    "message": "Which directory?",
                    "requestedSchema": {"type": "object",
                                        "properties": {"path": {"type": "string"}}}})
                emit({"jsonrpc": "2.0", "method": "session/update",
                      "params": {"sessionId": state["sessionId"],
                                 "update": {"sessionUpdate": "agent_message_chunk",
                                            "content": {"type": "text",
                                                        "text": f"answer={json.dumps(answer)}"}}}})
            emit({"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": state["sessionId"],
                             "update": {"sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "hello "}}}})
            emit({"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": state["sessionId"],
                             "update": {"sessionUpdate": "agent_thought_chunk",
                                        "content": {"type": "text", "text": "thinking"}}}})
            emit({"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": state["sessionId"],
                             "update": {"sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "world"}}}})
            if prompt_error:
                emit({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32000, "message": prompt_error}})
            else:
                emit({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
            continue

        if no_extensions and method.startswith(("mcode/", "session/activate")):
            # Not advertised AND not registered — the real server registers only the
            # extension strings it enables. Without this, `--no-extensions` would
            # still answer every extension call, and a client could not tell
            # "advertised" from "works by accident".
            emit({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32601,
                            "message": f'"Method not found": {method}',
                            "data": {"method": method}}})
            continue

        # Unknown method: a well-behaved dispatcher answers -32601 rather than dying.
        emit({"jsonrpc": "2.0", "id": rid,
              "error": {"code": -32601,
                        "message": f'"Method not found": {method}',
                        "data": {"method": method}}})

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
