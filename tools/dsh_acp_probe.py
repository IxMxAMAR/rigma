"""Probe `dsh --profile acp`: the one DSH transport Rigma does NOT drive.

WHY THIS MATTERS. Rigma drives DSH over the SDK stdio wire, whose client method map
is exactly `initialize`, `session/prompt`, `shutdown` — and whose server->client
half is NOTIFICATIONS ONLY (`HarnessSdkNotificationMap`). There is no way for Rigma
to answer anything: not an approval, not a question. That is why the whole
governance surface bridged in R3 is display-only, and it is a property of the
transport rather than of the UI.

ACP is a different transport and it IS bidirectional. Its contract includes
`session/request_permission` — "a permission prompt with one-shot allow/reject
choices; your client can answer automatically". And unlike mcode's ACP server, DSH's
requires NO authentication (`authenticate` returns immediate success).

So this probe answers one question: can Rigma have an INTERACTIVE permission prompt
by driving `dsh --profile acp` instead of (or alongside) the SDK wire?

Read-only: starts a server, asks `initialize`, asks for one session, reports what
came back, kills it. It never sends a prompt, so no model call is made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

DSH = r"C:\AI\deepseek-harness\python\sdk-runtime\node_modules\.bin\dsh.CMD"
WORKSPACE = r"C:\ComfyUI\RD\rigma-review\.scratch\acp-ws"


def main() -> int:
    os.makedirs(WORKSPACE, exist_ok=True)
    env = dict(os.environ)
    env["DSH_HOME"] = os.path.expandvars(r"%USERPROFILE%\.dsh")

    proc = subprocess.Popen(
        [DSH, "--profile", "acp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    out: list[str] = []
    err: list[str] = []
    threading.Thread(target=lambda: [out.append(x.rstrip("\n")) for x in proc.stdout],
                     daemon=True).start()
    threading.Thread(target=lambda: [err.append(x.rstrip("\n")) for x in proc.stderr],
                     daemon=True).start()

    def send(obj):
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def wait(n, seconds=20.0):
        end = time.time() + seconds
        while time.time() < end and len(out) < n:
            time.sleep(0.2)
        return len(out) >= n

    # 1. ACP handshake.
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": 1,
                     "clientCapabilities": {"fs": {"readTextFile": False,
                                                   "writeTextFile": False}}}})
    if not wait(1):
        print("  initialize: NO ANSWER")
        print("  stderr:")
        for line in err[:10]:
            print("    %s" % line[:200])
        proc.kill()
        return 1
    init = json.loads(out[0])
    res = init.get("result") or {}
    print("  initialize answered")
    print("    protocolVersion: %s" % res.get("protocolVersion"))
    print("    agentInfo: %s" % json.dumps(res.get("agentInfo")))
    caps = res.get("agentCapabilities") or {}
    print("    loadSession: %s" % caps.get("loadSession"))
    for k in ("sessionCapabilities", "promptCapabilities", "mcpCapabilities"):
        if k in caps:
            print("    %s: %s" % (k, json.dumps(caps[k])))
    # The DSH-specific extension block, if this transport carries one.
    meta = res.get("_meta") or {}
    if meta:
        print("    _meta keys: %s" % sorted(meta.keys()))
        for k, v in meta.items():
            print("      %s: %s" % (k, json.dumps(v)[:400]))

    # 2. A real session, so the config-option surface is visible.
    send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
          "params": {"cwd": WORKSPACE, "mcpServers": []}})
    if wait(2):
        sess = json.loads(out[1])
        if "error" in sess:
            print("")
            print("  session/new FAILED: %s" % json.dumps(sess["error"])[:400])
        else:
            r = sess.get("result") or {}
            print("")
            print("  session/new answered")
            print("    sessionId: %s" % r.get("sessionId"))
            opts = r.get("configOptions") or []
            print("    configOptions: %d" % len(opts))
            for o in opts:
                print("      id=%s name=%s" % (o.get("id"), o.get("name")))
                for ch in (o.get("options") or [])[:12]:
                    print("        - %s" % json.dumps(ch)[:120])
            modes = r.get("modes")
            if modes:
                print("    modes: %s" % json.dumps(modes)[:300])
    else:
        print("")
        print("  session/new: NO ANSWER")

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    if err:
        print("")
        print("  stderr (first 5):")
        for line in err[:5]:
            print("    %s" % line[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
