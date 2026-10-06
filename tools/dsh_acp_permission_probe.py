"""Set the model via `session/set_config_option`, then see whether a permission
prompt arrives and can be answered.

The first attempt failed with `no API key for provider route "deepseek-official"`,
which is a configuration mistake on the probe's part, not a finding about ACP — the
server offers a whole catalog and the default is not necessarily usable. This uses
the `tokenjuice` route, which the catalog advertises and which is free until
2026-09-30.

It also exercises `session/set_config_option` for real, which is the second thing
Rigma cannot do over the SDK wire (the wire has no configuration method at all).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

DSH = r"C:\AI\deepseek-harness\python\sdk-runtime\node_modules\.bin\dsh.CMD"
WORKSPACE = r"C:\ComfyUI\RD\rigma-review\.scratch\acp-perm-ws"
# Which catalog group to select a model from. `tokenjuice` is free until
# 2026-09-30 and is the route this project already uses for its own subagents.
ROUTE_GROUP = "tokenjuice"


def main() -> int:
    os.makedirs(WORKSPACE, exist_ok=True)
    env = dict(os.environ)
    env["DSH_HOME"] = os.path.expandvars(r"%USERPROFILE%\.dsh")

    proc = subprocess.Popen(
        [DSH, "--profile", "acp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
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

    def wait_for(pred, seconds=90.0):
        end = time.time() + seconds
        while time.time() < end:
            for line in list(out):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if pred(o):
                    return o
            time.sleep(0.25)
        return None

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": 1, "clientCapabilities": {}}})
    wait_for(lambda o: o.get("id") == 1, 30)
    send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
          "params": {"cwd": WORKSPACE, "mcpServers": []}})
    sess = wait_for(lambda o: o.get("id") == 2, 30)
    sid = (sess or {}).get("result", {}).get("sessionId")
    print("  session: %s" % sid)

    # Point the session at a route that actually has a key. The value must match
    # the catalog byte for byte — a guess is rejected with `unknown model option`,
    # so read the catalog rather than constructing the string.
    cat = wait_for(lambda o: o.get("id") == 5, 1)  # no-op; keeps ids distinct
    _ = cat
    send({"jsonrpc": "2.0", "id": 5, "method": "session/new",
          "params": {"cwd": WORKSPACE, "mcpServers": []}})
    sess2 = wait_for(lambda o: o.get("id") == 5, 30)
    opts2 = (sess2 or {}).get("result", {}).get("configOptions") or []
    model_opt = next((o for o in opts2 if o.get("id") == "model"), None)
    choice = None
    for g in (model_opt or {}).get("options") or []:
        if g.get("group") == ROUTE_GROUP:
            first = (g.get("options") or [{}])[0]
            choice = first.get("value")
            break
    print("  catalog route for %s: %s" % (ROUTE_GROUP, choice))
    if choice is None:
        print("  route %s not in the catalog; cannot continue" % ROUTE_GROUP)
        proc.kill()
        return 1
    send({"jsonrpc": "2.0", "id": 4, "method": "session/set_config_option",
          "params": {"sessionId": sid, "configId": "model", "value": choice}})
    cfg = wait_for(lambda o: o.get("id") == 4, 30)
    print("  set_config_option(model=%s):" % choice)
    print("    %s" % json.dumps(cfg)[:400])

    send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
          "params": {"sessionId": sid, "prompt": [
              {"type": "text",
               "text": "Create a file named probe.txt containing the word hello."}]}})
    print("  prompt sent; watching for a server->client REQUEST...")

    perm = wait_for(lambda o: "method" in o and "id" in o
                    and "result" not in o and "error" not in o, 120)
    if perm is None:
        print("")
        print("  no server->client request. What arrived:")
        for line in list(out)[-14:]:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "method" in o:
                print("    <- %s" % o.get("method"))
            elif o.get("id") == 3:
                print("    settled: %s" % json.dumps(o)[:260])
        proc.kill()
        return 0

    print("")
    print("  *** server->client REQUEST: %s (id=%s) ***"
          % (perm.get("method"), perm.get("id")))
    print("    params: %s" % json.dumps(perm.get("params"))[:800])
    opts = (perm.get("params") or {}).get("options") or []
    allow = next((o for o in opts if str(o.get("kind", "")).startswith("allow")), None)
    print("    offered: %s" % json.dumps(
        [{"optionId": o.get("optionId"), "kind": o.get("kind")} for o in opts]))
    if allow:
        send({"jsonrpc": "2.0", "id": perm["id"],
              "result": {"outcome": {"outcome": "selected",
                                     "optionId": allow.get("optionId")}}})
        print("    answered: allow (%s)" % allow.get("optionId"))
    else:
        send({"jsonrpc": "2.0", "id": perm["id"],
              "result": {"outcome": {"outcome": "cancelled"}}})
        print("    answered: cancelled (no allow offered)")

    settled = wait_for(lambda o: o.get("id") == 3, 120)
    print("")
    print("  prompt settled after answering: %s"
          % ("yes" if settled else "NO"))
    if settled:
        print("    %s" % json.dumps(settled)[:260])
    print("  file written? %s" % os.path.exists(os.path.join(WORKSPACE, "probe.txt")))
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
