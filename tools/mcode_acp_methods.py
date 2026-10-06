"""Print the FULL method list `mcode acp` advertises, and whether we are logged in.

The first probe truncated the list at the console width. The list is the decisive
fact for whether an ACP client is worth building, so get all of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

MCODE = os.path.expandvars(r"%APPDATA%\npm\mcode.cmd")


def probe() -> dict:
    env = dict(os.environ)
    env["MINIMAX_DATA_DIR"] = r"C:\ComfyUI\RD\rigma-review\.scratch\mcode-acp-probe"
    proc = subprocess.Popen(
        [MCODE, "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8",
        errors="replace", bufsize=1)
    out: list[str] = []
    threading.Thread(
        target=lambda: [out.append(x.rstrip("\n")) for x in proc.stdout],
        daemon=True).start()
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": 1, "clientCapabilities": {}}}) + "\n")
    proc.stdin.flush()
    end = time.time() + 12
    while time.time() < end and not out:
        time.sleep(0.15)
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return json.loads(out[0]) if out else {}


def main() -> int:
    r = probe()
    result = r.get("result") or {}
    caps = result.get("agentCapabilities") or {}
    meta = (result.get("_meta") or {}).get("minimax-code/extensions") or {}
    methods = meta.get("methods") or []
    print("  agent: %s %s" % (result.get("agentInfo", {}).get("name"),
                              result.get("agentInfo", {}).get("version")))
    print("  protocolVersion: %s" % result.get("protocolVersion"))
    print("  loadSession: %s" % caps.get("loadSession"))
    print("  sessionCapabilities: %s" % json.dumps(caps.get("sessionCapabilities")))
    print("  mcpCapabilities: %s" % json.dumps(caps.get("mcpCapabilities")))
    print("  promptCapabilities: %s" % json.dumps(caps.get("promptCapabilities")))
    print("")
    print("  ADVERTISED METHODS (%d):" % len(methods))
    for m in sorted(methods):
        print("    %s" % m)
    print("")
    print("  other _meta keys: %s" % sorted(k for k in meta if k != "methods"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
