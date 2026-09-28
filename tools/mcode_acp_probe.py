"""Ask `mcode acp` what it actually supports.

`mcode acp` is a real subcommand ("Run MiniMax Code as an Agent Client Protocol
server over stdio"), and the CHANGELOG claims it expands the session control
plane with Session fork, mode and configuration switching, and Skill discovery.
Before Rigma grows an ACP *client* — the largest single piece of work left — it is
worth establishing what the server actually advertises.

This speaks just enough ACP to ask. It sends `initialize` and reads whatever comes
back, then sends a deliberately unknown method to see whether the server answers
with a proper JSON-RPC error (meaning it is a well-behaved dispatcher we can talk
to) or simply dies.

Read-only: it starts a server, asks two questions, and kills it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

MCODE = r"C:\Users\amren\AppData\Roaming\npm\mcode.cmd"


def main() -> int:
    env = dict(os.environ)
    env["MINIMAX_DATA_DIR"] = r"C:\ComfyUI\RD\rigma-review\.scratch\mcode-acp-probe"
    proc = subprocess.Popen(
        [MCODE, "acp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    out: list[str] = []
    err: list[str] = []

    def pump(stream, sink):
        try:
            for line in stream:
                sink.append(line.rstrip("\n"))
        except Exception:                                   # stream closed
            pass

    threading.Thread(target=pump, args=(proc.stdout, out), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, err), daemon=True).start()

    def send(obj):
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def wait_for(n, seconds=12.0):
        """Wait for n lines, or give up. Bounded: this must not hang."""
        end = time.time() + seconds
        while time.time() < end and len(out) < n:
            time.sleep(0.15)
        return len(out) >= n

    # 1. ACP handshake. protocolVersion 1 is what the spec's initialize takes.
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": 1,
                     "clientCapabilities": {"fs": {"readTextFile": False,
                                                   "writeTextFile": False}}}})
    got_init = wait_for(1)
    print("  initialize answered: %s" % got_init)
    if got_init:
        print("  <- %s" % out[0][:600])

    # 2. An unknown method. A well-behaved dispatcher returns a JSON-RPC error
    #    naming the method; a crash means the surface is not worth building on.
    send({"jsonrpc": "2.0", "id": 2, "method": "session/fork",
          "params": {"sessionId": "probe", "cwd": os.getcwd()}})
    got_fork = wait_for(2)
    print("  session/fork answered: %s" % got_fork)
    if got_fork:
        print("  <- %s" % out[1][:600])

    # 3. Does it advertise a method list anywhere in the initialize result?
    if got_init:
        try:
            res = json.loads(out[0]).get("result") or {}
            print("  initialize result keys: %s" % sorted(res.keys()))
            for k in ("methods", "capabilities", "serverCapabilities"):
                if k in res:
                    print("  %s: %s" % (k, json.dumps(res[k])[:500]))
        except Exception as e:
            print("  (could not parse initialize result: %s)" % e)

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    print("  exit: %s" % proc.returncode)
    if err:
        print("  stderr (first 3):")
        for line in err[:3]:
            print("    %s" % line[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
