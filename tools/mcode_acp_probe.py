"""Ask `mcode acp` what it actually supports — and get a real answer.

`mcode acp` is a real subcommand ("Run MiniMax Code as an Agent Client Protocol
server over stdio"). Before Rigma grows an ACP *client* it is worth establishing
what the server actually does.

WHAT THIS PROBE GOT WRONG ONCE, AND WHY IT IS WRITTEN THIS WAY NOW. An earlier
version of the investigation reported that every session method answers

    {"code":-32000,"message":"Authentication required: Run `mcode login` and try again."}

and concluded the transport was blocked behind a browser sign-in. **That was not
true.** The claim came from a probe whose stdin was a FILE: mcode read the first
request, hit EOF on the rest, and exited 0 before answering `session/new`. An
unanswered request was then read as a refusal, and a blocker was manufactured out
of a probe bug. The binary does contain that string (`chunks/run-acp-command-
JPZMIXGP.js`, `sr()` wrapping a `Qd` error) but nothing here ever produced it.

Measured properly, with no credentials on this machine, `mcode acp` needs NO login:
`initialize`, `session/new`, `mcode/session/goal/get` and `session/set_mode` all
answer. It uses the BYOK provider Rigma already registers (`custom_provider: rigma`
in `~/.minimax/config.yaml`), which the README says "does not require a MiniMax
login".

So the rules this probe follows, and the reason for each:

  1. stdin is a PIPE and is kept OPEN for the process's life. Never a file: EOF
     makes mcode exit before it answers, which is indistinguishable from a refusal.
  2. stdout is drained by a thread and every wait is BOUNDED. A server that never
     answers must cost a timeout, not a hang.
  3. "No reply" is reported AS "no reply", never as an error or a refusal. Those
     are different facts and conflating them is exactly the bug above.
  4. `session/fork` is asked with a REAL session id from `session/new`. Asking with
     an invented id tests argument validation, not the capability.

Read-only by default: it starts a server, negotiates, reads capabilities, and kills
it. `--prompt` additionally sends one real prompt, which DOES call the model and
therefore needs Rigma's engine running (it will fail with an upstream Connection
error otherwise, which is itself the useful signal that the wiring is right).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

MCODE = os.environ.get("RIGMA_MCODE_BIN") or os.path.expandvars(r"%APPDATA%\npm\mcode.cmd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--prompt", default="",
                    help="also send one real prompt (calls the model; needs the engine up)")
    ap.add_argument("--timeout", type=float, default=25.0,
                    help="seconds to wait for each reply")
    args = ap.parse_args()

    proc = subprocess.Popen(
        [MCODE, "acp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
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

    def reply(want_id: int, seconds: float | None = None) -> dict | None:
        """The parsed reply with this id, or None if it never came.

        None means NO REPLY — which is reported as such and never as a refusal.
        """
        end = time.time() + (seconds if seconds is not None else args.timeout)
        while time.time() < end:
            for line in list(out):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("id") == want_id:
                    return obj
            time.sleep(0.1)
        return None

    def report(label: str, r: dict | None, clip: int = 700):
        if r is None:
            print(f"  {label:26} NO REPLY (the server did not answer)")
            return
        if "error" in r:
            print(f"  {label:26} ERROR {json.dumps(r['error'])[:clip]}")
        else:
            print(f"  {label:26} OK    {json.dumps(r.get('result'))[:clip]}")

    sid = None
    try:
        print("=== handshake ===")
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": 1,
                         "clientCapabilities": {"fs": {"readTextFile": True,
                                                       "writeTextFile": True}}}})
        r1 = reply(1)
        report("initialize", r1, 260)
        info = ((r1 or {}).get("result") or {})
        caps = info.get("agentCapabilities") or {}
        print("  agent:", json.dumps(info.get("agentInfo"))[:160])
        print("  sessionCapabilities:", json.dumps(caps.get("sessionCapabilities"))[:200])
        ext = ((info.get("_meta") or {}).get("minimax-code/extensions") or {})
        if ext:
            print("  EXTENSION METHODS (%d):" % len(ext.get("methods") or []))
            for m in ext.get("methods") or []:
                print("     ", m)
            print("  EXTENSION NOTIFICATIONS (%d):" % len(ext.get("notifications") or []))
            for n in ext.get("notifications") or []:
                print("     ", n)

        print("\n=== session ===")
        send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
              "params": {"cwd": args.cwd, "mcpServers": []}})
        r2 = reply(2)
        report("session/new", r2, 200)
        res = (r2 or {}).get("result") or {}
        sid = res.get("sessionId")

        if sid:
            modes = (res.get("modes") or {}).get("availableModes") or []
            if modes:
                print("  MODES:")
                for m in modes:
                    print(f"     {m.get('id'):10} {m.get('name','')[:12]:12} "
                          f"{(m.get('description') or '')[:64]}")
            for c in res.get("configOptions") or []:
                vals = [o.get("value") for o in (c.get("options") or [])]
                print(f"  configOption {c.get('id')!r} current={c.get('currentValue')!r}")
                print(f"     options={json.dumps(vals)[:300]}")

            # A REAL id, so this tests the capability rather than argument validation.
            send({"jsonrpc": "2.0", "id": 3, "method": "mcode/session/goal/get",
                  "params": {"sessionId": sid}})
            report("mcode/session/goal/get", reply(3), 300)

            # The capability DSH cannot offer: enter plan mode.
            send({"jsonrpc": "2.0", "id": 4, "method": "session/set_mode",
                  "params": {"sessionId": sid, "modeId": "plan"}})
            report("session/set_mode -> plan", reply(4), 300)

            if args.prompt:
                send({"jsonrpc": "2.0", "id": 5, "method": "session/prompt",
                      "params": {"sessionId": sid,
                                 "prompt": [{"type": "text", "text": args.prompt}]}})
                report("session/prompt", reply(5, max(args.timeout, 60)), 400)
        else:
            print("  (no sessionId, so the session surface cannot be probed)")

        print("\n=== an unknown method (is the dispatcher well behaved?) ===")
        send({"jsonrpc": "2.0", "id": 9, "method": "probe/does-not-exist",
              "params": {}})
        report("probe/does-not-exist", reply(9, 8), 200)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        if err:
            print("\n=== stderr (first 5) ===")
            for line in err[:5]:
                print("   ", line[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
