"""The `notifications` half of mcode's ACP advertisement, and login state.

`initialize` returned a `_meta["minimax-code/extensions"]` block with a
`notifications` key the previous probe did not print. Notifications are the
server->client push half, which is what Rigma would actually render — so the list
matters as much as the methods.

Also reports whether this machine is authenticated, because every session method
answered `-32000 Authentication required` and that decides whether an ACP client
is even testable here.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

MCODE = r"C:\Users\amren\AppData\Roaming\npm\mcode.cmd"


def main() -> int:
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

    r = json.loads(out[0]) if out else {}
    meta = (r.get("result") or {}).get("_meta", {}).get("minimax-code/extensions", {})
    notes = meta.get("notifications") or []
    print("  ADVERTISED NOTIFICATIONS (%d):" % len(notes))
    for n in sorted(notes):
        print("    %s" % n)
    print("")
    print("  extensions version: %s" % meta.get("version"))

    # Is this machine authenticated? The ACP session methods all refused with
    # `Authentication required`, which decides whether an ACP client is testable.
    print("")
    print("  --- login state ---")
    home = os.path.expanduser("~")
    for cand in (os.path.join(home, ".minimax"), os.path.join(home, ".mcode"),
                 r"C:\Users\amren\.rigma\mcode\auth"):
        if os.path.isdir(cand):
            files = os.listdir(cand)
            print("  %s -> %d entries" % (cand, len(files)))
            for f in files[:6]:
                print("      %s" % f)
        else:
            print("  %s -> absent" % cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
