"""Run the mcode adapter against the REAL mcode 0.5.4 CLI and a fake engine.

Not a pytest test: it needs a live mcode process and a live fake server. It is the
measurement that says whether the adapter's contract still holds on the installed
build, which is the one question a unit test against captured fixtures cannot answer.

    python tools/mcode_probe.py --port 11599
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigma import harness, harness_mcode          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11599)
    ap.add_argument("--model", default="local-test")
    ap.add_argument("--home", default="")
    ap.add_argument("--turn", default="",
                    help="run a real mcode turn with this prompt")
    args = ap.parse_args()

    if args.home:
        os.environ["RIGMA_HOME"] = args.home
    os.environ.setdefault("RIGMA_MCODE_KEY", "local")

    exe = harness_mcode.bin_path()
    print(f"exe       = {exe}")
    print(f"available = {harness_mcode.available()}")
    print(f"version   = {harness_mcode.backend_version()}")
    print(f"VERIFIED  = {harness_mcode.VERIFIED}")
    print(f"data_home = {harness_mcode.data_home()}")
    for row in harness.conformance():
        print(f"conformance[{row['name']}] = {row}")
    if not exe:
        print("FAIL: no mcode")
        return 2

    base = f"http://127.0.0.1:{args.port}/v1"
    pid, err = harness_mcode.ensure_provider(exe, base, args.model, 32768, 4096)
    print(f"ensure_provider -> pid={pid!r} err={err!r}")
    if err or not pid:
        return 3

    print("providers as mcode reports them:")
    for p in harness_mcode._providers(exe) or []:
        print("   ", json.dumps({k: p.get(k) for k in
                                 ("providerId", "baseUrl", "active", "enabled",
                                  "apiFormat")}))

    if not args.turn:
        return 0

    # A REAL turn. This is the part a fixture cannot check: whether the installed
    # build's event stream is still the shape the adapter was written against.
    state: dict = {}
    print(f"\n--- real turn: {args.turn!r} ---")
    n = 0
    for ev in harness_mcode.drive_turn(
            base_url=base, model=args.model, prompt=args.turn,
            cwd=str(Path.cwd()), timeout=180.0, state=state):
        n += 1
        print(f"  [{n:02d}] {ev.kind:<12} {str(getattr(ev, 'text', ''))[:160]!r}")
    print(f"events    = {n}")
    print(f"state     = {json.dumps(state)}")
    return 0 if n else 4


if __name__ == "__main__":
    raise SystemExit(main())
