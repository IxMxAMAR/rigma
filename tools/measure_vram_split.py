#!/usr/bin/env python
"""Measure the dedicated-vs-shared VRAM split, to re-test OPEN BUG #1.

`docs/HANDOFF.md` carries an open bug from 2026-07-13: on the loaded machine,
models generated at ~4 t/s instead of 25-57, with only ~10.4/16 GB of dedicated
VRAM used while ~5.8 GB spilled into "Shared GPU memory" *even though ~5 GB of
dedicated VRAM sat free*. Prefill stayed fast and decode collapsed, which is the
signature of weights living in system RAM instead of VRAM.

That analysis was written against "16 GB system RAM, RAM is the binding
constraint". The machine actually has **31.4 GB**, measured 2026-09-21, so the
reasoning needs redoing even if the symptom is real. This is the tool for that:
it reports the split, so the claim can be checked instead of inherited.

## Why a script and not a one-liner

The measurement is a SPLIT, not a number. "How much VRAM is used" is not the
question — the question is how much of it is the card's own memory and how much
is the driver quietly backing with system RAM. Windows exposes both through
`GPU Adapter Memory` performance counters, and the two must be read together and
attributed to the right adapter, which is more than fits comfortably on a command
line.

## Use

    # while a model is loaded and generating, in another shell
    python tools/measure_vram_split.py --label loaded

    # or let it drive the whole experiment (starts and stops the engine)
    python tools/measure_vram_split.py --label quiet --run-engine --yes

Sampling only, by default: starting an engine loads several GB and can make a
machine that is already busy much worse, so that needs saying out loud.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Windows reports each adapter as luid_0x00000000_0x<LUID>_phys_<N>. The names
# are opaque and are NOT stable across boots, so nothing here hardcodes one: the
# adapter is identified by which one is actually holding dedicated memory.
COUNTER_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
COUNTER_SHARED = r"\GPU Adapter Memory(*)\Shared Usage"

ROOT = Path(__file__).resolve().parent.parent


def _read() -> tuple[dict[str, dict[str, int]], float, float]:
    """One reading of every adapter, plus system RAM.

    Deliberately ONE subprocess for all of it. Reading each counter separately
    costs a PowerShell start each (~1s here), which made `--interval 1` sample
    every 2.5s — and a measurement whose sampling rate is a lie about its own
    resolution is worse than a slow one, because the samples get read as if they
    were evenly spaced. RAM rides along for the same reason: it is the variable
    the old bug blamed, so it has to be read at the SAME instant as the split.
    """
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        f"foreach($c in '{COUNTER_DEDICATED}','{COUNTER_SHARED}'){{"
        "(Get-Counter $c).CounterSamples|"
        "ForEach-Object{'{0}|{1}|{2}' -f $c,$_.InstanceName,$_.CookedValue}};"
        "$o=Get-CimInstance Win32_OperatingSystem;"
        "'ram|{0}|{1}' -f $o.TotalVisibleMemorySize,$o.FreePhysicalMemory"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=90).stdout
    except (OSError, subprocess.SubprocessError):
        return {}, 0.0, 0.0

    adapters: dict[str, dict[str, int]] = {}
    total_gb = free_gb = 0.0
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        which, name, value = parts
        try:
            if which == "ram":
                total_gb, free_gb = _gb(int(name) * 1024), _gb(int(value) * 1024)
                continue
            key = "dedicated" if which == COUNTER_DEDICATED else "shared"
            adapters.setdefault(name, {})[key] = int(float(value))
        except ValueError:
            continue
    return adapters, total_gb, free_gb


def _gb(n: int) -> float:
    return round(n / 2**30, 2)


def _engine_running() -> bool:
    """Whether Rigma thinks a model is up — `status`, not a process guess."""
    try:
        out = subprocess.run([str(ROOT / ".venv" / "Scripts" / "rigma.exe"),
                              "status"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "not running" not in out.lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="sample",
                    help="what this run is, e.g. 'quiet' or 'loaded'")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="how long to sample for (default 20)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--run-engine", action="store_true",
                    help="start the engine first, and stop it afterwards")
    ap.add_argument("--yes", action="store_true",
                    help="required with --run-engine: it loads several GB")
    ap.add_argument("--json", dest="json_out", default="",
                    help="also write the raw samples here")
    args = ap.parse_args()

    if args.run_engine and not args.yes:
        print("--run-engine loads several GB. Re-run with --yes if that is "
              "what you want.", file=sys.stderr)
        return 2

    started = False
    if args.run_engine:
        exe = ROOT / ".venv" / "Scripts" / "rigma.exe"
        print(f"starting the engine: {exe} up")
        # detached, so this script is not the engine's parent and does not take
        # it down when the sampling ends
        subprocess.Popen([str(exe), "up"], cwd=str(ROOT),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = True
        deadline = time.time() + 600
        while time.time() < deadline:
            if _engine_running():
                break
            time.sleep(5)
        else:
            print("the engine did not come up within 10 minutes", file=sys.stderr)
            return 1
        print("engine is up")

    rows: list[dict] = []
    print(f"\nsampling for {args.seconds:.0f}s, every {args.interval:.1f}s "
          f"— label: {args.label}")
    deadline = time.time() + args.seconds
    total = 0.0
    while time.time() < deadline:
        snap, total, free = _read()
        tot_ded = sum(v.get("dedicated", 0) for v in snap.values())
        tot_sha = sum(v.get("shared", 0) for v in snap.values())
        rows.append({"t": time.time(), "adapters": snap,
                     "total_dedicated": tot_ded, "total_shared": tot_sha,
                     "ram_free_gb": free})
        print(f"  dedicated {_gb(tot_ded):6.2f} GB   shared {_gb(tot_sha):6.2f} GB"
              f"   RAM free {free:5.2f}/{total:.1f} GB")
        # sleep only what is LEFT of the interval: each reading costs a
        # PowerShell start, so sleeping the full interval would silently stretch
        # the real spacing to ~2x and misreport the resolution
        elapsed = time.time() - rows[-1]["t"]
        if args.interval > elapsed:
            time.sleep(args.interval - elapsed)

    if not rows:
        print("no samples collected", file=sys.stderr)
        return 1

    ded = [r["total_dedicated"] for r in rows]
    sha = [r["total_shared"] for r in rows]
    gaps = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
    print(f"\n--- {args.label} ---")
    print(f"  dedicated : median {_gb(int(statistics.median(ded)))} GB  "
          f"max {_gb(max(ded))} GB")
    print(f"  shared    : median {_gb(int(statistics.median(sha)))} GB  "
          f"max {_gb(max(sha))} GB")
    print(f"  RAM free  : min {min(r['ram_free_gb'] for r in rows)} GB")
    # Stated, not implied. `Get-Counter` samples over an interval internally, so
    # a reading costs ~2.5s no matter what --interval asks for. Reporting the
    # ACTUAL spacing keeps a fast-spike result from being read as if it were
    # captured at a resolution it never had.
    if gaps:
        print(f"  samples   : {len(rows)} at a median spacing of "
              f"{statistics.median(gaps):.1f}s "
              f"(the floor is ~2.5s: Get-Counter measures over an interval)")
    else:
        print(f"  samples   : {len(rows)} (too few to report a spacing; raise "
              "--seconds for anything you intend to compare)")
    print("\n  The bug's signature is SHARED climbing by GBs while dedicated "
          "sits below the card's capacity.")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"label": args.label, "ram_total_gb": total,
                        "samples": rows}, indent=1), encoding="utf-8")
        print(f"  raw samples: {args.json_out}")

    if started:
        exe = ROOT / ".venv" / "Scripts" / "rigma.exe"
        print("\nstopping the engine")
        subprocess.run([str(exe), "stop"], cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
