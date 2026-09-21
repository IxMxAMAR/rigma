"""Measure the path-repair fix against the REAL recorded failures.

Method, and why it is this and not something easier:

- The inputs are the actual `view_images` calls recorded in
  `~/.rigma/runs/*/actions.jsonl`, replayed against the real folders they
  named (`D:\\Good Stuff` still holds its 2,286 files).
- The two arms are `git show HEAD:src/rigma/tools.py` (before this change) and
  the working tree (after). Both are the same code except for one function, so
  the difference measured is THIS change and not the July-to-HEAD drift.
- Replaying the corpus against HEAD alone would have credited this change with
  fixes that shipped weeks ago — two recorded `list_directory` failures
  reproduce as successes on unmodified HEAD, because the server that produced
  them was a stale build.
- Resolution only, not full tool execution: `_resolve_image` is the function
  that changed, and running the tool would base64-encode thousands of real
  PNGs to answer a question about path resolution.
"""
import glob
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))

from rigma import tools as new  # noqa: E402


def _old_module():
    src = subprocess.run(["git", "show", "HEAD:src/rigma/tools.py"], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8").stdout
    if not src.strip():
        raise SystemExit("could not read HEAD:src/rigma/tools.py")
    path = os.path.join(tempfile.mkdtemp(), "tools_before.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("tools_before", path)
    mod = importlib.util.module_from_spec(spec)
    # register BEFORE exec: dataclasses resolves the module through
    # sys.modules to read annotations, and a module that is not there yet
    # makes @dataclass raise AttributeError on None
    sys.modules["tools_before"] = mod
    spec.loader.exec_module(mod)
    return mod


old = _old_module()
CTX = {"workspace": REPO, "has_vision": True, "run_id": "replay"}


def resolved(mod, paths):
    """How many of these paths that module can resolve to a real image."""
    n = 0
    for ps in paths:
        try:
            p, _ = mod._resolve_image(ps, CTX)
        except Exception:
            p = None
        if p:
            n += 1
    return n


def calls():
    """Every recorded view_images call, with the paths it actually passed."""
    found = []
    for aj in sorted(glob.glob(os.path.join(
            os.path.expanduser("~"), ".rigma", "runs", "*", "actions.jsonl"))):
        for line in open(aj, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            if a.get("tool") != "view_images":
                continue
            try:
                args = json.loads(a.get("args") or "{}")
            except Exception:
                continue
            paths = args.get("paths") or []
            if isinstance(paths, str):
                paths = [paths]
            if paths:
                found.append((a.get("ok"), list(paths)))
    return found


rows = calls()
print(f"recorded view_images calls with paths: {len(rows)}")
total_paths = sum(len(p) for _, p in rows)
bo, bn = 0, 0               # paths resolved, before / after
zero_before = 0             # calls that resolved NOTHING = "no valid images"
rescued = []                # ... that now resolve at least one
still_zero = 0              # ... that still resolve nothing (correct: no match)
regressed = []              # calls that resolved something and now do not
for ok, paths in rows:
    o, n = resolved(old, paths), resolved(new, paths)
    bo += o
    bn += n
    if o == 0:
        zero_before += 1
        (rescued if n > 0 else [(None, None, None)]).append((paths, n, len(paths)))
        if n == 0:
            still_zero += 1
    elif n == 0:
        regressed.append(paths)

rescued = [r for r in rescued if r[0] is not None]
print(f"paths resolved          before {bo}/{total_paths}"
      f"   after {bn}/{total_paths}   ({bn - bo:+d})")
print(f"calls that resolved NOTHING before: {zero_before}"
      f"   of those now working: {len(rescued)}   still nothing: {still_zero}")
print(f"regressions (worked before, fails now): {len(regressed)}")
print()
for paths, n, total in rescued:
    print(f"  {n} of {total} recovered;  asked for: "
          + ", ".join(os.path.basename(p) for p in paths[:3]))
