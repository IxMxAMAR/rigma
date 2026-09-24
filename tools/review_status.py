"""Fail-safe ledger for a subagent review fan-out.

A review that runs as N independent agents has one failure mode that matters:
an agent dies half way -- provider 500, rate limit, truncated turn, killed
process -- and leaves a *file* behind.  A file that exists is not a result.  So
every agent is required to terminate its findings file with a sentinel line, and
this tool treats a file without that sentinel as absent, not as done.

It also checks *coverage*: each agent must write one ``coverage:`` line per file
in its scope, naming the lines it read.  An area whose coverage lines omit a
scoped file is reported as a gap, which catches the quiet failure where an agent
reports "area clean" without ever opening one of the files it was given.

The point is that a half-finished review is resumable by re-dispatching only
what is missing, instead of re-running everything or silently reporting partial
work as complete.

Usage::

    python tools/review_status.py                 # table for every planned area
    python tools/review_status.py --missing       # ids to re-dispatch, one per line
    python tools/review_status.py --check         # exit 1 unless every area is complete
    python tools/review_status.py --json          # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SENTINEL = re.compile(r"^<!--\s*END\b.*-->\s*$")
SEVERITY = re.compile(r"^\s*-\s*severity:\s*(CRITICAL|HIGH|MEDIUM|LOW|NIT)\b", re.I | re.M)
HEADING = re.compile(r"^###\s+\S")
SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NIT")
NOT_READ = "NOT READ"
MISSING, EMPTY, TRUNCATED, COMPLETE = "MISSING", "EMPTY", "TRUNCATED", "COMPLETE"


@dataclass
class Area:
    """One planned review area: an id, a slug, and the files it must cover."""

    id: str
    slug: str
    files: list[str]

    @property
    def rel_path(self) -> str:
        return f"docs/review/findings/{self.id}-{self.slug}.md"


@dataclass
class Status:
    area: Area
    state: str
    size: int = 0
    findings: int = 0
    counts: dict = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state == COMPLETE and not self.gaps


def parse_plan(text: str) -> list[Area]:
    """Parse a plan file: ``id<TAB>slug<TAB>file;file;file``, ``#`` comments allowed."""
    areas: list[Area] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        files = [f for f in (parts[2].split(";") if len(parts) > 2 else []) if f]
        areas.append(Area(parts[0], parts[1], files))
    return areas


def load_plan(path: Path) -> list[Area]:
    return parse_plan(path.read_text(encoding="utf-8"))


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _coverage_gaps(files: list[str], text: str) -> list[str]:
    """Planned files that no ``coverage:`` line accounts for, or that say NOT READ."""
    lines = [ln for ln in text.splitlines() if ln.strip().lower().startswith("coverage:")]
    gaps: list[str] = []
    for f in files:
        want = _norm(f)
        if want.endswith("/"):
            hits = [ln for ln in lines if want.rstrip("/") in _norm(ln)]
        else:
            hits = [ln for ln in lines if want in _norm(ln)]
        if not hits:
            gaps.append(f"{f} (not mentioned in any coverage line)")
        elif all(NOT_READ in ln.upper() for ln in hits):
            gaps.append(f"{f} (marked NOT READ)")
    return gaps


def scan(root: Path, areas: list[Area]) -> list[Status]:
    out: list[Status] = []
    for area in areas:
        path = root / area.rel_path
        if not path.is_file():
            out.append(Status(area, MISSING))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        size = len(text.encode("utf-8"))
        if not text.strip():
            out.append(Status(area, EMPTY, size=size))
            continue
        tail = [ln for ln in text.splitlines() if ln.strip()]
        if not tail or not SENTINEL.match(tail[-1].strip()):
            out.append(Status(area, TRUNCATED, size=size))
            continue
        counts = {s: 0 for s in SEVERITIES}
        for m in SEVERITY.finditer(text):
            counts[m.group(1).upper()] += 1
        out.append(
            Status(
                area,
                COMPLETE,
                size=size,
                findings=sum(1 for ln in text.splitlines() if HEADING.match(ln)),
                counts=counts,
                gaps=_coverage_gaps(area.files, text),
            )
        )
    return out


def render(statuses: list[Status]) -> str:
    lines = [
        f"{'id':<4} {'area':<22} {'state':<10} {'find':>4} {'C':>2} {'H':>2} {'M':>2} "
        f"{'L':>2} {'N':>2}  gaps",
        "-" * 92,
    ]
    for s in statuses:
        c = s.counts
        gap = "; ".join(s.gaps)
        lines.append(
            f"{s.area.id:<4} {s.area.slug:<22} {s.state:<10} {s.findings:>4} "
            f"{c.get('CRITICAL', 0):>2} {c.get('HIGH', 0):>2} {c.get('MEDIUM', 0):>2} "
            f"{c.get('LOW', 0):>2} {c.get('NIT', 0):>2}  {gap}"
        )
    done = sum(1 for s in statuses if s.ok)
    total_f = sum(s.findings for s in statuses)
    total = {k: sum(s.counts.get(k, 0) for s in statuses) for k in SEVERITIES}
    lines.append("-" * 92)
    lines.append(
        f"{done}/{len(statuses)} areas complete and fully covered; {total_f} findings "
        f"({total['CRITICAL']}C/{total['HIGH']}H/{total['MEDIUM']}M/{total['LOW']}L/{total['NIT']}N)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="fork root (default: cwd)")
    ap.add_argument("--plan", default="docs/review/plan.tsv")
    ap.add_argument("--missing", action="store_true", help="print ids needing (re)dispatch")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every area is ok")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    plan_path = root / args.plan
    if not plan_path.is_file():
        print(f"no plan at {plan_path}", file=sys.stderr)
        return 2
    statuses = scan(root, load_plan(plan_path))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": s.area.id,
                        "slug": s.area.slug,
                        "state": s.state,
                        "ok": s.ok,
                        "size": s.size,
                        "findings": s.findings,
                        "counts": s.counts,
                        "gaps": s.gaps,
                    }
                    for s in statuses
                ],
                indent=2,
            )
        )
    elif args.missing:
        for s in statuses:
            if not s.ok:
                print(f"{s.area.id}\t{s.area.slug}\t{s.state}\t{'; '.join(s.gaps)}")
    else:
        print(render(statuses))

    if args.check and any(not s.ok for s in statuses):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
