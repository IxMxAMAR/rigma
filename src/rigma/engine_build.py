"""Which engine build is actually running — measured, not assumed (R3-ENG-1).

Rigma identified its engine by the **manifest's version string**. That is a claim
about what was downloaded, not a fact about what is on disk, and on the owner's
machine the two had diverged badly: four directories all under
`~/.rigma/engines/b9867/`, i.e. all claiming to be the pin, and one of them was a
third-party fork roughly 1,800 builds newer.

Measured with `--version`:

    b9867/cpu                  -> 9867 (152d337fa)
    b9867/vulkan               -> 9867 (152d337fa)
    b9867/rocm-mainline-b9867  -> 9867 (152d337fa)
    b9867/rocm                 -> 0.2.0-dev (build 10709, commit 9a9394a89)

`runtime.ensure_engine()` returned whichever binary existed, checking only for the
file and a `.ready` sentinel. It never ran `--version`. The consequences were not
cosmetic:

  * `server_ops.engine_version()` returned the MANIFEST string, so a calibration
    could be labelled `b9867` while having been measured on build 10709;
  * `bench.calibration_stale`'s engine check compared that manifest string against
    itself, so an engine change could never invalidate a calibration;
  * the same model loaded or failed depending on which directory happened to hold a
    newer binary, for reasons nothing surfaced.

This module replaces the claim with a measurement. `--version` is precise and works
on every binary in the release, including the `llama-fit-params` oracle.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_S = 30.0

# Two schemes are in the wild and they must both parse, because the pin uses one
# and any hand-installed or future build may use the other:
#
#   version: 9867 (152d337fa)
#   version: 0.2.0-dev (build 10709, commit 9a9394a89)
#
# Both are followed by a compiler line, which is captured because it is free and
# it distinguishes builds that agree on everything else.
_LEGACY_RE = re.compile(
    r"version:\s*(?P<ver>\d+)\s*\((?P<commit>[0-9a-f]{6,40})\)", re.I)
_MODERN_RE = re.compile(
    r"version:\s*(?P<ver>[0-9][\w.\-]*)"
    r"\s*\(build\s+(?P<build>\d+)(?:,\s*commit\s+(?P<commit>[0-9a-f]{6,40}))?\)",
    re.I)
_COMPILER_RE = re.compile(r"built with\s+(?P<what>.+)", re.I)

# Lenient fallbacks, used when neither known format matches.
#
# The research for this module (see _research/reports/engine-versioning-landscape.md)
# was explicit that format-matching must not be the only path: the format already
# differs between the two builds involved in this very defect, and `llama-bench
# --version` was only added in PR #28971, so older and future binaries vary. A
# `bNNNN` is `git rev-list --count HEAD`, i.e. a plain integer, so any 3-6 digit
# number in a version-looking line is a credible build number.
#
# These are deliberately NOT used before the exact patterns: a loose match could
# pick a number out of a commit hash or a date, so it is a last resort that still
# beats reporting "unknown".
_LOOSE_BUILD_RE = re.compile(r"\bb?(\d{3,6})\b")
_LOOSE_COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b", re.I)


@dataclass(frozen=True)
class EngineBuild:
    """What a binary reports about itself.

    `build` is the comparable number and is the point of this module: it is what
    decides whether a given GGUF can be loaded. The legacy `bNNNN` scheme puts it
    in the version field; the modern scheme states it separately.
    """
    raw: str = ""
    version: str = ""          # "9867" or "0.2.0-dev"
    build: int | None = None   # 9867, or 10709
    commit: str = ""
    compiler: str = ""
    ok: bool = False
    reason: str = ""

    @property
    def identity(self) -> str:
        """A short stable string for logging and for keying a cache.

        Prefers build+commit, because two builds can share a version string while
        differing in what they can load — which is the entire reason this exists.
        """
        if self.build is not None and self.commit:
            return f"b{self.build}+{self.commit[:9]}"
        if self.build is not None:
            return f"b{self.build}"
        return self.version or "unknown"

    def as_dict(self) -> dict:
        return {"raw": self.raw, "version": self.version, "build": self.build,
                "commit": self.commit, "compiler": self.compiler,
                "identity": self.identity, "ok": self.ok, "reason": self.reason}


def parse_version(text: str) -> EngineBuild:
    """Parse `--version` output. Pure, never raises, degrades to ok=False.

    Tries the modern scheme first: its version field can be a non-numeric string
    like `0.2.0-dev`, which the legacy pattern would not match at all, but a
    hypothetical future legacy-looking line should still be read as legacy.
    """
    raw = (text or "").strip()
    if not raw:
        return EngineBuild(reason="the engine printed nothing for --version")

    m = _MODERN_RE.search(raw)
    if m:
        return EngineBuild(
            raw=raw,
            version=m.group("ver"),
            build=int(m.group("build")),
            commit=(m.group("commit") or "").lower(),
            compiler=_compiler(raw),
            ok=True,
        )

    m = _LEGACY_RE.search(raw)
    if m:
        return EngineBuild(
            raw=raw,
            version=m.group("ver"),
            build=int(m.group("ver")),   # bNNNN: the version IS the build number
            commit=m.group("commit").lower(),
            compiler=_compiler(raw),
            ok=True,
        )

    return _loose(raw)


def _loose(raw: str) -> EngineBuild:
    """Last resort: pull a build number and a commit out of whatever was printed.

    Only reached when the output carries a version line at all — a binary that
    printed nothing version-like still reports unknown, because inventing a build
    from arbitrary text is how a calibration gets labelled with a build that was
    never measured.
    """
    if "version" not in raw.lower():
        return EngineBuild(raw=raw, reason="could not find a version line in the output")
    m = _LOOSE_BUILD_RE.search(raw)
    c = _LOOSE_COMMIT_RE.search(raw)
    return EngineBuild(
        raw=raw,
        version=m.group(1) if m else "",
        build=int(m.group(1)) if m else None,
        commit=c.group(1).lower() if c else "",
        compiler=_compiler(raw),
        ok=m is not None,
        reason="" if m else "found a version line but no build number in it",
    )


def _compiler(text: str) -> str:
    m = _COMPILER_RE.search(text)
    return m.group("what").strip() if m else ""


def read_build(exe: str | Path, *,
               popen=subprocess.run, timeout: float = _TIMEOUT_S) -> EngineBuild:
    """Run `--version` on a binary and report what it says.

    Never raises: a binary that cannot be executed, or that prints something
    unexpected, yields `ok=False` with a reason. Callers treat that as "unknown",
    which is the honest answer, rather than as a version.
    """
    exe = Path(exe)
    if not exe.exists():
        return EngineBuild(reason=f"{exe.name} does not exist")
    try:
        cp = popen([str(exe), "--version"], capture_output=True, text=True,
                   timeout=timeout)
    except Exception as e:                      # missing, not executable, hung
        return EngineBuild(reason=f"could not run {exe.name} --version: {e}")
    # Which stream carries it is not guaranteed and differs between the server and
    # the oracle, so read both rather than assuming.
    return parse_version(f"{cp.stdout or ''}\n{cp.stderr or ''}")


def matches_manifest(build: EngineBuild, expected_version: str) -> bool:
    """Whether a measured build is the one the manifest pins.

    Compares the BUILD NUMBER, not the raw version string, because the two schemes
    spell the same build differently and a string compare would report a false
    mismatch on every build whose release was re-tagged.
    """
    want = str(expected_version or "").strip()
    if not want or build.build is None:
        return False
    m = re.fullmatch(r"b?(\d+)", want)
    if m:
        return build.build == int(m.group(1))
    # A non-numeric manifest version can only be compared as a string.
    return build.version == want


# Running `--version` is a process spawn, and the answer cannot change unless the
# file does, so it is cached on (path, size, mtime). `--verify` and any status
# surface can then call this freely.
_BUILD_CACHE: dict[tuple, EngineBuild] = {}


def _file_key(exe: Path) -> tuple:
    """The file identity the memo is keyed on.

    A13d: this used `int(st_mtime)`, i.e. whole seconds, so a same-size
    replacement written inside one second served the OLD build's identity —
    `--version` was never re-run and a stale build keyed the KV cache and the
    calibration provenance. NANOSECOND mtime is the fix: it is free (no I/O
    beyond the stat the key already did) and NTFS records 100 ns, so the two
    writes are told apart. A content digest was the alternative and is the wrong
    trade: a full digest re-reads a ~1 GiB binary on every call, which is the
    engine-spawn cost this memo exists to avoid, and a bounded head/tail sample
    would be probabilistic — a different silent hole, not a closed one.

    RESIDUAL, deliberately: a replacement that PRESERVES the timestamp to the
    same 100 ns tick (a copy that restores mtime, e.g. `shutil.copy2` onto the
    same file) is still not detected. Detecting that needs content, and the cost
    above is why it is not paid on every call.
    """
    try:
        st = exe.stat()
        return (str(exe), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(exe), -1, -1)


def cached_build(exe: str | Path, *, popen=subprocess.run) -> EngineBuild:
    """`read_build`, memoised on the file's identity.

    Keyed on size and nanosecond mtime as well as path, so replacing the binary
    at the same path — which is exactly how a hand-installed fork arrives —
    invalidates the cache instead of reporting the old build forever. Whole-second
    mtime was not enough: a same-size swap inside one second served the old
    identity (A13d; see `_file_key`).
    """
    exe = Path(exe)
    key = _file_key(exe)
    hit = _BUILD_CACHE.get(key)
    if hit is not None:
        return hit
    got = read_build(exe, popen=popen)
    if got.ok:
        # An unreadable binary is not cached: a transient failure would otherwise
        # freeze "unknown" for the life of the process, which is the same
        # silent-degradation trap the identity cache in bench.py guards against.
        _BUILD_CACHE[key] = got
    return got


def verify_engine(exe: str | Path, expected_version: str, *,
                  popen=subprocess.run) -> tuple[EngineBuild, bool]:
    """The engine on disk, and whether it is the one the manifest pins.

    Returns (build, matches). A build that cannot be read is NOT reported as a
    match — unknown is not a match, and treating it as one is how a fork goes
    unnoticed.
    """
    got = cached_build(exe, popen=popen)
    return got, matches_manifest(got, expected_version)


def same_build(a: str, b: str) -> bool:
    """Whether two engine strings name the same build.

    Engine strings are written by two different producers and must compare equal
    across both, or a calibration measured before R3-ENG-6 would be discarded the
    moment R3-ENG-6 started recording a fuller answer:

      * `"b9867"`             — the manifest version, recorded by older entries
      * `"b9867+152d337fa"`   — the measured identity recorded now
      * `"b10709+9a9394a89"`  — the fork, which must NOT match either of the above

    Compared by build number and, when both sides carry one, by commit. A commit
    mismatch on the same build number is treated as a DIFFERENT build, because a
    re-tagged release and a fork can share the counter — that is the whole reason
    identity carries the commit at all.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if a == b:
        return True
    if not a or not b:
        return False

    def split(s: str) -> tuple[int | None, str]:
        head, _, tail = s.partition("+")
        m = re.fullmatch(r"b?(\d+)", head.strip())
        return (int(m.group(1)) if m else None), tail.strip().lower()

    na, ca = split(a)
    nb, cb = split(b)
    if na is None or nb is None:
        return False
    if na != nb:
        return False
    # Both name a commit: they must agree.
    if ca and cb:
        return ca.startswith(cb) or cb.startswith(ca)
    # Only one names a commit (the older, shorter form): the build number is all we
    # have to go on, and refusing to match would throw away a valid calibration.
    return True
