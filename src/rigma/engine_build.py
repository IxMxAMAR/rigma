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

    return EngineBuild(raw=raw, reason="could not find a version line in the output")


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
