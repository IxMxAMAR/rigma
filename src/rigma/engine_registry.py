"""Engines that are not the pin, as first-class citizens (R3-ENG-3).

Rigma had exactly one engine: whatever the manifest pinned. That is the right
default and it should stay the default. But some models cannot run on mainline
llama.cpp **at all**, and the owner hit precisely that:

  * `Ternary-Bonsai-2-27B-PQ2_0` uses `GGML_TYPE_PQ2_0 = 142`, private to
    `PrismML-Eng/llama.cpp` branch `prism`. Mainline's enum stops at
    `GGML_TYPE_Q2_0 = 42`. No mainline build, however new, will load it.
  * The fork also needs a runtime Walsh-Hadamard activation transform, which is a
    *behaviour* difference — a version number cannot express it.

With no way to represent that, the owner installed the fork **by hand** into
`~/.rigma/engines/b9867/rocm/` and wrote an `ENGINE-PROVENANCE.txt` beside it
explaining what it was and how to undo it. It worked only because
`runtime.ensure_engine` trusts a `.ready` sentinel and never checks what the binary
actually is. That is a workaround, not support: Rigma could not name it, could not
verify it, could not restore the build it displaced, and would have reported the
wrong engine version in calibration forever.

This module makes that situation expressible without weakening the mainline path.

DESIGN CONSTRAINTS, and why
---------------------------
* **`ENGINE_URL_ALLOWLIST` is not touched.** Auto-downloading third-party binaries
  is a trust decision for the user, not a side effect of wanting to run a model. A
  registered engine is one the user obtained and vouched for.
* **The pin stays the default.** Registration is opt-in and additive; a machine that
  never registers anything behaves exactly as before.
* **Capability beats version.** A build is selected because it can load *this*
  model's tensor types, which is a fact we can check, rather than because its
  version string compares favourably, which is a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_NAME = "engines-custom.json"
SCHEMA = 1


def registry_path(home: Path | None = None) -> Path:
    if home is not None:
        return Path(home) / REGISTRY_NAME
    from .runtime import rigma_home
    return rigma_home() / REGISTRY_NAME


@dataclass
class CustomEngine:
    """A user-supplied engine build.

    `name` is the stable handle used on the command line. `path` points at the
    `llama-server` binary (its directory is treated as the build root, so
    `llama-fit-params` and the DLLs are found alongside it).
    """
    name: str
    path: str
    backend: str = ""
    source: str = ""              # e.g. "PrismML-Eng/llama.cpp branch 'prism'"
    note: str = ""                # free text: why it is here
    # ggml type ids this build accepts, as measured or declared. Empty means
    # "unknown", NOT "accepts nothing" — see `accepts`.
    types: list[int] = field(default_factory=list)
    types_known: bool = False

    @property
    def exe(self) -> Path:
        return Path(self.path)

    @property
    def root(self) -> Path:
        return Path(self.path).parent

    def accepts(self, type_ids) -> bool | None:
        """Whether this build can load a model using `type_ids`.

        True/False when this build's type table is known, **None when it is not** —
        an engine whose capabilities were never recorded must not be reported as
        unable to load something. Unknown is not a refusal.
        """
        if not self.types_known:
            return None
        have = set(self.types)
        return all(int(t) in have for t in type_ids)

    def as_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "backend": self.backend,
                "source": self.source, "note": self.note,
                "types": sorted(self.types), "types_known": self.types_known}


def load(home: Path | None = None) -> dict[str, CustomEngine]:
    """Registered engines by name. A corrupt file reads as empty, never raises.

    A registry that cannot be parsed must not stop Rigma starting: the worst case of
    ignoring it is that the pin is used, which is the default anyway.
    """
    p = registry_path(home)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    engines = raw.get("engines")
    if not isinstance(engines, dict):
        return {}
    out: dict[str, CustomEngine] = {}
    for name, e in engines.items():
        if not (isinstance(name, str) and isinstance(e, dict) and e.get("path")):
            continue
        types = e.get("types")
        out[name] = CustomEngine(
            name=name,
            path=str(e["path"]),
            backend=str(e.get("backend") or ""),
            source=str(e.get("source") or ""),
            note=str(e.get("note") or ""),
            types=[int(t) for t in types] if isinstance(types, list) else [],
            types_known=bool(e.get("types_known")),
        )
    return out


def save(engines: dict[str, CustomEngine], home: Path | None = None) -> None:
    p = registry_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA,
               "engines": {n: e.as_dict() for n, e in sorted(engines.items())}}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(p)


def register(engine: CustomEngine, home: Path | None = None) -> None:
    engines = load(home)
    engines[engine.name] = engine
    save(engines, home)


def forget(name: str, home: Path | None = None) -> bool:
    engines = load(home)
    if name not in engines:
        return False
    del engines[name]
    save(engines, home)
    return True


def for_backend(backend: str, home: Path | None = None) -> list[CustomEngine]:
    """Registered engines usable for a compute backend, best first.

    A registered engine with no declared backend is a candidate for any backend —
    the owner installed it knowing what it was for, and refusing to consider it
    because a field was left blank would be unhelpful.
    """
    out = [e for e in load(home).values() if not e.backend or e.backend == backend]
    return sorted(out, key=lambda e: e.name)


def select(type_ids, backend: str, *,
           home: Path | None = None) -> tuple[CustomEngine | None, str]:
    """Pick a registered engine that can load a model, or explain why none can.

    Returns (engine, reason). `engine` is None when nothing registered fits, and
    `reason` is then a sentence for the user. A registered engine whose capabilities
    are unknown is *offered* rather than silently chosen: it might be the right one,
    and it might not, and the user is the only one who can say.
    """
    ids = [int(t) for t in (type_ids or [])]
    cands = for_backend(backend, home)
    if not cands:
        return None, f"no engine is registered for the {backend} backend"

    fits = [e for e in cands if e.accepts(ids) is True]
    if fits:
        # Prefer the one declaring the most types: a fork that also carries mainline's
        # table is a safer default than one that only knows its own additions.
        fits.sort(key=lambda e: (-len(e.types), e.name))
        return fits[0], f"{fits[0].name} accepts every type this model uses"

    unknown = [e for e in cands if e.accepts(ids) is None]
    if unknown:
        return None, (f"{len(unknown)} registered engine(s) have no recorded type "
                      f"table, so none can be ruled in or out: "
                      f"{', '.join(e.name for e in unknown)}")

    return None, ("every registered engine for the "
                  f"{backend} backend rejects one of this model's types")


def restore_hint(name: str, home: Path | None = None) -> str:
    """A one-line 'how do I undo this' for a registered engine, for status output."""
    e = load(home).get(name)
    if e is None:
        return ""
    what = e.source or "a hand-installed build"
    return f"{name} ({what})"
