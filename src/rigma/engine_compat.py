"""Can this engine load this model? Ask the file, before launching (R3-ENG-2).

The failure this exists for:

    gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142.
                           should be in [0, 42)

Rigma planned `ternary-bonsai-2-27b`, downloaded it, launched the pinned engine, and
got that — an error naming an internal enum value, with no statement of what is
wrong or what to do. The fact needed to predict it was **in the file's own tensor
table the whole time**: `read_tensor_index` already walked every tensor and already
read the ggml type id (it sits between the dims and the offset, so it cannot be
skipped), and simply discarded it. Capturing it costs no extra I/O.

So a model/engine incompatibility is now detectable **without launching anything and
without downloading an engine**, from a few hundred KB of header.

WHY THIS IS NOT "JUST CHECK THE VERSION"
`GGML_TYPE_PQ2_0 = 142` is private to a third-party llama.cpp fork
(PrismML-Eng/llama.cpp, branch `prism`). Mainline's enum stops at
`GGML_TYPE_Q2_0 = 42` with `GGML_TYPE_COUNT = 43`. Type ids above the mainline range
are not "newer mainline" — they are somebody else's numbering, and no mainline build
however new will accept them. The checker therefore reports a type id it does not
recognise as a *fork or future* signal rather than guessing a minimum build.

Numbers verified against `ggml/include/ggml.h` on mainline `master` and on the
fork's `prism` branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ggml type ids, from mainline `ggml/include/ggml.h` (master). Deprecated and
# never-released numbers are included where they still appear in the wild, because a
# file using one is a real file someone will try to load.
KNOWN_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}

# The highest id mainline has ever defined, and its COUNT. A file using an id above
# `MAINLINE_MAX_TYPE` cannot be loaded by ANY mainline build, which is a different
# and more actionable statement than "your build is old".
MAINLINE_MAX_TYPE = 42
MAINLINE_TYPE_COUNT = 43

# Type ids known to come from a specific fork, so the message can name the project
# instead of saying "unknown". Each entry: (owner/repo, what it adds, why).
FORK_TYPES: dict[int, tuple[str, str, str]] = {
    142: ("PrismML-Eng/llama.cpp", "PQ2_0",
          "private ternary quant; also needs the fork's runtime Walsh-Hadamard "
          "activation transform, which is a behaviour difference no version "
          "number expresses"),
    143: ("PrismML-Eng/llama.cpp", "PTQ1_0",
          "private ternary quant, group 128; same fork and same transform"),
}


@dataclass
class Compatibility:
    """Whether an engine can load a model, and why not if it cannot."""
    ok: bool = True
    unknown_types: list[int] = field(default_factory=list)
    fork_types: dict[int, str] = field(default_factory=dict)
    reason: str = ""
    advice: str = ""

    @property
    def blocked(self) -> bool:
        return not self.ok

    def as_dict(self) -> dict:
        return {"ok": self.ok, "unknown_types": self.unknown_types,
                "fork_types": self.fork_types, "reason": self.reason,
                "advice": self.advice}


def type_name(tid: int) -> str:
    """The enum name for a type id, or a labelled 'unknown'."""
    if tid in KNOWN_TYPES:
        return KNOWN_TYPES[tid]
    if tid in FORK_TYPES:
        return FORK_TYPES[tid][1]
    return f"type {tid}"


def check_types(type_counts: dict | None, *,
                complete: bool = True) -> Compatibility:
    """Judge a tensor-type histogram against mainline's type table.

    `complete=False` (a truncated ranged read) returns ok=True with a caveat rather
    than a verdict: a partial histogram cannot show that a file is loadable, and
    reporting a false incompatibility would be worse than reporting none. Presence of
    a bad type is always trustworthy; absence from a partial read is not.
    """
    if not type_counts:
        return Compatibility(ok=True, reason="no tensor types were read")

    ids = sorted(type_counts)
    unknown = [t for t in ids if t not in KNOWN_TYPES and t not in FORK_TYPES]
    forks = {t: FORK_TYPES[t][0] for t in ids if t in FORK_TYPES}
    above = [t for t in ids if t > MAINLINE_MAX_TYPE]

    if not unknown and not forks and not above:
        return Compatibility(ok=True)

    if forks:
        names = ", ".join(f"{type_name(t)}={t}" for t in sorted(forks))
        repos = sorted(set(forks.values()))
        return Compatibility(
            ok=False,
            fork_types=forks,
            unknown_types=unknown,
            reason=(f"this model uses {names}, which "
                    f"{'are' if len(forks) > 1 else 'is'} private to "
                    f"{', '.join(repos)} — mainline llama.cpp numbers its types only "
                    f"up to {MAINLINE_MAX_TYPE} ({type_name(MAINLINE_MAX_TYPE)}), so "
                    f"NO mainline build can load this file"),
            advice=(f"install a build of {' or '.join(repos)} and point Rigma at it; "
                    f"upgrading the pinned mainline engine will not help"))

    if above:
        names = ", ".join(f"{type_name(t)}={t}" for t in above)
        return Compatibility(
            ok=False,
            unknown_types=unknown or above,
            reason=(f"this model uses {names}, above mainline's highest type id "
                    f"({MAINLINE_MAX_TYPE}); it was made for a different engine"),
            advice="install the engine this model was made for and point Rigma at it")

    names = ", ".join(f"{type_name(t)}={t}" for t in unknown)
    return Compatibility(
        ok=False,
        unknown_types=unknown,
        reason=(f"this model uses {names}, which this build of Rigma does not "
                f"recognise; it may need a newer engine"),
        advice="upgrade the engine pin, or install the engine this model was made for")


def check_gguf(path) -> Compatibility:
    """Read a GGUF's tensor types and judge them. Never raises.

    An unreadable or absent file is NOT reported as incompatible — that is an
    unknown, and claiming a model cannot load because its header could not be read
    would block files that are perfectly fine.
    """
    try:
        from .gguf_meta import read_tensor_index
        idx = read_tensor_index(path)
    except Exception as e:
        return Compatibility(ok=True, reason=f"could not read the model's types: {e}")
    return check_types(idx.type_counts, complete=idx.types_complete)
