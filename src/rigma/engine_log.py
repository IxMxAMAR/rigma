"""Things llama-server says once, quietly, that change what you are running.

The engine reports several decisions as a single WARN line at load and then
never mentions them again. They sit in a log nobody opens while the behaviour
they describe persists for the whole session.

The one that prompted this module, found in this machine's own logs:

    W srv load_model: cache_reuse is not supported by this context,
                      it will be disabled

Rigma passes `--cache-reuse 256`. On a hybrid / DeltaNet architecture the
engine cannot KV-shift, so it turns the flag off. Everything still works; edits
that are not a clean prefix just reprocess from scratch, silently, forever. The
flag being *set* and the flag being *in effect* are different facts, and only
the log knew which one was true.

Pure function over log text, so the patterns are testable without an engine.
"""
from __future__ import annotations

import re

# (id, compiled pattern, severity, what it means for the user)
#
# `confirmed` marks patterns actually observed in this machine's logs. The
# others come from llama.cpp's source and upstream reports and are matched
# best-effort — a pattern that never fires is invisible, which is the right
# failure mode for a diagnostic.
_PATTERNS: list[tuple[str, re.Pattern, str, str, bool]] = [
    ("cache_reuse_disabled",
     re.compile(r"cache[_ ]reuse is not supported", re.I),
     "warn",
     "KV-shift reuse is OFF — this model's architecture does not support it, "
     "so the engine disabled --cache-reuse. Edits that are not a clean prefix "
     "reprocess from scratch. Nothing to fix: it is an architecture limit, not "
     "a setting.",
     True),
    ("swa_disabled",
     # AUDIT 02-6: this used to be `n_swa\s*=\s*0|swa.*will be disabled`. The
     # first alternative is a bare `key = value` parameter dump, not a warning,
     # so a non-SWA model's hparams line reported a no-op for a flag Rigma never
     # passes (grep: `--swa-full` appears nowhere in the launch path). Require
     # the warning shape — `swa` AND `disabl` on the same line.
     re.compile(r"swa.*disabl", re.I),
     "info",
     "The engine reports no sliding window, so it disabled SWA. Nothing to fix: "
     "it is a property of this architecture, not a setting. On hybrid models the "
     "reprocessing trigger is the recurrent component, which SWA does not touch.",
     False),
    ("checkpoint_cascade",
     re.compile(r"erasing.*checkpoint|checkpoint.*eras|no checkpoint found",
                re.I),
     "warn",
     "Context checkpoints were discarded, which forces a full prompt "
     "reprocess. Hybrid and recurrent layers cannot be rewound, so when no "
     "checkpoint matches the resume position the engine drops them all. "
     "Lowering --checkpoint-min-step toward your turn length makes this rarer.",
     False),
    ("kv_cache_full",
     re.compile(r"KV cache is full|context (?:is )?full|slot context shift",
                re.I),
     "warn",
     "The context filled and the engine shifted or dropped tokens. The oldest "
     "part of the conversation is gone from the model's view.",
     False),
    ("fallback_template",
     re.compile(r"chat template.*not found|using default chat template", re.I),
     "warn",
     "No chat template in the gguf, so the engine is using a generic one. The "
     "model is being prompted in a format it was not trained on.",
     False),
]


def findings(log_text: str) -> list[dict]:
    """Significant one-off engine statements found in a log.

    Deduplicated: these fire once per load, and a log spanning several restarts
    would otherwise report the same fact many times. Ordered as listed, so the
    confirmed and consequential ones come first.
    """
    out = []
    for key, pat, severity, message, confirmed in _PATTERNS:
        hits = [ln.strip() for ln in (log_text or "").splitlines()
                if pat.search(ln)]
        if not hits:
            continue
        out.append({
            "id": key,
            "severity": severity,
            "message": message,
            "count": len(hits),
            "confirmed_here": confirmed,
            # the last occurrence: on a log covering several launches it is the
            # one describing the engine currently running
            "example": hits[-1][:300],
        })
    return out


# ---------------------------------------------------------------------------
# A17/S2 — the engine's own load accounting
# ---------------------------------------------------------------------------
#
# `findings()` answers "did the engine quietly change a setting". It does not
# answer the other half of the same question: "what did the engine actually
# allocate". That matters because on Windows/WDDM an allocation that does not
# fit in VRAM SUCCEEDS out of system RAM at a fraction of the speed and prints
# no error — the plan and the reality diverge silently, and the engine's own
# load log is the only place the real numbers appear.
#
# The formats below are copied from this machine's real load log of the owner's
# 27B hybrid, `.scratch/prism-v.log` (second load, lines 4485-4672), whitespace
# preserved:
#
#   0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU
#   0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB
#   0.01.522.676 I load_tensors:        ROCm0 model buffer size =  6539.67 MiB
#   0.04.285.857 I llama_context: n_seq_max             = 1
#   0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB
#   0.04.369.729 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB
#   0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB
#   0.04.417.938 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB
#   0.04.417.939 I sched_reserve: graph splits = 2
#
# The same file's first load (lines 2159-2340) is a FITTING pass with
# `ROCm0 model buffer size = 0.00 MiB`. Two loads in one file is normal, so the
# parser segments on the `offloaded N/M layers` marker and callers report the
# LAST segment (the engine that is actually up), never the first.
#
# Pure text in, plain dicts out: no engine call, no file read, no I/O.

# A new load starts at this marker. The neighbouring lines "offloading output
# layer to GPU" and "offloading 63 repeating layers to GPU" also contain
# "offload" but not the `offloaded N/M` shape, so they are not boundaries.
_LOAD_MARKER = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")

# One buffer line. The label is a single token — `CPU`, `CPU_Mapped`, `ROCm0`,
# `ROCm_Host`, `CUDA0`, `Vulkan0`, `Metal`, ... The kind is one of the four the
# engine prints; `output buffer size` is deliberately not one of them, so the
# `llama_context: ... output buffer size` line is ignored.
_BUFFER = re.compile(
    r"(?P<label>[A-Za-z][\w.]*)\s+"
    r"(?P<kind>model|KV|RS|compute)\s+buffer size\s*=\s*"
    r"(?P<mb>\d+(?:\.\d+)?)\s*MiB")

_N_SEQ_MAX = re.compile(r"n_seq_max\s*=\s*(\d+)")
_GRAPH_SPLITS = re.compile(r"graph splits\s*=\s*(\d+)")

_KIND_KEY = {
    "model": "model_buffers",
    "KV": "kv_buffers",
    "RS": "rs_buffers",
    "compute": "compute_buffers",
}
_BUFFER_KEYS = ("model_buffers", "kv_buffers", "rs_buffers", "compute_buffers")


def _is_host_buffer(label: str) -> bool:
    """True when a buffer label names host RAM rather than device VRAM.

    `CPU` and `CPU_Mapped` are system RAM (`CPU_Mapped` is the mmap'd weight
    copy); the `<device>_Host` labels (`ROCm_Host`, `Vulkan_Host`, `CUDA_Host`)
    are the pinned host-side staging buffers `sched_reserve` sets aside. Every
    other label (`ROCm0`, `CUDA0`, `Vulkan0`, `Metal`) is a device buffer.

    Confirmed for `CPU_Mapped` / `ROCm0` / `ROCm_Host` in prism-v.log:4485-4670.
    The CUDA/Vulkan/Metal spellings follow the same llama.cpp buffer-type naming
    but were not observed in this machine's log; a synthetic test pins the
    classification (UNVERIFIED against llama.cpp source in this run).
    """
    return label in ("CPU", "CPU_Mapped") or label.endswith("_Host")


def _empty_load() -> dict:
    """A load with nothing known about it. `found` is False, never a fake 0."""
    return {
        "found": False,
        "offloaded": None,          # "65/65" as printed
        "offloaded_layers": None,   # (65, 65) for arithmetic
        "model_buffers": [],        # [{"label", "mb", "host"}, ...]
        "kv_buffers": [],
        "rs_buffers": [],
        "compute_buffers": [],
        "n_seq_max": None,
        "graph_splits": None,
    }


def parse_loads(log_text: str) -> list[dict]:
    """Every model load in the log, in order.

    One entry per `offloaded N/M layers to GPU` line; buffer, `n_seq_max` and
    `graph splits` lines attach to the load marker that precedes them. A log
    covering a fitting pass and the real load yields two entries — the last is
    the engine that is up now.

    A log tail that starts after the marker (no `offloaded` line but buffer
    lines present) still yields one entry, so the data is not dropped.
    """
    loads: list[dict] = []
    cur: dict | None = None

    for line in (log_text or "").splitlines():
        marker = _LOAD_MARKER.search(line)
        if marker:
            cur = _empty_load()
            cur["found"] = True
            cur["offloaded"] = f"{marker.group(1)}/{marker.group(2)}"
            cur["offloaded_layers"] = (int(marker.group(1)),
                                       int(marker.group(2)))
            loads.append(cur)
            continue

        buf = _BUFFER.search(line)
        if buf:
            if cur is None:
                cur = _empty_load()
                cur["found"] = True
                loads.append(cur)
            label = buf.group("label")
            cur[_KIND_KEY[buf.group("kind")]].append({
                "label": label,
                "mb": float(buf.group("mb")),
                "host": _is_host_buffer(label),
            })
            continue

        if cur is None:
            continue
        m = _N_SEQ_MAX.search(line)
        if m:
            cur["n_seq_max"] = int(m.group(1))
            continue
        m = _GRAPH_SPLITS.search(line)
        if m:
            cur["graph_splits"] = int(m.group(1))

    return loads


def parse_load(log_text: str) -> dict:
    """The load the engine is running now: the last one in the log.

    Returns an explicit not-found load (`found is False`) when the log has no
    load lines, so a caller cannot mistake "unknown" for "allocated 0 MiB".
    """
    loads = parse_loads(log_text)
    return loads[-1] if loads else _empty_load()


def _sum_buffers(load: dict, *, host: bool) -> float:
    """Total MiB of the buffers whose `host` flag equals `host`."""
    return sum(b["mb"] for key in _BUFFER_KEYS
               for b in load.get(key, []) if b["host"] is host)


def _unknown_plan(expected_vram_mb, expected_splits: int) -> dict:
    return {
        "known": False,
        "actual_vram_mb": None,
        "host_ram_mb": None,
        "expected_vram_mb": expected_vram_mb,
        "divergence_mb": None,
        "divergence_pct": None,
        "graph_splits": None,
        "expected_splits": expected_splits,
        "unexpected_splits": False,
        "detail": ("no buffer lines in the engine log: what the engine "
                   "allocated is UNKNOWN, not zero — a silent 0 would read as "
                   "'it fits'."),
    }


def compare_plan(parsed, expected_vram_mb: float,
                 expected_splits: int = 1) -> dict:
    """What the engine allocated vs what the plan charged it for.

    `actual_vram_mb` sums the DEVICE buffers only:

        device model buffers + KV + RS + device compute

    That is the set that lives in VRAM and is therefore the set the plan's
    charge is supposed to cover. `CPU_Mapped`, `CPU` and every `<device>_Host`
    buffer are host RAM (weights mapped from disk, pinned staging buffers) and
    are reported separately as `host_ram_mb`; counting them as VRAM would
    overstate the overrun and could turn a real one into a false alarm.

    `unexpected_splits` is True when the log's `graph splits` exceeds
    `expected_splits`: the scheduler split the graph across more than one
    backend, so some operations ran somewhere the plan did not assume and the
    charge was computed against a graph the engine did not build. (Reasoning
    from llama.cpp's scheduler semantics, not measured in this run —
    PREDICTION.)

    `parsed` may be one load from `parse_loads` or the whole list (the last
    entry is used). When nothing was parsed the result is explicitly
    `known is False` with `actual_vram_mb is None`, never a 0 that reads as
    "it fits".
    """
    if isinstance(parsed, list):
        parsed = parsed[-1] if parsed else None
    if not parsed or not parsed.get("found"):
        return _unknown_plan(expected_vram_mb, expected_splits)
    if not any(parsed.get(key) for key in _BUFFER_KEYS):
        return _unknown_plan(expected_vram_mb, expected_splits)

    actual = _sum_buffers(parsed, host=False)
    host_ram = _sum_buffers(parsed, host=True)
    divergence = actual - expected_vram_mb
    pct = (divergence / expected_vram_mb * 100.0) if expected_vram_mb else None
    splits = parsed.get("graph_splits")
    unexpected = splits is not None and splits > expected_splits

    if splits is None:
        why = "the log did not report graph splits; backend assignment unknown."
    elif unexpected:
        why = ("graph splits = %d > %d: some operations were assigned to a "
               "different backend than the plan assumed." % (splits,
                                                             expected_splits))
    else:
        why = "graph splits = %d, matching the plan." % splits

    detail = (
        "engine allocated %.2f MiB of device VRAM (device model + KV + RS + "
        "device compute); host RAM %.2f MiB excluded. Plan charged %.2f MiB; "
        "divergence %+.2f MiB (%s). %s" % (
            actual, host_ram, expected_vram_mb, divergence,
            "n/a" if pct is None else "%+.1f%%" % pct, why))

    return {
        "known": True,
        "actual_vram_mb": actual,
        "host_ram_mb": host_ram,
        "expected_vram_mb": expected_vram_mb,
        "divergence_mb": divergence,
        "divergence_pct": pct,
        "graph_splits": splits,
        "expected_splits": expected_splits,
        "unexpected_splits": unexpected,
        "detail": detail,
    }
