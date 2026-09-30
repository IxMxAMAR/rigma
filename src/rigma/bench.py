from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

from . import hwid
from .atomicio import atomic_write_json, atomic_write_text
from .models import ComboFlags, RunPlan
from .runtime import launch_server, rigma_home


class BenchResult(BaseModel):
    pp_tps: float
    tg_tps: float
    prompt_tokens: int
    gen_tokens: int
    # D1/S3: the DEPTH dimension. `depth` is the window occupancy the number was
    # measured at, and `ctx` the context the engine was launched with. A stored
    # result without them is from before depth existed: that is UNKNOWN, never
    # zero (see `measured_depth`). `prompt_tokens` is the size actually sent.
    depth: int | None = None
    ctx: int | None = None
    # Which prompt generator produced the number (see FILLER_GENERATION). The
    # filler changed while the calibration KEY did not, so a stored entry has to
    # say which generator it came from or old and new numbers are silently
    # compared.
    filler: str | None = None

    def as_measured(self) -> dict:
        """The dict to persist — `model_dump`, but `depth`/`ctx` are OMITTED when
        unknown rather than written as null.

        A caller reading `"depth" in measured` must be able to tell "no depth was
        recorded" (an old entry, or a depth-less run) from a recorded value. Both
        `measured_depth` and `measured_filler` read a missing key as UNKNOWN, but
        writing the key as null is a second spelling of the same fact and invites
        the `in` test to disagree.
        """
        d = self.model_dump()
        for k in ("depth", "ctx"):
            if d.get(k) is None:
                d.pop(k, None)
        return d


# The id of the prompt generator whose numbers are in the store. Bump this if
# `bench_text` ever changes shape again: a stored entry with no `filler` key (or
# a different one) was measured with different text and is not comparable with
# the current numbers, even under the same calibration key.
FILLER_GENERATION = "varied-v1"


# --- the bench prompt ---------------------------------------------------------
#
# The prompt used to be ONE sentence repeated: `"The quick brown fox jumps over
# the lazy dog. " * (prompt_tokens // 8)`. Two things were wrong with that.
#
# 1. A MoE routes to a NARROW set of experts on repetitive text. The owner
#    measured this on 2026-08-23: `n_cpu_moe 18` was crowned on filler, and on
#    varied text `n_cpu_moe 0` beat it by 24%. The benchmark therefore measured a
#    routing pattern no real conversation produces, and could crown the wrong
#    placement off it.
# 2. There was no depth dimension at all. llama.cpp attends only over OCCUPIED
#    cells (`get_n_kv` pads to 256), so a short run at a huge ctx measures
#    nothing about that ctx; `docs/review/findings-r3/37-262k-context.md` §3.
#
# So the filler is replaced with deterministic, seeded, lexically varied text,
# generated here — never fetched and never the owner's prose. It is built once
# per size and cached, so it cannot come to dominate a benchmark that loads a
# model per config. Its LENGTH is deliberately unchanged: `bench_text(n)` emits
# the same 9*(n//8) words the old filler did, so the default measurement keeps
# the exact prompt size it always had and only the text's variety differs.
_BENCH_VOCAB = tuple(dict.fromkeys((
    "time", "year", "people", "way", "day", "thing", "woman", "life", "child",
    "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "body",
    "information", "back", "parent", "face", "level", "office", "door",
    "health", "person", "art", "history", "party", "result", "change",
    "morning", "reason", "research", "girl", "moment", "air", "teacher",
    "force", "education", "foot", "age", "policy", "process", "music",
    "market", "sense", "nation", "plan", "college", "interest", "death",
    "experience", "effect", "class", "control", "care", "field",
    "development", "role", "effort", "rate", "heart", "drug", "show",
    "leader", "light", "voice", "wife", "police", "mind", "price", "report",
    "decision", "view", "relationship", "town", "road", "arm", "difference",
    "value", "building", "action", "model", "season", "society", "tax",
    "director", "position", "player", "record", "paper", "space", "ground",
    "form", "event", "official", "matter", "center", "couple", "site",
    "project", "activity", "star", "table", "need", "court", "production",
    "situation", "cost", "industry", "figure", "street", "image", "phone",
    "data", "cover", "picture", "practice", "piece", "land", "product",
    "doctor", "wall", "patient", "worker", "news", "test", "movie", "north",
    "love", "support", "technology", "step", "baby", "computer", "type",
    "attention", "film", "tree", "source", "organization", "cause", "hair",
    "century", "evidence", "window", "culture", "chance", "brother", "energy",
    "period", "course", "summer", "plant", "opportunity", "term", "letter",
    "condition", "choice", "rule", "daughter", "administration", "south",
    "husband", "floor", "campaign", "material", "population", "economy",
    "medical", "hospital", "church", "risk", "current", "fire", "future",
    "defense", "increase", "security", "bank", "west", "sport", "board",
    "subject", "officer", "private", "behavior", "performance", "fight",
    "goal", "second", "order", "author", "focus", "foreign", "blood",
    "agency", "nature", "color", "store", "reduce", "sound", "note",
    "movement", "page", "share", "common", "natural", "race", "concern",
    "series", "similar", "language", "response", "animal", "factor",
    "decade", "article", "artist", "scene", "stock", "career", "central",
    "treatment", "happy", "approach", "size", "fund", "media", "ready",
    "sign", "thought", "individual", "quality", "pressure", "answer",
    "resource", "meeting", "disease", "success", "amount", "ability",
    "staff", "character", "growth", "loss", "degree", "wonder", "attack",
    "region", "television", "training", "trade", "election", "physical",
    "general", "feeling", "standard", "message", "outside", "analysis",
    "benefit", "forward", "lawyer", "present", "section", "glass", "skill",
    "sister", "professor", "operation", "financial", "crime", "stage",
    "compare", "authority", "design", "knowledge", "station", "strategy",
    "discuss", "truth", "song", "example", "check", "environment", "public",
    "various", "guess", "executive", "prove", "entire", "rock", "forget",
    "claim", "remove", "manager", "enjoy", "network", "legal", "religious",
    "cold", "final", "main", "science", "green", "memory", "card", "seat",
    "cell", "establish", "trial", "expert", "spring", "firm", "option",
    "normal", "separate", "direct", "reveal", "weight", "tonight", "tough",
    "hill", "leg", "arrive", "master", "track", "spend",
)))

_BENCH_SEED = 0x5EED_C0DE

# Built once per size: generating the text must not dominate a benchmark whose
# real cost is a model load. A sweep benches one size, so this holds one entry.
_TEXT_CACHE: dict[int, str] = {}


def _prng(seed: int):
    """A tiny seeded LCG, so the text is byte-identical across processes and
    Python versions rather than depending on `random`'s internals.

    The constants are Knuth's MMIX linear congruential generator (multiplier
    6364136223846793005, increment 1442695040888963407), the same pair PCG uses
    by default; it has a full 2**64 period. We take bits 11..63 of the state
    rather than the low bits, because an LCG's low bits have short periods and
    would repeat within a single sentence. This is text variety, not
    cryptography — the only requirements are determinism and no visible cycle.
    """
    state = seed & 0xFFFFFFFFFFFFFFFF

    def _next(lo: int, hi: int) -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) \
            & 0xFFFFFFFFFFFFFFFF
        return lo + (state >> 11) % (hi - lo + 1)

    return _next


def bench_text(n_tokens: int) -> str:
    """Deterministic, varied synthetic text, sized EXACTLY like the old filler.

    "Tokens" here are words — the unit the old filler was sized in — and the
    count is `9 * (n_tokens // 8)`, the same words-per-budget the repeated
    sentence produced. So the default path's prompt is the same SIZE it always
    was and only its variety changes: replacing the filler must not move the
    prompt size, or every stored number silently shifts for a second reason. The
    text is seeded from the size, so the same budget always builds the same
    bytes while different budgets build different text.

    The INTENT of the variety is to avoid measuring a routing pattern that
    repetitive text can produce: the owner measured on 2026-08-23 that
    `n_cpu_moe 18` was crowned on filler while `n_cpu_moe 0` beat it by 24% on
    varied text (see the module comment above). Whether this particular text
    spreads any given MoE router is not measured here — no engine is run by this
    function or its tests.
    """
    if n_tokens < 8:
        return ""                      # 9 * (n // 8) == 0, exactly as before
    n_words = 9 * (n_tokens // 8)
    cached = _TEXT_CACHE.get(n_words)
    if cached is not None:
        return cached
    rng = _prng(_BENCH_SEED ^ ((n_words * 2654435761) & 0xFFFFFFFFFFFFFFFF))
    vocab = _BENCH_VOCAB
    last = len(vocab) - 1
    sentences: list[str] = []
    remaining = n_words
    while remaining > 0:
        take = min(rng(6, 16), remaining)   # varied sentence lengths
        words = [vocab[rng(0, last)] for _ in range(take)]
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
        remaining -= take
    text = " ".join(sentences)
    _TEXT_CACHE[n_words] = text
    return text


def measured_depth(entry: dict) -> int | None:
    """The fill depth a STORED measurement was taken at, or None if unknown.

    A result written before depth existed, a depth-less run, and a stored null
    all mean the same thing: the window depth is UNKNOWN. None is returned for
    all of them — never 0, which would claim the window was measured empty, a
    claim the old entries cannot support.
    """
    d = (entry.get("measured") or {}).get("depth")
    if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
        return None
    return d


def measured_filler(entry: dict) -> str | None:
    """Which prompt generator produced a STORED measurement, or None if unknown.

    The filler changed while the calibration key did not, so without this marker
    an old entry's number and a new one's are silently compared as if the same
    prompt produced them. None means the entry predates the marker and was
    measured with the old repeated sentence.
    """
    f = (entry.get("measured") or {}).get("filler")
    return f if isinstance(f, str) and f else None


def run_bench(port: int, prompt_tokens: int = 2048, gen_tokens: int = 128,
              depth: int | None = None, ctx: int | None = None) -> BenchResult:
    """Measure prefill/generation on the running server.

    `depth` is the window occupancy to measure at. When given, the prompt is
    EXTENDED (never shortened) so at least that many tokens occupy the KV window
    at generation time — otherwise "tok/s at 131K" is measured with an almost
    empty window and says nothing about a filled one. `depth=None` is the old
    behaviour exactly: the measured prompt is the whole window, at the same size
    the old filler produced. `ctx` is only recorded, as provenance for the number.

    With no `depth`, `prompt_tokens` is recorded exactly as before; with one, it
    records the occupancy actually requested (the size of the prompt sent is
    `bench_text`'s legacy `9*(n//8)` words for that budget).
    """
    sent_tokens = max(prompt_tokens, depth) if depth else prompt_tokens
    filler = bench_text(sent_tokens)
    r = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "user",
                            "content": filler + "\nSummarize in one sentence."}],
              "max_tokens": gen_tokens},
        timeout=1800)
    r.raise_for_status()
    t = r.json().get("timings")
    # AUDIT F08-7: "the engine answered but did not report timings" used to be
    # collapsed into "the engine measured 0 tokens/s", with ok=True set from "no
    # exception". A build/proxy that omits timings then made every sweep row read
    # 0.0 while looking successful — and `crowned_row` could crown one, writing
    # its flags and `calibrated: true` so the model was tuned forever on no data.
    if not t:
        raise RuntimeError("engine returned no timings — cannot measure speed")
    pp = float(t.get("prompt_per_second", 0.0))
    tg = float(t.get("predicted_per_second", 0.0))
    if pp == 0.0 and tg == 0.0:
        raise RuntimeError("engine reported 0 tokens/s for both prefill and "
                           "generation — no usable measurement")
    return BenchResult(pp_tps=pp, tg_tps=tg,
                       prompt_tokens=sent_tokens, gen_tokens=gen_tokens,
                       depth=depth, ctx=ctx, filler=FILLER_GENERATION)


def _capabilities(slug: str) -> tuple:
    try:
        from .registry import Registry
        spec = Registry.load().models.get(slug)
        if spec is not None:
            return tuple(spec.capabilities)
    except Exception:
        pass
    return ()


def _sweepable_caps(plan) -> tuple:
    """Capabilities the sweep may act on, with `mtp` decided by the FILE.

    A sweep runs real llama-server launches, so a capability that the model
    advertises but this quant does not carry is not a wasted row — it is a
    driver reset mid-sweep. The gguf being benched is on disk by definition,
    so the tensor table can always be consulted."""
    caps = set(_capabilities(plan.model_slug))
    caps.discard("mtp")
    try:
        from .hangar import file_has_mtp
        if file_has_mtp(plan.gguf) is True:
            caps.add("mtp")
    except Exception:
        pass          # unverifiable stays out: absence of proof is not proof
    return tuple(caps)


def _tools_capable(slug: str) -> bool:
    """Whether the model declares the `tools` capability. Unknown → True:
    assume tools and protect quality rather than risk degrading it."""
    try:
        from .registry import Registry
        spec = Registry.load().models.get(slug)
        if spec is not None:
            return "tools" in spec.capabilities
    except Exception:
        pass
    return True


def calibration_path() -> Path:
    return rigma_home() / "calibration.json"


def load_calibration() -> dict:
    try:
        return json.loads(calibration_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def current_identity(backend: str = "", gpu=None) -> hwid.HardwareIdentity:
    """The hardware this process is running on, for keying a calibration.

    Never raises: an identity that cannot be read is an EMPTY one, which degrades
    to the old model:quant:backend behaviour rather than losing the calibration
    entirely. A probe failure must not make the tool refuse to measure.

    Reads the raw enumeration directly rather than going through
    `probe_hardware()`, which needs the registry's GPU table. The table only picks
    which BACKENDS a card advertises, and the backend is passed in here already —
    so the table would add a registry dependency to a cache key and change nothing
    about the identity itself.
    """
    try:
        if gpu is None:
            from .probe import enumerate_vulkan
            rows = enumerate_vulkan()
            gpu = rows[0] if rows else None
        if gpu is None:
            return hwid.HardwareIdentity(backend=backend or "cpu")
        if isinstance(gpu, dict):
            return hwid.identity_from_gpu(backend or "vulkan", gpu)
        return hwid.identity_from_gpu(backend or "vulkan", {
            "vendor_id": getattr(gpu, "vendor_id", None),
            "device_id": getattr(gpu, "device_id", None),
            "device_uuid": getattr(gpu, "device_uuid", ""),
            "driver_version": getattr(gpu, "driver_version", ""),
            "name": getattr(gpu, "name", ""),
        })
    except Exception:
        return hwid.HardwareIdentity(backend=backend or "")


def _identity_cache_key(backend: str) -> hwid.HardwareIdentity:
    """Cached per backend: `probe_hardware()` is a subprocess away from slow, and
    the calibration key is built on several hot paths.

    An EMPTY identity is deliberately not cached. A transient probe failure would
    otherwise be frozen for the life of the process, and every calibration key
    built afterwards would silently lose its hardware component — the exact
    silent-degradation this module exists to prevent.
    """
    got = _IDENT_CACHE.get(backend)
    if got is not None:
        return got
    ident = current_identity(backend)
    if ident.vendor_id or ident.uuid:
        _IDENT_CACHE[backend] = ident
    return ident


_IDENT_CACHE: dict[str, hwid.HardwareIdentity] = {}


def calibration_key(model: str, quant: str, backend: str) -> str:
    """The current calibration key for a model+quant+backend."""
    return hwid.calibration_key(model, quant, _identity_cache_key(backend))


def legacy_key(model: str, quant: str, backend: str) -> str:
    """The key used before R3-CAL-1: model:quant:backend, no hardware.

    Kept because entries written under it are still valid ON THE MACHINE THAT
    WROTE THEM — discarding every existing calibration on upgrade would make the
    tool worse for the person upgrading, which is not what fixing a cache key is
    supposed to do.
    """
    return f"{model}:{quant}:{backend}"


def calibration_entry(cal: dict, model: str, quant: str,
                      backend: str) -> tuple[str, dict]:
    """The entry for a model+quant+backend, and the key it was found under.

    Tries the identity key, then the legacy key. Returns ("", {}) on a miss, which
    callers already treat as "not calibrated".
    """
    for k in (calibration_key(model, quant, backend), legacy_key(model, quant, backend)):
        if k in cal:
            return k, cal[k]
    return "", {}


def _base_calibration_key(k: str) -> str:
    """`model:quant:backend:<hardware digest>` -> `model:quant:backend`.

    The identity digest is appended to the key (R3-CAL-1), so pruning has to
    strip it to know WHICH model an entry belongs to. A legacy key has no
    trailing digest and is returned unchanged.
    """
    return k.rsplit(":", 1)[0] if k.count(":") >= 3 else k


def prune_calibration(cal: dict, keep_per_identity: int = 1) -> dict:
    """Keep only the newest `keep_per_identity` entries per (model, identity).

    R3-CAL-2. This grouped by HARDWARE IDENTITY ALONE, so `keep_per_identity=1`
    kept exactly ONE entry per GPU — across every model on it. Calibrating a
    second model therefore evicted the first model's row, and because
    `save_calibration` prunes unconditionally on every save, two models on one
    card oscillated: each save destroyed the other's measurement, so neither
    ever stayed calibrated and the expensive first-load sweep re-ran forever.

    The unit that may accumulate is "this model, measured on this hardware", so
    that is the grouping key. The identity stays in it because the point of the
    cap is still to bound growth across GPU swaps and driver experiments.

    Entries with no recorded hardware are kept — they are from before identity
    existed and cannot be attributed to an identity to prune.
    """
    by_group: dict[tuple[str, str], list[tuple[str, str]]] = {}
    keep: set[str] = set()
    for k, entry in cal.items():
        ident = (entry.get("hardware") or {}).get("id")
        if not ident:
            keep.add(k)
            continue
        by_group.setdefault((_base_calibration_key(k), ident), []).append(
            (entry.get("date") or "", k))
    for rows in by_group.values():
        rows.sort(reverse=True)          # newest date first, then key
        keep.update(k for _, k in rows[:keep_per_identity])
    return {k: v for k, v in cal.items() if k in keep}


def save_calibration(key: str, measured: dict, flags: dict | None = None,
                     calibrated: bool = False, ctx: int = 0,
                     identity: hwid.HardwareIdentity | None = None,
                     backend: str = "") -> None:
    cal = load_calibration()
    entry = cal.get(key, {})
    entry["measured"] = measured
    if flags is not None:
        entry["flags"] = flags
    if calibrated:
        entry["calibrated"] = True   # one-time first-load tune has run
    # What the numbers were measured ON. An entry used to carry a day-granularity
    # date and nothing else, so there was no way to tell that a calibration
    # predated an engine bump or was measured at a different context — it simply
    # kept being applied. `schema` marks entries that carry this; anything
    # without it is from before and is read leniently.
    entry["schema"] = 3
    # R3-ENG-6: the MEASURED build, via the backend this calibration was taken on.
    # Recording the manifest string here made this field useless: calibration_stale
    # compared it against the same manifest string, so an engine change could never
    # invalidate anything. On the owner's machine the fork in `rocm/` was labelled
    # `b9867` for exactly this reason.
    entry["engine"] = _engine_version(backend)
    # R3-CAL-1: WHICH CARD. Without it a 3090 silently inherits a 4090's number,
    # which is the failure mode that is invisible rather than loud. The identity is
    # recorded as a field as well as being part of the key so a stale entry can
    # explain itself instead of just being absent.
    if identity is not None:
        entry["hardware"] = identity.as_dict()
    # What the desktop was holding when this was measured. Without it there is
    # nothing in the entry to distinguish 9.95 tok/s from 37.59 for the same
    # model on the same engine (measured 2026-08-21).
    try:
        from .probe import gpu_used_mb
        used = gpu_used_mb()
        if used is not None:
            entry["vram_used_mb"] = round(used)
    except Exception:
        pass
    if ctx:
        entry["ctx"] = ctx
    entry["date"] = datetime.date.today().isoformat()
    cal[key] = entry
    cal = prune_calibration(cal)
    # R3-STORE-1: atomic. A torn calibration.json used to load as `{}`, and the
    # NEXT save wrote that `{}` back plus one entry — so one interrupted write
    # destroyed every other measured model, permanently.
    atomic_write_json(calibration_path(), cal)


# How much the desktop's VRAM footprint may drift before a calibration stops
# meaning anything. Measured 2026-08-21: 2,828MB of drift changed the same model
# on the same engine from 9.95 to 37.59 tok/s, because Windows silently paged a
# third of the weights to system RAM. A few hundred MB is ordinary desktop
# noise; anything more changes what the number means.
VRAM_DRIFT_TOLERANCE_MB = 700


def calibration_stale(entry: dict, vram_used_mb: float | None,
                      engine: str = "", ctx: int = 0,
                      identity: hwid.HardwareIdentity | None = None) -> str | None:
    """Why this calibration should not be trusted, or None if it should.

    Speed on a fixed model+quant+engine is not a constant: it depends on whether
    the weights actually fit in VRAM, and on Windows that depends on what ELSE
    is holding VRAM. A calibration taken while a browser held 4GB describes a
    machine that no longer exists once the browser closes.

    R3-CAL-1 adds two more ways to be stale, and they are NOT the same:
      * a different CARD (hard) — the number is about other hardware entirely;
      * the same card under a new DRIVER (soft) — real, but a re-measure rather
        than a discard. A driver update made non-FA PP 5% and FA 15% faster.
    Both are reported here as reasons not to trust the number, which is the one
    thing callers do with them.
    """
    if not entry:
        return None
    if identity is not None:
        hard = hwid.hard_mismatch(entry, identity)
        if hard:
            return hard
        soft = hwid.soft_reasons(entry, identity, engine=engine, ctx=ctx)
        if soft:
            return "; ".join(soft)
        # Identity already covered engine/ctx/driver, so fall through only to the
        # VRAM check rather than repeating the same comparisons below.
        was = entry.get("vram_used_mb")
        if was is None or vram_used_mb is None:
            return None
        drift = abs(float(vram_used_mb) - float(was))
        if drift > VRAM_DRIFT_TOLERANCE_MB:
            return (f"measured with {was:,.0f} MiB of VRAM held by other apps, "
                    f"now {vram_used_mb:,.0f} MiB — a {drift:,.0f} MiB shift changes "
                    "whether the model fits on the GPU at all")
        return None
    if engine and entry.get("engine") and entry["engine"] != engine:
        return f"measured on engine {entry['engine']}, now on {engine}"
    if ctx and entry.get("ctx") and entry["ctx"] != ctx:
        return f"measured at ctx {entry['ctx']:,}, now {ctx:,}"
    was = entry.get("vram_used_mb")
    # Entries written before this was recorded, and machines where it cannot be
    # read, are read leniently: no reading is not a reading of zero, and
    # invalidating every old entry at once helps nobody.
    if was is None or vram_used_mb is None:
        return None
    drift = abs(float(vram_used_mb) - float(was))
    if drift > VRAM_DRIFT_TOLERANCE_MB:
        return (f"measured with {was:,.0f} MiB of VRAM held by other apps, "
                f"now {vram_used_mb:,.0f} MiB — a {drift:,.0f} MiB shift changes "
                "whether the model fits on the GPU at all")
    return None


def is_calibrated(model: str, quant: str, backend: str) -> bool:
    """True once a model+quant+backend has been auto-tuned on THIS HARDWARE —
    so first-load calibration runs exactly once, never on every load.

    R3-CAL-1: the lookup is identity-aware, so a calibration measured on another
    GPU does not count as calibrated here. The legacy key is still honoured for
    the machine that wrote it, which is why this is not simply a key change.
    """
    _, entry = calibration_entry(load_calibration(), model, quant, backend)
    if not entry:
        return False
    # A calibrated flag from different hardware is not a calibration for this one:
    # re-tune rather than serve a number measured elsewhere.
    if hwid.hard_mismatch(entry, _identity_cache_key(backend)):
        return False
    return bool(entry.get("calibrated"))


def clear_calibration(model: str, quant: str, backend: str) -> bool:
    """Forget one model's tune so it re-optimizes on next load (or falls back to
    the safe defaults). Returns True if there was an entry to clear.

    Clears BOTH the identity key and the legacy key: "forget this tune" should mean
    forgotten, and leaving a legacy entry behind would make the next lookup find it
    and report the model as still calibrated.
    """
    cal = load_calibration()
    hit = [k for k in (calibration_key(model, quant, backend),
                       legacy_key(model, quant, backend)) if cal.pop(k, None) is not None]
    if not hit:
        return False
    atomic_write_json(calibration_path(), cal)
    return True


def reset_all_calibration() -> int:
    """Wipe every stored tune. Returns how many were cleared."""
    n = len(load_calibration())
    atomic_write_text(calibration_path(), "{}")
    return n


SPEC_CROWN_MARGIN = 1.15   # a speculation config must beat the best
                           # non-spec row by >=15% to be crowned: the short
                           # predictable-text bench flatters draft acceptance,
                           # so a narrow bench win is a real-world loss


def rows_log_path() -> Path:
    return rigma_home() / "logs" / "bench-rows.jsonl"


def _log_row(entry: dict) -> None:
    """One measurement, appended. Separated so a test can make it fail."""
    p = rows_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_rows(plan, rows: list[dict], best: dict | None) -> None:
    """Keep every measurement a sweep made, not only the one it crowned.

    A sweep loads a real engine per config and benches it — the most expensive
    numbers Rigma produces. Only the winner's two floats reached
    calibration.json; every losing row was returned, printed once, and dropped.
    And because calibration is only written when the winner has flags, an
    explicit sweep where the BASELINE won recorded nothing at all, not even the
    baseline speed it had just spent several model loads measuring.

    Append-only, structure-only, and wrapped: this is bookkeeping, and a sweep
    that lost its log is still a sweep. Same shape as serve._shape_log.
    """
    try:
        won = (best or {}).get("label")
        stamp = datetime.date.today().isoformat()
        for r in rows:
            entry = {"date": stamp, "model": plan.model_slug,
                     "quant": plan.gguf.quant, "backend": plan.backend,
                     "ctx": plan.flags.ctx, "engine": _engine_version(plan.backend),
                     "label": r.get("label"), "flags": r.get("flags") or {},
                     "tg_tps": r.get("tg_tps"), "pp_tps": r.get("pp_tps"),
                     "filler": r.get("filler"),
                     "ok": bool(r.get("ok")), "error": r.get("error", ""),
                     "crowned": r.get("label") == won}
            # Omit rather than null: "no depth recorded" must not be spelled the
            # same way as a recorded value (see BenchResult.as_measured).
            if r.get("depth") is not None:
                entry["depth"] = r["depth"]
            _log_row(entry)
    except Exception:
        pass          # a sweep that lost its log is still a sweep


def _engine_version(backend: str = "") -> str:
    """Which llama.cpp build produced these numbers. A calibration measured on
    one engine is not evidence about another, and nothing recorded this.

    R3-ENG-6: takes the backend because the builds on disk can differ per backend —
    on the owner's machine `rocm` held a third-party fork while `vulkan`/`cpu` held
    the pin — and `server_ops.engine_version` now measures the binary rather than
    returning the manifest string.
    """
    try:
        from .server_ops import engine_version
        return engine_version(backend)
    except Exception:
        return ""


# C11: engine env vars that turn OFF a quality-preserving default. A sweep
# scores tokens/sec only, so a row carrying one of these may be MEASURED but
# must not be crowned — and so never persisted into calibration, which
# `resolve()` then applies to EVERY later launch.
_QUALITY_ENV_LEVERS = ("LLAMA_ATTN_ROT_DISABLE",)


def _carries_quality_env_lever(flags: dict | None) -> bool:
    """Whether a trial's override flips an engine default that protects QUALITY.

    `LLAMA_ATTN_ROT_DISABLE` disables llama.cpp's Hadamard attention rotation,
    which the engine enables BY DEFAULT for a quantized KV cache and which is
    what keeps that cache's loss down: `quant_quality._KV_PPL` records q4_0/q4_0
    at +0.19% perplexity WITH rotation vs +2.42% without (PR 7412). Crowning it
    on tokens/sec alone would write it to calibration.json, and
    `runtime.launch_server` would merge it into every later launch's child
    environment — a silent quality regression traded for a speed win.
    """
    env = (flags or {}).get("env") or {}
    return any(name in env for name in _QUALITY_ENV_LEVERS)


def _effective_env(plan_flags, override: dict) -> dict:
    """The child environment a trial ACTUALLY runs with.

    `run_sweep` builds each trial as `plan.flags.model_copy(update=override)`,
    so an override that names `env` REPLACES the plan's whole env dict rather
    than merging into it; `runtime.launch_server` then merges the result over
    `os.environ`. The trial gate must ask about THIS dict, not the override
    alone: a lever already on `plan.flags.env` — e.g. one merged from a
    `calibration.json` entry by `resolve._apply_calibration` — is invisible to a
    check of the override, so on a tools-capable model the sweep would launch
    every trial with rotation off while believing it had dropped the axis.
    """
    return getattr(plan_flags.model_copy(update=override), "env", None) or {}


def crowned_row(rows: list[dict],
                allow_quality_levers: bool = False) -> dict | None:
    """The config a sweep actually crowns. ONE rule, two readers.

    `run_sweep` saves the winner; `rigma sweep` prints it. They used to decide
    separately, and the CLI's version had no margin gate and required non-empty
    flags — so it could announce "winner: spec-mtp-2 … saved to calibration"
    while calibration.json held the baseline.

    The margin gate: real-world draft acceptance is LOWER than on the bench's
    predictable filler (live 2026-07-21: crowned at 44.7 t/s, ran at 36.8). A
    speculation config must beat the best non-spec row decisively or the
    non-spec row is crowned instead.

    The quality gate (C11): a row that carries a quality-degrading env lever is
    not a candidate at all unless `allow_quality_levers` is set. Both readers
    call this with the default, so the winner the CLI announces and the flags
    `run_sweep` saves cannot disagree.

    Sorts a copy, so the answer does not depend on whether the caller sorted.
    """
    ok = sorted((r for r in rows
                 if r.get("ok")
                 and (allow_quality_levers
                      or not _carries_quality_env_lever(r.get("flags")))),
                key=lambda r: r.get("tg_tps", 0.0), reverse=True)
    best = next(iter(ok), None)
    if best is not None and (best.get("flags") or {}).get("spec_type"):
        plain = next((r for r in ok
                      if not (r.get("flags") or {}).get("spec_type")), None)
        if plain is not None and \
                best["tg_tps"] < plain["tg_tps"] * SPEC_CROWN_MARGIN:
            best = plain
    return best


def sweep_configs(base: ComboFlags, moe: bool,
                  caps: tuple | list = ()) -> list[tuple[str, dict]]:
    """Flag-override sets to A/B on this machine. Baseline first; each entry is
    a partial ComboFlags update. Axes come from the RDNA4 findings: FA gates the
    fast KV path, symmetric KV precision, prefill batch, Vulkan coopmat,
    host-side KV-op placement (C2 `--no-op-offload`), attention-rotation
    disabling (C11 `LLAMA_ATTN_ROT_DISABLE`), and (MoE only) graphics-queue +
    offload depth.

    Whether the C11 axis may be TRIALLED and whether its row may be CROWNED are
    decided by `run_sweep` (tools-capable models drop it; no model crowns it
    without an explicit opt-in), not here — this function only lists the axes.
    """
    cfgs: list[tuple[str, dict]] = [("baseline", {})]
    cfgs.append(("fa-off", {"flash_attn": "off"}))
    cfgs.append(("kv-q8", {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}))
    cfgs.append(("kv-q4", {"cache_type_k": "q4_0", "cache_type_v": "q4_0"}))
    cfgs.append(("batch-big", {"batch": 16384, "ubatch": 2048}))
    cfgs.append(("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}}))
    # C2: keep the KV KQ/KQV ops on the CPU (llama.cpp `--no-op-offload`). Same
    # class as the driver env toggles above — a per-machine placement lever whose
    # direction on RDNA4 is a measurement, not a derivation. The engine accepts
    # it at both pins and its compiled default is op-offload ON, so the OFF case
    # is the untouched baseline and emits nothing; the ON case is this one entry,
    # and the row records `{"no_op_offload": true}` as its flags. Deliberately
    # NOT in quick_configs: a default load must not change. See
    # ComboFlags.no_op_offload for the upstream URLs/lines and the PREDICTION.
    cfgs.append(("no-op-offload", {"no_op_offload": True}))
    # C11: disable the Hadamard attention rotation. There is NO argv flag for
    # this at either pin — only the env `LLAMA_ATTN_ROT_DISABLE` — so unlike C2
    # it is trialled through the engine-spawn env (`ComboFlags.env`), exactly
    # like the GGML_VK_* driver toggles above. Provenance:
    #   PrismML-Eng/llama.cpp@87268f77 src/llama-kv-cache.cpp L316-326
    #   ggml-org/llama.cpp@b9867     src/llama-kv-cache.cpp L329-339
    #     const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
    #     attn_rot_k = !attn_rot_disable && ... && ggml_is_quantized(type_k) ...
    # Fork override (same pin, src/llama-kv-cache.cpp L329-332): for
    # LLAMA_ARCH_DEEPSEEK32 / DEEPSEEK4 / GLM_DSA / DOTS3NOTE, when
    # n_embd_head_k_full == indexer_head_size, `attn_rot_k` is forced back to
    # true AFTER the `!attn_rot_disable` expression — so on those archs this
    # axis disables only V rotation. The general path above is what the owner's
    # current load runs.
    # `getenv` returning null is the engine default, so the OFF case sets
    # nothing at all and every pre-existing trial's child environment is
    # byte-identical; the ON case merges the variable over `os.environ` for the
    # CHILD only (`runtime.launch_server`), never the parent. The rotation is ON
    # by default for a QUANTIZED KV cache and this machine's own load runs it
    # (`.scratch/prism-v.log:4576-4578`: `K (q8_0)` with `attn_rot_k = 1`), so
    # the axis has a real effect on the resolved plan — not only on kv-q8/q4.
    # It is a QUALITY lever, not a driver toggle: `run_sweep` does not trial it
    # on a tools-capable model and does not crown it for any model without an
    # explicit opt-in (`allow_quality_levers`), so it cannot reach calibration.
    # Deliberately NOT in quick_configs: a default load must not change.
    cfgs.append(("attn-rot-off", {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}))
    if moe:
        cfgs.append(("gfxqueue-on", {"env": {"GGML_VK_ALLOW_GRAPHICS_QUEUE": "1"}}))
        if base.n_cpu_moe > 0:
            cfgs.append(("moe-less-offload",
                         {"n_cpu_moe": max(0, base.n_cpu_moe - 1)}))
    # speculation trials: exhaustive sweep ONLY, and gated by
    # SPEC_CROWN_MARGIN in run_sweep — the short bench flatters acceptance.
    # `caps` here is the FILE's answer (see run_sweep), not the model's: MTP is
    # dropped or kept per quant, and trialling draft-mtp against a quant without
    # the tensors resets the Vulkan driver mid-sweep.
    if "mtp" in (caps or ()):
        cfgs.append(("spec-mtp-2", {"spec_type": "draft-mtp", "spec_n_max": 2}))
        cfgs.append(("spec-mtp-4", {"spec_type": "draft-mtp", "spec_n_max": 4}))
    return cfgs


def quick_configs(base: ComboFlags, moe: bool,
                  caps: tuple | list = ()) -> list[tuple[str, dict]]:
    """The short first-load set: only the toggles that genuinely can't be
    defaulted and are worth a per-machine measurement. The rest are already
    applied automatically by the resolver. Full exhaustive set = sweep_configs
    (via `rigma sweep`)."""
    cfgs: list[tuple[str, dict]] = [("baseline", {})]
    cfgs.append(("fa-off", {"flash_attn": "off"}))
    cfgs.append(("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}}))
    if moe:
        cfgs.append(("gfxqueue-on", {"env": {"GGML_VK_ALLOW_GRAPHICS_QUEUE": "1"}}))
    # NOTE: the C2 `--no-op-offload` axis is deliberately NOT in the first-load
    # set either: it is a sweep-only trial, so a default load's argv is unchanged
    # whether or not the sweep has ever been run. The C11 `LLAMA_ATTN_ROT_DISABLE`
    # axis is omitted for the same reason, and because it is a quality lever: a
    # default load must not acquire it.
    # NOTE: spec-mtp trials deliberately NOT in the auto first-load set.
    # Live lesson 2026-07-21: the 96-token bench summarises highly
    # predictable filler, which inflates MTP draft acceptance — draft-mtp
    # measured 44.7 t/s and was crowned, then real varied chat ran at 36.8
    # (rejected drafts are pure overhead). Speculation lives in the explicit
    # exhaustive sweep, where the margin gate below applies.
    return cfgs


def run_sweep(plan: RunPlan, exe, model_path, port: int = 11601,
              prompt_tokens: int = 2048, gen_tokens: int = 96,
              progress=None, configs=None, extra_args=None,
              mark_calibrated: bool = False, depth: int | None = None,
              allow_quality_levers: bool = False) -> list[dict]:
    """Launch `plan` under each config on `port`, bench it, and persist the best
    tg/s config to calibration (which resolve() then applies automatically). The
    caller guarantees `port` is free (scratch port, or mid-switch with the old
    engine already killed) — this never touches a live server.

    `depth` is passed through to every `run_bench` and recorded on every row, so
    a sweep can A/B at a FILLED window rather than the almost-empty one a short
    prompt leaves (see `run_bench`).

    Quality levers (C11): a config that flips a quality-degrading engine env var
    (`_carries_quality_env_lever`) is treated like the q4_0 KV cache is on a
    tools-capable model — it is not crowned, and so never persisted into
    calibration and applied to every later launch. On a tools-capable model it
    is not even trialled (it cannot win, so the engine load would be wasted);
    on any other model it IS trialled and recorded, because the measurement is
    the point of a sweep, but `allow_quality_levers=True` is required before it
    may win.
    """
    is_moe = plan.flags.n_cpu_moe > 0
    if configs is None:
        configs = sweep_configs(plan.flags, is_moe,
                                caps=_sweepable_caps(plan))
    # Never let the sweep crown q4_0 KV on a tools-capable model: llama.cpp's
    # own function-calling docs warn extreme KV quantization significantly
    # degrades tool calling, and the sweep scores tokens/sec only — it would
    # trade a silent quality regression for a speed win. (Mirrors the
    # registry's DeltaNet q8_0 cache policy.) C11's rotation env toggle is the
    # same class of hazard through the child environment, so it is dropped here
    # too; on a non-tools model `crowned_row` still refuses to crown it.
    # C11-nits: ask about the EFFECTIVE env (`_effective_env`), not the override
    # alone — the plan may already carry the lever (a merged calibration row),
    # in which case every env-less override would inherit it.
    if _tools_capable(plan.model_slug):
        configs = [(label, o) for label, o in configs
                   if o.get("cache_type_k") != "q4_0"
                   and o.get("cache_type_v") != "q4_0"
                   and not _carries_quality_env_lever(
                       {"env": _effective_env(plan.flags, o)})]
    rows: list[dict] = []
    for label, override in configs:
        flags = plan.flags.model_copy(update=override)
        trial = plan.model_copy(update={"flags": flags})
        if progress:
            progress(label)
        # `depth` is added only when one was requested: a depth-less row must not
        # carry a null that a reader could mistake for a recorded value.
        base = {"label": label, "flags": override,
                "filler": FILLER_GENERATION}
        if depth is not None:
            base["depth"] = depth
        try:
            srv = launch_server(exe, trial, model_path, port=port, timeout=300.0,
                                extra_args=extra_args)
        except Exception as e:  # a config that OOMs/crashes is a valid "loss"
            rows.append({**base, "tg_tps": 0.0, "pp_tps": 0.0, "ok": False,
                         "error": str(e)[:200]})
            continue
        try:
            res = run_bench(port, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                            depth=depth, ctx=plan.flags.ctx or None)
            rows.append({**base, "tg_tps": res.tg_tps,
                         "pp_tps": res.pp_tps, "ok": True})
        except Exception as e:  # loaded but wouldn't serve — count as a loss
            rows.append({**base, "tg_tps": 0.0, "pp_tps": 0.0, "ok": False,
                         "error": str(e)[:200]})
        finally:
            srv.stop()
    rows.sort(key=lambda r: r["tg_tps"], reverse=True)
    best = crowned_row(rows, allow_quality_levers=allow_quality_levers)
    _log_rows(plan, rows, best)
    if best is not None and (best["flags"] or mark_calibrated):
        key = calibration_key(plan.model_slug, plan.gguf.quant, plan.backend)
        measured = {"tg_tps": best["tg_tps"], "pp_tps": best["pp_tps"],
                    "filler": best.get("filler") or FILLER_GENERATION}
        if best.get("depth") is not None:
            measured["depth"] = best["depth"]
        save_calibration(key, measured,
                         flags=best["flags"], calibrated=mark_calibrated,
                         ctx=plan.flags.ctx, backend=plan.backend,
                         identity=_identity_cache_key(plan.backend))
    return rows


def auto_calibrate(plan: RunPlan, exe, model_path, port: int = 11601,
                   extra_args=None, progress=None) -> RunPlan:
    """One-time, first-load tune: if this model+quant+backend has never been
    calibrated on this machine, A/B the quick config set on `port` and return
    the plan with the winning flags applied. Cached forever after (subsequent
    loads return instantly). No-op on CPU or when already calibrated."""
    # Read under whichever key has the entry (identity first, legacy second) but
    # always WRITE and re-read under the identity key: `run_sweep` persists to the
    # identity key, so reading flags back under a legacy key would miss the tune
    # that was just measured and silently apply nothing.
    key = calibration_key(plan.model_slug, plan.gguf.quant, plan.backend)
    ident = _identity_cache_key(plan.backend)
    _, entry = calibration_entry(load_calibration(), plan.model_slug,
                                plan.gguf.quant, plan.backend)
    # R3-CAL-1: an entry measured on other hardware is not a calibration for this
    # one. Ignore it (rather than erroring) and re-measure.
    if entry and hwid.hard_mismatch(entry, ident):
        entry = {}

    def _apply(p: RunPlan) -> RunPlan:
        flags = load_calibration().get(key, {}).get("flags") or {}
        if not flags:
            return p
        return p.model_copy(update={
            "flags": p.flags.model_copy(update=flags),
            "origin": p.origin if p.origin.endswith("+calibrated")
            else p.origin + "+calibrated"})

    if entry.get("calibrated"):
        # Adopt a legacy entry onto the identity key so the next lookup is a
        # direct hit and the stale key can be pruned away.
        if key not in load_calibration():
            save_calibration(key, entry.get("measured", {}),
                             flags=entry.get("flags"), calibrated=True,
                             ctx=entry.get("ctx", 0), backend=plan.backend,
                             identity=ident)
        return _apply(plan)
    if plan.backend == "cpu":
        return plan   # nothing worth measuring on CPU
    run_sweep(plan, exe, model_path, port=port,
              configs=quick_configs(plan.flags, plan.flags.n_cpu_moe > 0,
                                    caps=_capabilities(plan.model_slug)),
              extra_args=extra_args, progress=progress, mark_calibrated=True)
    return _apply(plan)


def verdict(result: BenchResult, expected: dict | None) -> str:
    if not expected or "tg_tps" not in expected:
        return "no expectation recorded for this combo"
    floor = expected["tg_tps"][0]
    if result.tg_tps >= floor:
        return f"OK (within/above expected range, floor {floor} t/s)"
    return (f"BELOW expected floor ({floor} t/s) — combo may need tuning "
            f"on this machine")
