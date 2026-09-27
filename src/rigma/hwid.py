"""What a calibration was measured ON (R3-CAL-1).

A calibration entry used to be keyed `model:quant:backend`. On one machine that is
almost right; across machines, or across a GPU swap, or on a laptop with both an
iGPU and a dGPU, it is wrong in the SILENT direction — a 3090 inherits a 4090's
number and the UI reports a throughput the machine cannot reach.

The evidence for why the key needs hardware identity at all (llama.cpp CUDA
scoreboard, Llama-2-7B Q4_0):

  * nine distinct 24 GB cards span tg128 54.74 -> 189.33, so VRAM size is not an
    identity proxy;
  * 3090 -> 4090 is pp512 +132%; A100 -> H100 +105%;
  * merely having 4 GPUs visible cut a 7900 XTX's pp512 from 2,023 to 1,545
    (-24%), and an Optimus laptop measured Vulkan prefill "around 3500 instead of
    6000 t/s" without correct device-routing env vars — 42% from device selection.

The evidence for why identity is NOT sufficient, which is why this module returns
a *soft* verdict separately from a hard one:

  * power limits alone moved a 3090's pp512 from 4,175 to 5,406 (+29%) with no
    identity change at all;
  * the same RX 7900 XTX varies 2.4x across driver, OS and overclock, and both
    llama.cpp maintainers say so explicitly.

So this module answers two different questions and never conflates them:

  `hard_identity`  — a different CARD. The cached number is about other hardware
                     and must not be shown at all.
  `soft_reasons`   — the same card, changed conditions (driver, context, engine).
                     The number is stale; re-measure, but it is not a lie.

WHAT IS DELIBERATELY NOT IN THE KEY
  * `pipelineCacheUUID` — identifies "a compatible device and driver combination",
    so it moves on a driver update and would throw away a still-valid measurement.
  * `deviceName` — embeds the driver ("... (RADV NAVI31)" vs "... (AMD
    proprietary driver)"); it is a label, not an identity.
  * the driver version — it is recorded as a FIELD and checked as a soft reason.
    It is real (a driver update made non-FA PP 5% and FA 15% faster) but
    `driverVersion` is implementation-defined, so its encoding is not comparable
    across vendors and keying on it fragments the cache for an effect validation
    catches better.
  * the VRAM size and the GPU count — both change without the card changing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# 12 hex chars = 48 bits. Collisions are not a real risk at the 1-4 identities a
# single machine has, and a short key keeps calibration.json readable by a human,
# which is the whole point of it being JSON.
_HASH_CHARS = 12

# Backends whose identity is a real GPU. `cpu` has no device identity, so keying
# it on hardware would invent a distinction that does not exist.
_GPU_BACKENDS = {"vulkan", "cuda", "rocm", "metal"}


def _norm(value) -> str:
    """One normalisation for every component, so "0x7550" and 30032 and " 7550 "
    cannot produce three different identities for one card."""
    if value is None:
        return ""
    if isinstance(value, int):
        return hex(value).lower()
    s = str(value).strip().lower()
    if not s:
        return ""
    try:
        return hex(int(s, 0)).lower()
    except ValueError:
        return s


@dataclass(frozen=True)
class HardwareIdentity:
    """The hardware a calibration belongs to, and its stable short key."""
    backend: str = ""
    vendor_id: str = ""
    device_id: str = ""
    uuid: str = ""
    name: str = ""
    driver_version: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.backend in _GPU_BACKENDS

    @property
    def digest(self) -> str:
        """Truncated sha256 of the identity components, in a fixed order.

        Only the components that identify the CARD. The driver is excluded on
        purpose — see the module docstring — and so is the marketing name.
        """
        if not self.is_gpu:
            # CPU: there is no card to identify. The backend is the whole
            # identity, which is what the old key already used.
            material = f"cpu|{_norm(self.backend)}"
        else:
            material = "|".join([
                _norm(self.backend), _norm(self.vendor_id),
                _norm(self.device_id), _norm(self.uuid)])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_HASH_CHARS]

    def as_dict(self) -> dict:
        return {"backend": self.backend, "vendor_id": self.vendor_id,
                "device_id": self.device_id, "uuid": self.uuid,
                "name": self.name, "driver_version": self.driver_version,
                "id": self.digest}


def identity_from_gpu(backend: str, gpu: dict | None) -> HardwareIdentity:
    """Build an identity from a raw probe row (`probe.enumerate_vulkan()` shape).

    Every component is normalised HERE, at construction, so a HardwareIdentity
    always holds canonical values. Normalising only at comparison time was a real
    bug: `as_dict()` stored the raw form while `hard_mismatch` compared a
    normalised form, so a saved entry and a live identity could disagree about a
    card that had not changed.

    A missing `device_uuid` is left empty rather than filled with a placeholder:
    an absent component must not collide with a real one, and the digest is still
    stable for a driver that never reports a UUID.
    """
    g = gpu or {}
    return HardwareIdentity(
        backend=str(backend or ""),
        vendor_id=_norm(g.get("vendor_id")),
        device_id=_norm(g.get("device_id")),
        uuid=_norm(g.get("uuid") if g.get("uuid") is not None
                   else g.get("device_uuid")),
        name=str(g.get("name") or ""),
        driver_version=_norm(g.get("driver_version")),
    )


def from_dict(entry: dict | None) -> HardwareIdentity:
    """Rebuild an identity from a saved `hardware` block.

    Accepts both spellings of the per-card UUID (`uuid` and `device_uuid`) because
    the two exist in this codebase's history and in the probe's raw row shape. One
    canonical reader means a saved entry and a live reading can never disagree
    about the spelling of a field that did not change.
    """
    e = entry or {}
    return HardwareIdentity(
        backend=str(e.get("backend") or ""),
        vendor_id=_norm(e.get("vendor_id")),
        device_id=_norm(e.get("device_id")),
        uuid=_norm(e.get("uuid") if e.get("uuid") is not None
                   else e.get("device_uuid")),
        name=str(e.get("name") or ""),
        driver_version=_norm(e.get("driver_version")),
    )


def calibration_key(model: str, quant: str, identity: HardwareIdentity) -> str:
    """The calibration key: model x quant x backend x hardware identity.

    `backend` appears in both the key text and the digest because they answer
    different questions — the text keeps the file readable, the digest keeps two
    cards that share a vendor/device pair from colliding.
    """
    return f"{model}:{quant}:{identity.backend}:{identity.digest}"


def hard_mismatch(entry: dict, identity: HardwareIdentity) -> str | None:
    """Why a cached entry is about DIFFERENT HARDWARE, or None.

    This is the confident case: a different card, so the cached throughput is not
    this machine's and must not be shown. Invalidate and re-measure; never error —
    a cache miss is not a failure, and refusing to run because a calibration is
    for another GPU would make the tool worse, not safer.
    """
    was = entry.get("hardware") or {}
    if not was:
        return None                      # pre-identity entry: read leniently
    if not identity.is_gpu:
        return None                      # CPU entries have no card to mismatch
    old = from_dict(was)
    for label, prev, now in (("backend", old.backend, identity.backend),
                             ("vendor_id", old.vendor_id, identity.vendor_id),
                             ("device_id", old.device_id, identity.device_id),
                             ("uuid", old.uuid, identity.uuid)):
        if prev and now and prev != now:
            # The uuid is the one component Khronos warns may change when a card
            # moves slots ("not intended to be usable as a serializable
            # persistent identifier"), so a uuid-only difference is reported as a
            # mismatch but the caller is told it is the weak component.
            extra = (" (the device UUID is the component Khronos warns can change "
                     "if the card moves slots)" if label == "uuid" else "")
            return f"measured on {label} {prev}, now {now}{extra}"
    return None


def _same_engine(a: str, b: str) -> bool:
    """Whether two engine strings name the same build.

    Delegates to `engine_build.same_build`, which exists because engine strings come
    from two producers: older entries recorded the bare manifest version ("b9867")
    and current code records the measured identity ("b9867+152d337fa"). A plain `!=`
    would call those different and discard every valid calibration on upgrade.

    Falls back to a strict compare if `engine_build` cannot be imported, so a partial
    install degrades to the old behaviour rather than raising inside a staleness
    check.
    """
    try:
        from .engine_build import same_build
        return same_build(a, b)
    except Exception:
        return a == b


def soft_reasons(entry: dict, identity: HardwareIdentity, *,
                 engine: str = "", ctx: int = 0) -> list[str]:
    """Conditions that changed on the SAME card. Stale, not wrong.

    Kept separate from `hard_mismatch` because the two deserve different
    treatment: a different card means do not show the number, a driver update
    means show it as stale and re-measure when convenient.
    """
    out: list[str] = []
    old = from_dict(entry.get("hardware"))
    if old.driver_version and identity.driver_version \
            and old.driver_version != identity.driver_version:
        out.append(f"measured on driver {old.driver_version}, "
                   f"now {identity.driver_version}")
    if engine and entry.get("engine") and not _same_engine(entry["engine"], engine):
        out.append(f"measured on engine {entry['engine']}, now {engine}")
    if ctx and entry.get("ctx") and entry["ctx"] != ctx:
        out.append(f"measured at ctx {entry['ctx']:,}, now {ctx:,}")
    return out
