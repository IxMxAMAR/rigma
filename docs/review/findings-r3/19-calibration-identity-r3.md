# R3-CAL-1 — a calibration is keyed by the hardware it was measured on

**Severity: HIGH (silent, and it is wrong in the confident direction)**
**Status: fixed** — `src/rigma/hwid.py`, `bench.py`, `probe.py`, `models.py`,
`resolve.py`, `serve.py`, `cli.py`; tests in `test_r3_hwid.py` and
`test_r3_calibration_identity.py`.

## The defect

The calibration key was `model:quant:backend`. Throughput on a fixed
model+quant+backend is not a property of the model — it is a property of the
machine, and the key said nothing about the machine.

Measured, llama.cpp CUDA scoreboard, Llama-2-7B Q4_0:

* nine distinct **24 GB** cards span tg128 **54.74 → 189.33**, so VRAM size is not
  an identity proxy;
* RTX 3090 → 4090 is pp512 **+132%**; A100 → H100 **+105%**;
* merely having 4 GPUs visible cut a 7900 XTX's pp512 from 2,023 to 1,545
  (**−24%**), and an Optimus laptop measured Vulkan prefill *"around 3500 instead
  of 6000 t/s"* without correct device-routing env vars — **42% from device
  selection alone**.

The failure that matters is the **silent** one: a 3090 inherits a 4090's stored
number, and nothing about the number looks wrong. Rigma's UI then reports a
throughput the machine cannot reach, and `bench.verdict` judges a correctly-running
machine as under-performing.

## What the key is now

`model:quant:backend:<identity>`, where the identity is a truncated sha256 of
`backend|vendorID|deviceID|deviceUUID` — only the components that identify the
**card**.

**Deliberately not in the key**, each for a measured reason:

| excluded | why |
|---|---|
| `pipelineCacheUUID` | identifies *"a compatible device and driver combination"*, so it moves on a driver update and would discard a still-valid measurement |
| `deviceName` | embeds the driver (`… (RADV NAVI31)` vs `… (AMD proprietary driver)`); it is a label, not an identity — which is why Ollama has to match it heuristically |
| driver version | real (a driver update made non-FA PP 5% and FA 15% faster) but `driverVersion` is implementation-defined, so its encoding is not comparable across vendors; recorded as a **field** and reported as a **soft** reason |
| VRAM size, GPU count | both change without the card changing |

`deviceID` alone is not enough: Vulkan specifies *"the same device ID should be
used for all physical implementations of that device version"*, so four 7900 XTXes
report identically. `deviceUUID` is the per-card value. It is read through
`vkGetPhysicalDeviceProperties2` + a `VkPhysicalDeviceIDProperties` pNext chain,
added to `probe.py` with the same compile-time ABI size assertion the properties
struct already had. Verified live on the owner's machine:

```
{'vendor_id': '0x1002', 'device_id': '0x7550',
 'device_uuid': '00000000040000000000000000000000', 'driver_version': '0x800184'}
```

## Hard vs soft, which is the part that matters

Identity is **necessary but not sufficient** — power limits alone moved a 3090's
pp512 from 4,175 to 5,406 (**+29%**) with no identity change, and the same 7900 XTX
varies **2.4×** across driver/OS/overclock. So the two questions are never
conflated:

* **hard mismatch** — a different card. The cached number is about other hardware
  and must not be shown. Invalidate and re-measure; never error, because a cache
  miss is not a failure and refusing to run because a calibration belongs to
  another GPU would make the tool worse, not safer.
* **soft reasons** — the same card under new conditions (driver, engine, ctx).
  Stale, not wrong. Re-measure, do not discard.

A UUID-only difference is reported as hard but the message says the UUID is the
component Khronos warns can change if a card moves slots — so the weak component
announces that it is weak.

## Migration

Every existing calibration on every machine predates identity. Reads fall back to
the legacy `model:quant:backend` key, so the machine that wrote an entry still
finds it; `is_calibrated` additionally refuses a legacy entry whose recorded
hardware is a different card. A legacy entry that is still valid is adopted onto
the identity key on first use. `clear_calibration` clears **both** keys — leaving
the legacy entry behind would make the next lookup find it and report the model as
still calibrated.

Verified against the owner's real 9-entry `calibration.json`: all 9 still resolve,
all still report `calibrated`.

The file is pruned to the newest entry per identity, so it cannot grow without
bound across GPU swaps and driver experiments. Entries with no recorded hardware
are kept — they cannot be attributed to an identity, and guessing would delete
them.

## A second defect this uncovered: the engine's "free" is not always true

Chasing the user's note that ComfyUI was running during the earlier measurements
produced a finding that changes how the memory oracle must be read.

With a real `llama-server` holding ~13 GB of a 16 GB card, the Windows GPU
performance counter read **13,149 MiB dedicated in use**, and `llama-fit-params`
reported at the same moment:

```
common_memory_breakdown_print: | - ROCm0 (RX 9070 XT) | 16304 = 16140 + (899 = 82 + 720 + 97) + -735 |
common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed
```

**16,140 MiB free on a card with 3,155 MiB free, and its own fit verdict said
"fits".** Whatever `hipMemGetInfo` returns on this driver, it is not available
VRAM. Reproduced deliberately under load, where `rigma plan --verify` now prints:

```
verify:  engine measures 359 MiB (model 82 + context 180 + compute 97) against 3917 MiB free of 16304 MiB
         engine's own fit verdict: UNRELIABLE (see below)
         the engine reports 15,416 MiB free but the OS says 3,918 MiB is free
         (12,386 MiB of 16,304 MiB is in use by other processes) — the engine's fit
         verdict was made against the wrong number, so only its accounting is usable here
```

This is the AMD/Windows form of the trap the NVIDIA sysmem-fallback research
described: the allocation succeeds, nothing errors, the model runs over PCIe.

**The consequence is that the engine's fit *verdict* is not evidence on this
platform, and its per-device *accounting* is the only part worth using.** Rigma
therefore cross-checks the engine's `free` against the OS counter
(`probe.gpu_used_mb()`, already used for `vram_used_mb`), believes the OS, rewrites
`free` to the OS figure, re-runs the overcommit check against it, and marks the
verdict UNRELIABLE rather than printing "fits" over a warning. When the two agree
the engine's figure is left exactly as it was — the cross-check corrects a wrong
reading, it does not substitute its own.

## A correction to my own earlier reading

I initially reported that the earlier `--verify` runs were "contaminated by
ComfyUI". That was half wrong and worth stating precisely. `gpu_used_mb()` reads
**residency**, not allocation: with ComfyUI alive but idle it reads **22 MiB**, and
during an active generation it read **12,586 MiB**. So the earlier runs were
measured while ComfyUI held nothing, and their `free` figures were right at that
instant. The 16,140-vs-3,155 measurement is the one that is valid and it is the one
that matters. The same property is why `vram_used_mb` is treated as a *soft*
calibration signal with a 700 MiB tolerance rather than a hard key.

## Verification

* 21 tests in `test_r3_hwid.py` (the identity rules) and 18 in
  `test_r3_calibration_identity.py` (the integration), plus 6 for the
  free-memory cross-check in `test_r3_memtruth.py`.
* The cross-check was verified live under real VRAM contention, not only mocked.
* The owner's 9 real calibration entries were read with the new code and all 9
  still resolve.
* Identity is verified against the real card via real Vulkan enumeration.

## Open

1. **`vkGetPhysicalDeviceProperties2` may be missing** on a very old loader, in
   which case `device_uuid` is `""` and identity degrades to vendor+device. The
   digest stays stable; it is just weaker. Not observed here.
2. **The AMD gfx target is not in the key.** The research recommends it for ROCm
   (`HSA_OVERRIDE_GFX_VERSION` can change the effective kernel target, which governs
   speed). `arch` is already in `GpuInfo`, so this is a small addition, but I did
   not add it because I could not measure a case where it differs from the
   vendor/device pair on this machine — and a key component I cannot demonstrate is
   a key component I would be guessing at.
3. **NVIDIA was not exercised at all.** The UUID path for CUDA goes through
   `_nvml_gpus()`, which I did not touch and could not test on an AMD machine. The
   Vulkan path is what is verified.
