"""Ask llama.cpp what it will actually use, instead of guessing.

WHY THIS EXISTS

`resolve.py` predicts VRAM with a closed-form formula over GGUF metadata:

    kv_bytes_per_token = full_attn_layers * kv_heads * head_dim * (bytes_k + bytes_v)

That is a *guess*. It is never compared against reality, and its own source says
one of its terms "errs toward overcommitting the card rather than refusing a
plan". A guess that overcommits is exactly how you get an OOM on the long-context
turn rather than a refusal at plan time.

Meanwhile the pinned llama.cpp build ships `llama-fit-params`, which does not
guess. It performs a no-alloc dummy load (`no_alloc = true`,
`load_mode = LLAMA_LOAD_MODE_NONE`), asks the backend for exact per-device
accounting, and compares that against REAL free device memory:

    common_memory_breakdown_print: | memory breakdown [MiB] | total    free     self   model   context   compute    unaccounted |
    common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + ( 899 =    82 +     720 +      97) +        -735 |
    common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
    common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed

`model + context + compute` is the number Rigma has been estimating by hand, and
`free` is the number it has been assuming. This module reads both.

WHY NOT A LEARNED CORRECTION INSTEAD

The obvious alternative — measure VRAM after launch, store the delta, correct
future predictions — was researched and rejected on evidence:

  * Windows/WDDM exposes no per-process GPU memory at all, and a device-level
    sample is a sample, never a peak.
  * AMD's own tracker has a measured case where `amd-smi` USED_VRAM moved 1 MB
    for a 2 GB allocation (hipMemGetInfo showed 2058 MB).
  * `--fit` itself returns different answers on the same machine and model before
    and after a sleep/resume (llama.cpp#26401), so a persisted delta would be
    applied to a baseline that already moved.

The oracle is per-launch and reads the live device, so none of that applies.

WHAT IT DOES NOT DO

It is a VERIFICATION, not a planner. llama.cpp only auto-adjusts arguments the
caller left UNSET, and Rigma sets `-c` and `-ngl` explicitly, so the oracle
reports on Rigma's chosen plan rather than changing it. That is the useful
direction: Rigma decides, then asks whether the decision fits.

`llama-fit-params` also ABORTS on a request that cannot be met at all (measured:
`-c 2000000` on a 16 GB card printed the breakdown, then died without a fit
line). That is reported here as `ok=False` with whatever was parsed, never as a
crash and never as agreement.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# The oracle writes fitted CLI args to stdout and its reasoning to stderr.
# stderr is where the numbers are.
_TIMEOUT_S = 120.0

# `|   - ROCm0 (RX 9070 XT) | 16304 = 16140 + ( 899 =    82 +     720 +      97) +        -735 |`
# Device lines are the ones carrying a parenthesised self/model/context/compute
# group; the `Host` line has the same first columns and no such group, so
# requiring the group is what excludes it. `unaccounted` is signed — it was
# -735 on a real run and -45832 on an impossible one.
_DEV_RE = re.compile(
    r"\|\s*-\s*(?P<dev>.+?)\s*\|\s*(?P<total>-?\d+)\s*=\s*(?P<free>-?\d+)\s*\+\s*"
    r"\(\s*(?P<self>-?\d+)\s*=\s*(?P<model>-?\d+)\s*\+\s*(?P<context>-?\d+)\s*\+\s*"
    r"(?P<compute>-?\d+)\s*\)\s*\+\s*(?P<unaccounted>-?\d+)\s*\|")
_PROJECTED_RE = re.compile(
    r"projected to use (?P<use>-?\d+) MiB of device memory vs\. "
    r"(?P<free>-?\d+) MiB of free device memory")
_LEAVE_RE = re.compile(
    r"will leave (?P<leave>-?\d+) >= (?P<target>-?\d+) MiB of free device memory")
_NEEDS_RE = re.compile(r"cannot meet free memory target of (?P<target>-?\d+) MiB")
# The oracle cannot measure a model it cannot load, and the reason matters: a
# quantised GGUF whose ggml type the pinned build does not know is a REAL,
# actionable defect ("upgrade the engine pin"), not a parsing problem. Measured
# on the owner's machine: Ternary-Bonsai-2-27B-PQ2_0 is ggml type 142, the pin is
# b9867 which accepts [0, 42), so Rigma planned a model its own engine cannot
# serve. Reporting that as "printed no memory breakdown" would hide it.
_LOAD_FAIL_RE = re.compile(
    r"(?P<why>(?:failed to load model|invalid ggml type|failed to read tensor"
    r" info|unknown model architecture|error loading model)[^\n]*)", re.I)


@dataclass
class DeviceMemory:
    """One device's real memory accounting, in MiB, as llama.cpp reports it."""
    device: str
    total: int
    free: int
    model: int
    context: int
    compute: int
    unaccounted: int

    @property
    def self_mb(self) -> int:
        """What this process will hold: the three terms it allocates."""
        return self.model + self.context + self.compute


@dataclass
class FitResult:
    """What the oracle said, and whether it committed to an answer."""
    ok: bool
    devices: list[DeviceMemory] = field(default_factory=list)
    projected_mb: int | None = None
    free_mb: int | None = None
    leaves_mb: int | None = None
    target_mb: int | None = None
    fitted_args: str = ""
    reason: str = ""
    load_error: str = ""

    @property
    def primary(self) -> DeviceMemory | None:
        return self.devices[0] if self.devices else None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "projected_mb": self.projected_mb,
            "free_mb": self.free_mb,
            "leaves_mb": self.leaves_mb,
            "target_mb": self.target_mb,
            "fitted_args": self.fitted_args,
            "load_error": self.load_error,
            "devices": [vars(x) | {"self_mb": x.self_mb} for x in self.devices],
        }


def parse_fit_output(stderr: str, stdout: str = "") -> FitResult:
    """Parse the oracle's output. Pure, so the fragile part is testable.

    Never raises on a format it does not recognise: a llama.cpp upgrade that
    changes these strings must degrade to `ok=False` with a reason, because a
    parser that silently stops matching and reports "no problem found" is worse
    than no check at all.
    """
    devices = [DeviceMemory(
        device=m.group("dev"), total=int(m.group("total")),
        free=int(m.group("free")), model=int(m.group("model")),
        context=int(m.group("context")), compute=int(m.group("compute")),
        unaccounted=int(m.group("unaccounted")),
    ) for m in _DEV_RE.finditer(stderr or "")]

    proj = _PROJECTED_RE.search(stderr or "")
    leave = _LEAVE_RE.search(stderr or "")
    needs = _NEEDS_RE.search(stderr or "")
    # `will leave X >= Y` and `cannot meet ... Y` both name the target margin;
    # the latter means it could NOT reach it, which is a refusal, not a fit.
    target = None
    if leave:
        target = int(leave.group("target"))
    elif needs:
        target = int(needs.group("target"))

    ok = bool(devices) and proj is not None and needs is None
    load = _LOAD_FAIL_RE.search(stderr or "")
    load_error = load.group("why").strip() if load else ""
    # R3-MEM-1, the Windows trap. WDDM (and NVIDIA's Sysmem Fallback Policy,
    # driver branch 536.40+) lets an allocation LARGER than dedicated VRAM
    # SUCCEED out of system RAM. The model then runs over PCIe at a fraction of
    # the speed and llama.cpp prints NO error — so on Windows "the server
    # started" is not evidence that it fits, and a verdict resting on the
    # allocation succeeding would be wrong in the SILENT direction.
    #
    # So Rigma does the arithmetic itself: if what the engine says it will hold
    # exceeds what it says is free, that is a refusal regardless of what the fit
    # step concluded. This is the one check the driver cannot fool by handing out
    # memory the card does not have.
    over = [d for d in devices if d.self_mb > d.free]
    if load_error:
        # A model the engine cannot load is the most actionable answer this
        # module can give, so it outranks the generic wording.
        reason = f"the engine cannot load this model: {load_error}"
    elif over:
        d0 = over[0]
        reason = (f"this does not fit: the engine will hold {d0.self_mb} MiB on "
                  f"{d0.device} but only {d0.free} MiB is free (on Windows this "
                  f"can still 'succeed' by paging to system RAM, which is why it "
                  f"is refused here rather than trusted)")
        ok = False
    elif not devices:
        reason = "the engine's fit oracle printed no memory breakdown"
    elif needs is not None:
        reason = (f"the engine cannot meet its free-memory target of "
                  f"{target} MiB with these arguments")
    elif proj is None:
        reason = "the engine printed a breakdown but did not commit to a fit"
    else:
        reason = ""

    return FitResult(
        ok=ok and not load_error, devices=devices,
        projected_mb=int(proj.group("use")) if proj else None,
        free_mb=int(proj.group("free")) if proj else None,
        leaves_mb=int(leave.group("leave")) if leave else None,
        target_mb=target,
        fitted_args=(stdout or "").strip(),
        reason=reason,
        load_error=load_error,
    )


def parse_fit_jsonl(stdout: str) -> FitResult:
    """Parse the `--log-jsonl` memory breakdown, when the build emits one.

    Newer llama.cpp has `--log-jsonl`, under which `common_memory_breakdown_print`
    emits a structured object instead of the aligned table:

        {"type":"fit_memory_breakdown","data":{"unit":"MiB","rows":[
           {"kind":"device","name":"CUDA0",...}]}}

    That is strictly better than scraping the table, so it is preferred whenever
    it is present.

    HONEST STATUS: the pinned build (b9867) does NOT have `--log-jsonl` —
    verified by running it, which answers `error: invalid argument`. So this path
    is written from the documented schema and is exercised only by tests against
    that schema; it has never run against a real emitting build. The stderr
    table parser below is the one verified live. If a future pin gains the flag,
    this becomes the primary path and the table becomes the fallback — which is
    why both are kept rather than one replacing the other.

    Returns `ok=False` with no devices when no such line is present, so a build
    without JSONL simply falls through to the table parser.
    """
    import json
    devices: list[DeviceMemory] = []
    projected = free = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "fit_memory_breakdown":
            continue
        data = obj.get("data") or {}
        for row in data.get("rows") or []:
            if not isinstance(row, dict) or row.get("kind") != "device":
                continue
            def _i(k):
                try:
                    return int(row.get(k) or 0)
                except Exception:
                    return 0
            devices.append(DeviceMemory(
                device=str(row.get("name") or row.get("description") or "?"),
                total=_i("total"), free=_i("free"), model=_i("model"),
                context=_i("context"), compute=_i("compute"),
                unaccounted=_i("unaccounted")))
    if not devices:
        return FitResult(ok=False, reason="no JSONL memory breakdown in the output")
    if projected is None:
        projected = devices[0].self_mb
    if free is None:
        free = devices[0].free
    return FitResult(ok=True, devices=devices, projected_mb=projected,
                     free_mb=free,
                     reason="") if projected is not None else FitResult(
        ok=False, devices=devices, reason="JSONL breakdown had no totals")


def parse_any(stderr: str, stdout: str = "") -> FitResult:
    """Prefer the structured JSONL breakdown; fall back to the aligned table.

    Both are tried because the two exist in different llama.cpp versions and
    neither is present in every build. Whichever answers is used, and if neither
    does the result is `ok=False` with a reason — never a silent pass.
    """
    structured = parse_fit_jsonl(stdout)
    if structured.devices:
        # the table may still carry the fit decision, which JSONL does not
        table = parse_fit_output(stderr, stdout)
        structured.projected_mb = table.projected_mb or structured.projected_mb
        structured.leaves_mb = table.leaves_mb
        structured.target_mb = table.target_mb
        structured.fitted_args = table.fitted_args
        if table.reason and "cannot meet" in table.reason:
            structured.ok = False
            structured.reason = table.reason
        return structured
    return parse_fit_output(stderr, stdout)


def fit_params_bin(server_exe: str | Path) -> Path | None:
    """The oracle that ships beside a given `llama-server`, or None.

    Resolved from the server path rather than from a manifest entry because the
    oracle is part of the same build and must never be a second thing to pin,
    download or verify. `None` means "this build has no oracle" — an older pin,
    or a build where the tool was not packaged — and every caller must treat that
    as "no evidence", not as "fits".
    """
    p = Path(server_exe)
    name = "llama-fit-params.exe" if p.suffix.lower() == ".exe" else "llama-fit-params"
    cand = p.with_name(name)
    return cand if cand.exists() else None


def run_fit(server_exe: str | Path, argv: list[str], *,
            timeout: float = _TIMEOUT_S,
            popen=subprocess.run) -> FitResult:
    """Run the oracle with `argv` (the flags Rigma is about to launch with).

    `argv` should carry the model and the arguments being verified — `-c`,
    `-ngl`, `-b`, `-ub`, `-fa`, cache types — because the oracle reports on
    exactly what it is given and auto-adjusts only what is absent.

    A non-zero exit is NOT an error here: the oracle aborts on a request it
    cannot meet, and the breakdown it printed before aborting is the evidence
    that says so. Only a missing binary or a launch failure is `ok=False` with
    no devices.
    """
    exe = fit_params_bin(server_exe)
    if exe is None:
        return FitResult(ok=False, reason=(
            "this engine build has no llama-fit-params, so Rigma cannot verify "
            "the plan against the real device"))
    try:
        cp = popen([str(exe), *argv], capture_output=True, text=True,
                   timeout=timeout)
    except Exception as e:                       # missing dll, timeout, ENOENT
        return FitResult(ok=False, reason=f"the fit oracle could not run: {e}")
    res = parse_any(cp.stderr or "", cp.stdout or "")
    if not res.ok and not res.reason:
        res.reason = f"the fit oracle exited {cp.returncode}"
    return res


def compare(plan_mb: float, result: FitResult) -> str | None:
    """Rigma's own estimate vs the engine's measurement, or None if they agree.

    Returns a human sentence naming BOTH numbers when they disagree enough to
    matter. The threshold is deliberately generous: the two are not the same
    quantity — Rigma's figure is weights+KV against a reserved budget, the
    engine's is a real allocation against real free memory — so a small gap is
    expected and only a large one is evidence of a wrong model of the world.
    """
    d = result.primary
    if d is None:
        return None
    # 15% or 512 MiB, whichever is larger: below that the difference is terms
    # Rigma legitimately does not model (compute buffer, page alignment).
    slack = max(512.0, plan_mb * 0.15)
    if abs(d.self_mb - plan_mb) <= slack:
        return None
    direction = "more" if d.self_mb > plan_mb else "less"
    return (f"Rigma planned {plan_mb:.0f} MiB but the engine measures "
            f"{d.self_mb} MiB for the same arguments ({direction} by "
            f"{abs(d.self_mb - plan_mb):.0f} MiB: model {d.model} + context "
            f"{d.context} + compute {d.compute}).")


def planned_mb(plan) -> float:
    """What Rigma's own arithmetic says the plan will hold, in MiB.

    Deliberately the SAME two terms the resolver budgets — the quantised weights
    plus the KV cache it computed — and deliberately NOT including the compute
    buffer, which the resolver does not model at all. Excluding it is what makes
    `compare`'s disagreement meaningful: if Rigma is under-counting the compute
    buffer, that gap shows up as a difference instead of being hidden by adding
    an estimate of it here.

    The spec comes from the REGISTRY, not from the plan: `RunPlan` carries
    `model_slug`, `gguf`, `backend`, `flags`, `origin` and `explain` — there is no
    `spec` field, and an earlier version of this function read `plan.spec` behind
    a `hasattr`, so it returned 0.0 for every real plan and compared nothing.
    A silent zero here would make every plan look catastrophically under-budgeted.
    """
    from .resolve import kv_bytes_per_token, swa_kv_bytes
    from .registry import Registry
    spec = Registry.load().models.get(getattr(plan, "model_slug", ""))
    if spec is None:
        raise ValueError(
            f"no registry spec for {getattr(plan, 'model_slug', '')!r}, so "
            "Rigma cannot compute what it planned")
    f = plan.flags
    ctx = int(getattr(f, "ctx", 0) or 0)
    k = getattr(f, "cache_type_k", "") or "f16"
    v = getattr(f, "cache_type_v", "") or "f16"
    kv = (ctx * kv_bytes_per_token(spec, k, v)
          + swa_kv_bytes(spec, k, v, ctx)) / 2**20
    return plan.gguf.bytes / 2**20 + kv


def fit_argv(plan, model_path: str, *, backend_args: list[str] | None = None
             ) -> list[str]:
    """The argv to hand the oracle: the model plus the flags being verified.

    Built from the plan's own flags rather than from a hand-written list, so a
    flag added to `server_args` is verified automatically. `-lv 4` is what makes
    the build print its memory breakdown at all.
    """
    f = plan.flags
    argv = ["-m", str(model_path)]
    if int(getattr(f, "ctx", 0) or 0):
        argv += ["-c", str(int(f.ctx))]
    if int(getattr(f, "ngl", -1)) >= 0:
        argv += ["-ngl", str(int(f.ngl))]
    if int(getattr(f, "batch", 0) or 0) > 0:
        argv += ["-b", str(int(f.batch))]
    if int(getattr(f, "ubatch", 0) or 0) > 0:
        argv += ["-ub", str(int(f.ubatch))]
    if getattr(f, "flash_attn", ""):
        argv += ["-fa", str(f.flash_attn)]
    for name, flag in (("cache_type_k", "--cache-type-k"),
                       ("cache_type_v", "--cache-type-v")):
        val = getattr(f, name, "")
        if val:
            argv += [flag, str(val)]
    if int(getattr(f, "n_cpu_moe", 0) or 0) > 0:
        argv += ["--n-cpu-moe", str(int(f.n_cpu_moe))]
    # --parallel 2 is what Rigma actually launches with, and the oracle reports
    # on what it is given, so verifying without it would verify a different plan.
    argv += ["--parallel", "2"]
    argv += list(backend_args or [])
    argv += ["-lv", "4"]
    return argv


def verify_plan(plan, model_path: str, server_exe, *,
                backend_args: list[str] | None = None,
                popen=subprocess.run) -> tuple[FitResult, str | None]:
    """Run the oracle for a plan. Returns (result, disagreement-or-None).

    The caller decides what to do with a disagreement. This function never
    refuses anything itself: a wrong estimate is worth reporting, and whether it
    is worth blocking a launch is a policy question that belongs with the launch.
    """
    argv = fit_argv(plan, model_path, backend_args=backend_args)
    res = run_fit(server_exe, argv, popen=popen)
    return res, compare(planned_mb(plan), res)

