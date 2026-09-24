"""The ENGINE RUNTIME axis: llama.cpp vs vLLM.

THREE DIFFERENT THINGS IN THIS CODEBASE ARE CALLED "BACKEND". Do not conflate
them, and do not add a fourth name for any of them:

  1. the llama.cpp COMPUTE backend  — "vulkan" | "cuda" | "rocm" | "cpu".
     `models.RunPlan.backend`, `server_ops.available_backends`,
     `data/engines.json`'s "windows/vulkan" keys.
  2. the HARNESS backend            — "native" | "dsh" | "mcode".
     `harness.py`, `LaunchDefaults.backend` in the UI.
  3. the ENGINE RUNTIME (this module) — "llamacpp" | "vllm". WHICH PROGRAM
     serves the OpenAI API. This is the axis vLLM adds.

A future reader WILL conflate (1) and (3), because both are spelled "backend"
in the CLI and both are per-launch choices. So this module never says
"backend" for (3): it says engine runtime, or engine kind.

WHY THIS MODULE EXISTS AT ALL, given that vLLM cannot run on the machine this
was written on: vLLM's support matrix is not a boolean, it is a tuple of (OS,
GPU vendor, GPU arch, Python, ROCm/CUDA version) with at least one SILENT
failure mode in it — on an AMD host, any Python other than 3.12 makes the
installer fall back to the CUDA wheel, and the mistake only surfaces at
runtime as `libcudart.so: cannot open shared object file`. "Not installed" and
"installed but this OS is unsupported" and "installed and runnable" are three
different answers and a user needs the right one. Everything here is
dependency-free and side-effect free at import time ON PURPOSE: importing
`vllm` or `torch` at module scope would break every `rigma` command on a
machine that does not have them (i.e. this one).

Sources for every constant below, all read 2026-09-25 from vllm-project/vllm
`main`; the full argument is in docs/design/2026-09-25-vllm-engine-spec.md.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- the engine runtime axis, named once ------------------------------------

LLAMACPP = "llamacpp"
VLLM = "vllm"
ENGINE_RUNTIMES = (LLAMACPP, VLLM)
DEFAULT_ENGINE_RUNTIME = LLAMACPP

# --- vLLM's measured support matrix ----------------------------------------
# docs/getting_started/installation/gpu.md: "OS: Linux", "Python: 3.10 -- 3.13",
# and the note that vLLM "does not support Windows natively" (WSL, or the
# community fork, are the only Windows routes).
VLLM_DOC_REQUIREMENTS_URL = (
    "https://docs.vllm.ai/en/latest/getting_started/installation/gpu/")
VLLM_DOC_SERVE_ARGS_URL = (
    "https://docs.vllm.ai/en/latest/configuration/serve_args/")
VLLM_DOC_GGUF_URL = (
    "https://docs.vllm.ai/en/latest/features/quantization/gguf/")

VLLM_PYTHON_MIN = (3, 10)
VLLM_PYTHON_MAX = (3, 13)

# docs/getting_started/installation/gpu.rocm.inc.md: "vLLM supports AMD GPUs
# with ROCm 6.3 or above. Pre-built wheels are available for ROCm 7.0 and ROCm
# 7.2.1." The prebuilt-wheel table says Python Version 3.12 for BOTH variants,
# with a warning that any other Python "will silently fall back to the CUDA
# wheel from PyPI, which will fail on AMD GPUs with errors like
# `libcudart.so: cannot open shared object file`". glibc >= 2.35 for both.
ROCM_WHEEL_PYTHON = (3, 12)
ROCM_WHEEL_GLIBC = (2, 35)
ROCM_MIN_VERSION = (6, 3)
ROCM_INDEX_URL = "https://wheels.vllm.ai/rocm/"

# docs/getting_started/installation/gpu.rocm.inc.md, "GPU:" — the whole
# supported list, verbatim: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950),
# Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201),
# Ryzen AI MAX / AI 300 Series (gfx1151/1150).
SUPPORTED_ROCM_ARCHES = frozenset({
    "gfx90a", "gfx942", "gfx950",
    "gfx1100", "gfx1101",
    "gfx1200", "gfx1201",
    "gfx1150", "gfx1151",
})

# docs/getting_started/installation/gpu.cuda.inc.md: "GPU: compute capability
# 7.5 or higher". The prebuilt wheels are compiled against CUDA 12.9, with
# 12.8 and 13.0 variants published too.
CUDA_MIN_COMPUTE = (7, 5)

# llama.cpp server flags. NONE of these exist in `vllm serve`, and a vLLM
# launch that carried any of them would die in argparse before loading a
# weight. Kept as data so `vllm_argv` can be asserted against it rather than
# against a hand-written list inside the test.
LLAMACPP_ONLY_FLAGS = (
    "-m", "-ngl", "-c", "-b", "-ub", "-fa",
    "--n-cpu-moe", "--cache-type-k", "--cache-type-v",
    "--parallel", "--kv-unified", "--cache-reuse", "--checkpoint-min-step",
    "--slot-save-path", "--spec-type", "--spec-draft-n-max",
    "--reasoning", "--reasoning-budget", "--reasoning-budget-message",
    "--mmproj", "--chat-template-file", "--no-mmap", "--tensor-split",
)

# Flags a vLLM V0 launch used that the V1 CLI does not accept. Emitting one is
# the same class of bug as emitting an llama.cpp flag: argparse exits before
# the model is read. `--disable-log-requests` is the one people reach for from
# memory; V1 replaced it with `--enable-log-requests` (its inverse), and
# `swap_space` is no longer a CacheConfig field at all.
VLLM_RETIRED_FLAGS = ("--swap-space", "--disable-log-requests")


# --- probes -----------------------------------------------------------------
# Every one of these is a separate function so a test can replace exactly the
# fact it is about, and so nothing here reads the real machine at import time.

def _os_name() -> str:
    return {"Windows": "windows", "Linux": "linux",
            "Darwin": "darwin"}.get(platform.system(), platform.system().lower())


def _python_version() -> tuple[int, int]:
    return (sys.version_info[0], sys.version_info[1])


def _vllm_executable() -> str | None:
    """The `vllm` console script, if it is on PATH."""
    return shutil.which("vllm")


def _vllm_module_present() -> bool:
    """Whether `import vllm` would resolve. find_spec only — never imports it."""
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError, AttributeError):
        return False


_GFX = re.compile(r"\b(gfx[0-9a-f]{3,4})\b", re.I)


def _gpu_arch() -> str | None:
    """The ROCm gfx target of the first AMD GPU `rocminfo` reports, or None.

    ROCm only, and only on Linux: this is the identifier vLLM's own support
    list is written in, and `rocminfo` is the tool that list tells you to run
    (`rocminfo | grep gfx`). On Windows there is no rocminfo, and Rigma's own
    registry table names the arch "rdna4" — a family, not a gfx target — so
    this deliberately returns None rather than guessing. Windows is blocked by
    the OS check before the arch is ever consulted, so the None costs nothing
    there; it is reported honestly everywhere else.
    """
    if _os_name() != "linux":
        return None
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    m = _GFX.search(out.stdout or "")
    return m.group(1).lower() if m else None


_ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")


def _rocm_version() -> tuple[int, int] | None:
    """ROCm's version from the install prefix's own stamp file.

    Deliberately NOT from `rocminfo` or a torch import: the question is which
    ROCm the *driver stack* is, and reading a file cannot fail in a way that
    costs ten seconds.
    """
    try:
        raw = _ROCM_VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    m = re.match(r"(\d+)\.(\d+)", raw)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _cuda_present() -> bool:
    """An NVIDIA driver is installed. Presence of the tool only — no version
    parsing, because vLLM ships its own CUDA runtime and only the *driver* has
    to be new enough."""
    return shutil.which("nvidia-smi") is not None


def _glibc_version() -> tuple[int, int] | None:
    """glibc version, or None when unmeasurable.

    `os.confstr` does not exist on Windows at all (AttributeError) and can
    return an empty string, so both are handled; None is reported as
    unverified rather than treated as a pass.
    """
    if _os_name() != "linux":
        return None
    try:
        raw = os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (AttributeError, OSError, ValueError):
        return None
    m = re.search(r"(\d+)\.(\d+)", raw)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _fmt(v: tuple[int, int] | None) -> str | None:
    return None if v is None else f"{v[0]}.{v[1]}"


# --- the verdict ------------------------------------------------------------

@dataclass(frozen=True)
class EngineAvailability:
    """Whether an engine runtime can run HERE, and exactly why not.

    `state` is the machine-readable half and is deliberately finer-grained
    than the boolean: "not-installed", "unsupported-os", "unsupported-python",
    "unsupported-gpu", "unsupported-rocm", "unsupported-libc",
    "unverified-gpu", "no-pinned-build" and "runnable" are all different
    situations with different remedies, and collapsing them into `available`
    is what makes an availability report useless.
    """
    engine: str
    available: bool
    state: str
    reason: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"engine": self.engine, "available": self.available,
                "state": self.state, "reason": self.reason,
                "evidence": dict(self.evidence)}


def vllm_availability() -> EngineAvailability:
    """Can vLLM run on this machine? A full sentence either way.

    Check order is OS, then installed, then accelerator, then Python, then
    arch, then ROCm/glibc. OS first is deliberate: on Windows the answer is
    "Windows is not supported, use WSL" no matter what is installed, and
    answering "not installed" there would send the user to `pip install` —
    a multi-GB download that cannot work. The install state is still reported,
    in the evidence and in the sentence.
    """
    os_name = _os_name()
    py = _python_version()
    exe = _vllm_executable()
    module = _vllm_module_present()
    installed = bool(exe) or module
    arch = _gpu_arch()
    rocm = _rocm_version()
    cuda = _cuda_present()
    glibc = _glibc_version()
    evidence = {
        "os": os_name,
        "python": _fmt(py),
        "vllm_executable": exe,
        "vllm_module": module,
        "installed": installed,
        "gpu_arch": arch,
        "rocm_version": _fmt(rocm),
        "cuda_present": cuda,
        "glibc": _fmt(glibc),
        "rocm_wheel_python": _fmt(ROCM_WHEEL_PYTHON),
        "rocm_min_version": _fmt(ROCM_MIN_VERSION),
        "rocm_wheel_glibc": _fmt(ROCM_WHEEL_GLIBC),
        "supported_rocm_arches": sorted(SUPPORTED_ROCM_ARCHES),
        "docs": VLLM_DOC_REQUIREMENTS_URL,
    }

    def verdict(state: str, reason: str, available: bool = False):
        return EngineAvailability(VLLM, available, state, reason, evidence)

    # 1. The OS. vLLM's own words: "vLLM does not support Windows natively."
    if os_name != "linux":
        label = {"windows": "Windows", "darwin": "macOS"}.get(os_name, os_name)
        present = ("a `vllm` executable or module IS present on this machine, but "
                   "that does not make it supported here" if installed else
                   "vLLM is absent from this machine as well, and installing it "
                   "would not change this answer")
        return verdict("unsupported-os", (
            f"vLLM does not support {label} natively — its stated requirement is "
            f"OS: Linux with Python {_fmt(VLLM_PYTHON_MIN)}-{_fmt(VLLM_PYTHON_MAX)}. "
            "On Windows the supported routes are WSL2 with a Linux distribution "
            "that can see the GPU (plus a WSL ROCm or CUDA driver), or a real "
            f"Linux partition; {present}. See {VLLM_DOC_REQUIREMENTS_URL}"))

    # 2. Is it here at all? A different answer from every one below it.
    if not installed:
        return verdict("not-installed", (
            "vLLM is not installed: no `vllm` executable on PATH and no importable "
            "`vllm` module. Installing it is a multi-GB torch download and, on an "
            "AMD card, it must be the ROCm wheel rather than the CUDA one — see "
            "docs/design/2026-09-25-vllm-engine-spec.md before running pip."))

    # 3. Which accelerator path would it take? The gfx target decides, because
    #    that is the identifier vLLM's ROCm list is written in.
    if arch is not None and arch.startswith("gfx"):
        if py != ROCM_WHEEL_PYTHON:
            return verdict("unsupported-python", (
                f"vLLM is installed but this interpreter is Python {_fmt(py)}, and "
                "the prebuilt ROCm wheels are published for Python "
                f"{_fmt(ROCM_WHEEL_PYTHON)} only. On any other Python the installer "
                "SILENTLY falls back to the CUDA wheel from PyPI, which then fails "
                "on an AMD GPU at server start with `libcudart.so: cannot open "
                "shared object file` — the install looks fine and the failure "
                f"arrives much later. Use a Python {_fmt(ROCM_WHEEL_PYTHON)} "
                f"environment (uv venv --python {_fmt(ROCM_WHEEL_PYTHON)}) and "
                f"install from {ROCM_INDEX_URL}."))
        if arch not in SUPPORTED_ROCM_ARCHES:
            return verdict("unsupported-gpu", (
                f"vLLM is installed and the interpreter is right, but this GPU is "
                f"{arch}, which is not on vLLM's supported ROCm list "
                f"({', '.join(sorted(SUPPORTED_ROCM_ARCHES))}). It may still work "
                "from a source build, but nothing about it is validated upstream. "
                f"See {VLLM_DOC_REQUIREMENTS_URL}"))
        if rocm is not None and rocm < ROCM_MIN_VERSION:
            return verdict("unsupported-rocm", (
                f"vLLM needs ROCm {_fmt(ROCM_MIN_VERSION)} or above and this machine "
                f"has ROCm {_fmt(rocm)} (from {_ROCM_VERSION_FILE}). Upgrade the "
                "driver stack, or use the ROCm prebuilt wheel whose bundled runtime "
                "matches the driver."))
        if glibc is not None and glibc < ROCM_WHEEL_GLIBC:
            return verdict("unsupported-libc", (
                f"The ROCm prebuilt wheels require glibc >= "
                f"{_fmt(ROCM_WHEEL_GLIBC)} and this machine reports "
                f"{_fmt(glibc)}. A newer distribution, or a container, is the "
                "supported route."))
        unverified = []
        if rocm is None:
            unverified.append(
                f"the ROCm version could not be read from {_ROCM_VERSION_FILE}, so "
                f"the {_fmt(ROCM_MIN_VERSION)}+ floor is unverified")
        if glibc is None:
            unverified.append(
                f"glibc could not be measured, so the "
                f"{_fmt(ROCM_WHEEL_GLIBC)}+ floor is unverified")
        tail = (" Unverified: " + "; ".join(unverified) + ".") if unverified else ""
        return verdict("runnable", (
            f"vLLM can run here: Linux, Python {_fmt(py)}, and {arch} is on vLLM's "
            f"supported ROCm list, with a prebuilt wheel published at "
            f"{ROCM_INDEX_URL}.{tail}"), available=True)

    if cuda:
        if not (VLLM_PYTHON_MIN <= py <= VLLM_PYTHON_MAX):
            return verdict("unsupported-python", (
                f"vLLM supports Python {_fmt(VLLM_PYTHON_MIN)}-"
                f"{_fmt(VLLM_PYTHON_MAX)} and this interpreter is {_fmt(py)}."))
        return verdict("runnable", (
            f"vLLM can run here on the CUDA path: Linux, Python {_fmt(py)}, and an "
            "NVIDIA driver is present. vLLM's prebuilt CUDA binaries need compute "
            f"capability {_fmt(CUDA_MIN_COMPUTE)} or higher, which Rigma does not "
            "verify — check the card before trusting this."), available=True)

    return verdict("unverified-gpu", (
        "vLLM is installed on Linux but Rigma cannot tell which GPU it would use — "
        "`rocminfo` reported no gfx target and there is no `nvidia-smi` — so it "
        "cannot say whether this card is supported. That matters more than it "
        "sounds: on an AMD card the prebuilt ROCm wheels are Python "
        f"{_fmt(ROCM_WHEEL_PYTHON)} only, and on any other Python the installer "
        "silently installs the CUDA wheel, which then dies at startup with "
        "`libcudart.so: cannot open shared object file`."))


def llamacpp_availability(registry=None) -> EngineAvailability:
    """llama.cpp, from the same seam the UI already reads.

    Delegates to `server_ops.available_backends` rather than re-deriving the
    manifest path: that function is the one place that knows which compute
    backends this GPU has, which of them Rigma pins a build for, and which are
    already extracted. A second copy of that logic is a second answer.
    """
    from . import server_ops
    try:
        rows = server_ops.available_backends(registry)
    except Exception as e:                       # never fatal to a report
        return EngineAvailability(
            LLAMACPP, False, "manifest-unreadable",
            f"llama.cpp's engine pin could not be read: {e}", {"error": str(e)})
    buildable = [r["name"] for r in rows if r.get("buildable")]
    ready = [r["name"] for r in rows if r.get("ready")]
    evidence = {"compute_backends": rows, "buildable": buildable, "ready": ready}
    if not buildable:
        return EngineAvailability(
            LLAMACPP, False, "no-pinned-build",
            "Rigma has no pinned llama.cpp build for this machine's GPU and OS, so "
            "it could not serve anything. Run `rigma update` to refresh the engine "
            "pin, or `rigma doctor` for the per-backend detail.", evidence)
    note = (f"; {', '.join(ready)} is already downloaded" if ready
            else "; it downloads on first run")
    return EngineAvailability(
        LLAMACPP, True, "runnable",
        "llama.cpp is the default engine runtime: Rigma pins a build for "
        f"{', '.join(buildable)} on this machine{note}.", evidence)


def engine_runtimes(registry=None) -> list[EngineAvailability]:
    """Every engine runtime Rigma knows, with its verdict for this machine."""
    return [llamacpp_availability(registry), vllm_availability()]


# --- the vLLM command line --------------------------------------------------

def vllm_argv(model: str, *, port: int, host: str = "127.0.0.1",
              served_model_name: str = "", max_model_len: int | None = None,
              gpu_memory_utilization: float | None = None,
              tensor_parallel_size: int | None = None,
              quantization: str | None = None, dtype: str | None = None,
              trust_remote_code: bool = False, executable: str = "vllm",
              gguf_plugin: bool = False,
              extra_args: list[str] | None = None) -> list[str]:
    """The exact `vllm serve` command line a vLLM launch would run.

    EVERY flag here was checked against vLLM's own argument definitions, not
    against memory: the serve parser is built by
    `vllm/entrypoints/launchers/cli_args.py` (`FrontendArgs` + `AsyncEngineArgs`)
    and registers each config field as `--field-name`, so `--served-model-name`
    comes from `ModelConfig.served_model_name`, `--max-model-len` from
    `ModelConfig.max_model_len`, `--gpu-memory-utilization` from
    `CacheConfig.gpu_memory_utilization` (default 0.92), `--tensor-parallel-size`
    from `ParallelConfig.tensor_parallel_size`, `--quantization` and `--dtype`
    from `ModelConfig`, and `--host`/`--port` from `FrontendArgs`. The model
    itself is a POSITIONAL argument: `vllm serve [model_tag] [options]`.

    What is deliberately NOT here:
      * anything from `LLAMACPP_ONLY_FLAGS` — argparse would exit before a
        weight was read.
      * `VLLM_RETIRED_FLAGS` — V0 spellings the V1 CLI dropped.
      * `--enable-prefix-caching`: vLLM's automatic prefix caching is ON by
        default (`CacheConfig.enable_prefix_caching = True`), so passing it
        would claim credit for a default and, worse, imply it is optional.
      * `--slot-save-path`: llama.cpp's on-disk KV snapshot has no vLLM
        equivalent. See the spec — a snapshot must never cross engine runtimes.
    """
    if not model or not isinstance(model, str):
        raise ValueError("vLLM needs a model: a HuggingFace repo id "
                         "(Qwen/Qwen3-8B) or a local directory holding HF-format "
                         "safetensors weights")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"port must be 1-65535, got {port!r}")
    if max_model_len is not None:
        # vLLM accepts -1 (and the literal "auto") as "largest that fits".
        if max_model_len != -1 and max_model_len <= 0:
            raise ValueError("max_model_len must be positive, or -1 for 'auto' "
                             f"(vLLM's own sentinel), got {max_model_len!r}")
    if gpu_memory_utilization is not None and not (
            0 < float(gpu_memory_utilization) <= 1):
        raise ValueError(
            "gpu_memory_utilization is a FRACTION of device memory in (0, 1] "
            f"(vLLM's own bound), got {gpu_memory_utilization!r}. Rigma's "
            "GGUF/vram_mb fit math does not apply here — a percentage expressed "
            "as megabytes would be silently clamped to 1.0 by vLLM.")
    if tensor_parallel_size is not None and int(tensor_parallel_size) < 1:
        raise ValueError("tensor_parallel_size must be >= 1, got "
                         f"{tensor_parallel_size!r}")
    if str(model).lower().endswith(".gguf") and not gguf_plugin:
        # vLLM's GGUF page: support is "highly experimental and under-optimized",
        # has migrated to the out-of-tree vllm-gguf-plugin, and still wants an
        # HF config/tokenizer beside the file. Rigma's whole model library is
        # gguf, so the tempting move — hand vLLM the same path llama.cpp gets —
        # is the one that must fail loudly instead of at load time.
        raise ValueError(
            f"vLLM does not load {model!r} the way llama.cpp does: GGUF support in "
            "vLLM is documented as 'highly experimental and under-optimized' and "
            "has moved to the out-of-tree vllm-gguf-plugin, which also needs a "
            "matching HuggingFace config/tokenizer next to the file. Pass "
            "gguf_plugin=True (and the tokenizer) only when that plugin is really "
            f"installed. See {VLLM_DOC_GGUF_URL}")

    argv = [executable, "serve", model, "--host", host, "--port", str(int(port))]
    if served_model_name:
        argv += ["--served-model-name", served_model_name]
    if max_model_len is not None:
        argv += ["--max-model-len", str(max_model_len)]
    if gpu_memory_utilization is not None:
        # :g, not str(): 0.9 must not become "0.90000000000000002"
        argv += ["--gpu-memory-utilization", f"{float(gpu_memory_utilization):g}"]
    if tensor_parallel_size is not None:
        argv += ["--tensor-parallel-size", str(int(tensor_parallel_size))]
    if quantization:
        argv += ["--quantization", quantization]
    if dtype:
        argv += ["--dtype", dtype]
    if trust_remote_code:
        argv += ["--trust-remote-code"]
    argv += list(extra_args or [])
    return argv


# --- choosing one -----------------------------------------------------------

@dataclass(frozen=True)
class EngineRuntimeDecision:
    """Which engine runtime to use, plus why — never just a string.

    `runtime` is what a launch should use; `requested` is what the user asked
    for; a fallback is visible as the two disagreeing, and `reason` says so in
    words. Silently serving llama.cpp to someone who asked for vLLM would be
    the same class of bug as silently serving the CUDA wheel to someone with an
    AMD card.
    """
    runtime: str
    requested: str
    reason: str
    availability: EngineAvailability | None = None


def detect_engine_runtime(requested: str | None = None) -> EngineRuntimeDecision:
    """Pick an engine runtime. llama.cpp unless vLLM was asked for AND works.

    The default is not a preference, it is the only choice that cannot
    surprise: llama.cpp is pinned, downloaded, verified and calibrated by the
    code that already exists, on every platform Rigma supports. vLLM is
    opt-in, and an opt-in that cannot run falls back EXPLICITLY (the decision
    records both the request and the reason) rather than raising, because a
    stored preference for vLLM must not make `rigma up` unrunnable on a machine
    where it used to work.
    """
    want = (requested or "").strip().lower()
    if not want:
        return EngineRuntimeDecision(
            LLAMACPP, "", "no engine runtime was requested, and llama.cpp is "
            "Rigma's default")
    if want not in ENGINE_RUNTIMES:
        raise ValueError(f"unknown engine runtime {requested!r}: must be one of "
                         f"{', '.join(ENGINE_RUNTIMES)}")
    if want == LLAMACPP:
        return EngineRuntimeDecision(LLAMACPP, want, "llama.cpp was requested")
    av = vllm_availability()
    if av.available:
        return EngineRuntimeDecision(
            VLLM, want, "vLLM was requested and this machine can run it", av)
    return EngineRuntimeDecision(
        LLAMACPP, want,
        "vLLM was requested but this machine cannot run it, so llama.cpp is used "
        f"instead: {av.reason}", av)


# --- launching one ----------------------------------------------------------
# The pure parts above are what the tests exercise. This is the thin,
# platform-specific part, kept here so the llama.cpp launch path in runtime.py
# does not gain a second engine's worth of branching.

def _vllm_failure_hint(tail: str) -> str:
    """Turn the two vLLM failures a Rigma user will actually hit into a
    sentence that names the cause. Pure, so it is testable without a GPU."""
    if "libcudart.so" in tail or "libcuda.so" in tail:
        return ("\nvLLM is running the CUDA build on an AMD (or driver-less) host. "
                "The ROCm prebuilt wheels are published for Python "
                f"{_fmt(ROCM_WHEEL_PYTHON)} only, and the installer falls back to "
                "the CUDA wheel on any other Python WITHOUT saying so. Reinstall "
                f"from {ROCM_INDEX_URL} inside a Python "
                f"{_fmt(ROCM_WHEEL_PYTHON)} environment.")
    if ("No available memory for the cache blocks" in tail
            or "OutOfMemoryError" in tail or "CUDA out of memory" in tail):
        return ("\nvLLM could not fit the model plus its KV cache. This is NOT "
                "Rigma's GGUF fit arithmetic — that math is llama.cpp's and does "
                "not transfer. Lower --gpu-memory-utilization or --max-model-len, "
                "or use a smaller quantisation.")
    if "not supported" in tail and "architecture" in tail.lower():
        return ("\nvLLM does not implement this model architecture. Check it "
                "against https://docs.vllm.ai/en/latest/models/supported_models.html")
    return ""


def launch_vllm_server(argv: list[str], port: int, *, timeout: float = 600.0,
                       log_path=None, popen=None,
                       is_healthy=None):
    """Popen a `vllm serve` line and poll `GET /health` until it answers.

    Returns a `runtime.ServerProcess`, the same handle the llama.cpp launch
    returns, so a caller does not branch on which engine it started.

    The timeout default is 600s, not llama.cpp's 300s: vLLM profiles GPU
    memory and captures CUDA graphs at startup, and a first run also downloads
    weights.

    UNVERIFIED ON THIS MACHINE: vLLM cannot run on Windows, so this function
    has never been executed against a real vLLM. The argv it is handed is
    tested; the polling loop is tested against a fake process. Its `/health`
    assumption comes from vLLM's own serve-arg help, which names "/health,
    /metrics, /ping" as the endpoints worth excluding from access logs — a
    live check on Linux is the first thing the next phase must do.
    """
    from . import runtime                      # lazy: keeps import cheap
    popen = popen or subprocess.Popen
    logs = runtime.rigma_home() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path is not None else logs / f"vllm-{port}.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        proc = popen(argv, stdout=log_f, stderr=subprocess.STDOUT)
    sp = runtime.ServerProcess(proc, port, log_path)
    probe = is_healthy or sp.is_healthy
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if probe():
            return sp
        time.sleep(0.5)
    code = proc.poll()
    sp.stop()
    tail = "".join(log_path.read_text(encoding="utf-8",
                                      errors="replace").splitlines(True)[-40:])
    hint = _vllm_failure_hint(tail)
    if code is not None and code != 0:
        raise RuntimeError(
            f"vllm serve exited with {code & 0xFFFFFFFF:#010x} before becoming "
            f"healthy on :{port}.{hint}\n{tail}")
    raise RuntimeError(
        f"vllm serve did not answer GET /health on :{port} within {timeout:.0f}s."
        f"{hint}\n{tail}")
