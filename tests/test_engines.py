"""The engine RUNTIME axis: llama.cpp vs vLLM.

Written against a machine that CANNOT run vLLM (Windows, RX 9070 XT, no vllm,
no torch), which is the point: the availability verdict has to be right about a
machine it cannot test on, and the argv builder has to be right about a program
that is not installed. So every probe is replaced here and nothing in this file
imports vllm, torch, or touches the real GPU.

Each test below fails if the behaviour it names is removed — verified by
mutation, not by inspection (see docs/design/2026-09-25-vllm-engine-spec.md,
"how these tests were checked").
"""
import pytest

from rigma import engines


def _probe(monkeypatch, *, os_name="linux", python=(3, 12), executable=None,
           module=False, arch=None, rocm=None, cuda=False, glibc=(2, 35)):
    """Replace every fact `vllm_availability` reads. One place, so a test that
    forgets a probe cannot accidentally read the real machine."""
    monkeypatch.setattr(engines, "_os_name", lambda: os_name)
    monkeypatch.setattr(engines, "_python_version", lambda: python)
    monkeypatch.setattr(engines, "_vllm_executable", lambda: executable)
    monkeypatch.setattr(engines, "_vllm_module_present", lambda: module)
    monkeypatch.setattr(engines, "_gpu_arch", lambda: arch)
    monkeypatch.setattr(engines, "_rocm_version", lambda: rocm)
    monkeypatch.setattr(engines, "_cuda_present", lambda: cuda)
    monkeypatch.setattr(engines, "_glibc_version", lambda: glibc)


# --- the availability verdict ----------------------------------------------

def test_windows_is_unavailable_and_the_reason_is_wsl(monkeypatch):
    """The failure this prevents: a Windows user reads "not installed", runs
    `pip install vllm`, downloads several GB of torch and a CUDA wheel, and
    only then learns vLLM has no Windows support at all."""
    _probe(monkeypatch, os_name="windows", python=(3, 12))
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-os"
    assert "Windows" in av.reason
    assert "WSL" in av.reason
    # the whole point: it must not send them to pip
    assert "not installed" not in av.reason.lower()


def test_windows_is_still_unsupported_when_vllm_is_actually_present(monkeypatch):
    """'installed but this OS is unsupported' is a different answer from 'not
    installed', and both differ from 'runnable'. Evidence must say which."""
    _probe(monkeypatch, os_name="windows", python=(3, 12), executable="C:/x/vllm.exe")
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-os"
    assert av.evidence["installed"] is True
    assert "WSL" in av.reason


def test_linux_without_vllm_says_not_installed(monkeypatch):
    _probe(monkeypatch, os_name="linux", arch="gfx1201")
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "not-installed"
    assert "not installed" in av.reason.lower()
    assert av.evidence["installed"] is False


def test_linux_python_311_names_the_rocm_wheel_trap(monkeypatch):
    """THE test this module exists for. On an AMD host, any Python other than
    3.12 makes the installer fall back to the CUDA wheel silently; the user
    finds out from `libcudart.so: cannot open shared object file` much later,
    with nothing in the install output to explain it."""
    _probe(monkeypatch, os_name="linux", python=(3, 11), executable="/usr/bin/vllm",
           arch="gfx1201")
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-python"
    assert "3.12" in av.reason
    assert "libcudart.so" in av.reason
    assert "silently" in av.reason.lower()


def test_linux_python_313_on_amd_is_blocked_too(monkeypatch):
    """The trap is not 'older than 3.12', it is 'not exactly 3.12'."""
    _probe(monkeypatch, os_name="linux", python=(3, 13), executable="/usr/bin/vllm",
           arch="gfx1201")
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-python"
    assert "libcudart.so" in av.reason


def test_linux_python_312_on_gfx1201_is_runnable(monkeypatch):
    """The owner's card (RX 9070 XT = gfx1201) IS on vLLM's supported ROCm
    list — the blocker on this machine is Windows, not the GPU."""
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201", rocm=(7, 0), glibc=(2, 39))
    av = engines.vllm_availability()
    assert av.available is True
    assert av.state == "runnable"
    assert "gfx1201" in av.reason


def test_a_module_install_counts_as_installed(monkeypatch):
    """No console script on PATH but `import vllm` resolves is a real install
    (uv/pipx layouts, and the ROCm index's own instructions)."""
    _probe(monkeypatch, os_name="linux", python=(3, 12), module=True,
           arch="gfx1201")
    av = engines.vllm_availability()
    assert av.available is True
    assert av.evidence["vllm_executable"] is None
    assert av.evidence["vllm_module"] is True


def test_a_gfx_target_outside_the_supported_list_is_refused(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1010")                      # RX 5700 XT, not on the list
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-gpu"
    assert "gfx1010" in av.reason


def test_rocm_older_than_6_3_is_refused(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201", rocm=(6, 2))
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-rocm"
    assert "6.3" in av.reason


def test_glibc_below_the_wheel_floor_is_refused(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201", rocm=(7, 0), glibc=(2, 31))
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unsupported-libc"
    assert "2.35" in av.reason


def test_an_unreadable_rocm_version_is_reported_unverified_not_blocked(monkeypatch):
    """None must not be treated as a pass silently — the runnable verdict has
    to say what it could not check."""
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201", rocm=None, glibc=(2, 39))
    av = engines.vllm_availability()
    assert av.available is True
    assert "Unverified" in av.reason
    assert "ROCm version" in av.reason


def test_the_cuda_path_accepts_python_311(monkeypatch):
    """The Python-3.12 rule is the ROCm wheels' rule, not vLLM's. Applying it
    to an NVIDIA host would refuse a configuration vLLM supports."""
    _probe(monkeypatch, os_name="linux", python=(3, 11), executable="/usr/bin/vllm",
           arch=None, cuda=True)
    av = engines.vllm_availability()
    assert av.available is True
    assert av.state == "runnable"


def test_an_unidentifiable_gpu_is_unverified_rather_than_available(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch=None, cuda=False)
    av = engines.vllm_availability()
    assert av.available is False
    assert av.state == "unverified-gpu"
    assert "3.12" in av.reason


# --- the argv builder -------------------------------------------------------

def test_argv_carries_the_model_the_host_and_the_port():
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499,
                             served_model_name="qwen3-8b")
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "11499"
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-8b"


def test_argv_contains_no_llamacpp_only_flag():
    """A single llama.cpp flag makes vLLM exit in argparse before a weight is
    read — the launch fails with a usage message that names a flag Rigma
    printed itself."""
    argv = engines.vllm_argv(
        "Qwen/Qwen3-8B", port=11499, served_model_name="qwen3-8b",
        max_model_len=32768, gpu_memory_utilization=0.9, tensor_parallel_size=2,
        quantization="awq", dtype="bfloat16", trust_remote_code=True)
    assert not set(argv) & set(engines.LLAMACPP_ONLY_FLAGS)
    # and none of the V0 spellings V1 dropped
    assert not set(argv) & set(engines.VLLM_RETIRED_FLAGS)


def test_argv_does_not_claim_credit_for_a_vllm_default():
    """vLLM's automatic prefix caching is on by default. Passing the flag would
    imply it is optional and that Rigma arranged it."""
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499)
    assert "--enable-prefix-caching" not in argv
    assert "--slot-save-path" not in argv      # llama.cpp's KV snapshot: no peer


def test_context_and_memory_flags_appear_only_when_supplied():
    bare = engines.vllm_argv("Qwen/Qwen3-8B", port=11499)
    assert "--max-model-len" not in bare
    assert "--gpu-memory-utilization" not in bare
    assert "--tensor-parallel-size" not in bare

    full = engines.vllm_argv("Qwen/Qwen3-8B", port=11499, max_model_len=16384,
                             gpu_memory_utilization=0.85, tensor_parallel_size=2)
    assert full[full.index("--max-model-len") + 1] == "16384"
    assert full[full.index("--gpu-memory-utilization") + 1] == "0.85"
    assert full[full.index("--tensor-parallel-size") + 1] == "2"


def test_gpu_memory_utilization_is_a_fraction_and_megabytes_are_refused():
    """The single most dangerous reuse in this feature: Rigma's fit math is in
    MB and llama.cpp takes no such flag. vLLM's bound is (0, 1], so a value
    expressed as megabytes is either refused by vLLM or silently clamped to
    1.0 — i.e. it takes the whole card."""
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499,
                             gpu_memory_utilization=0.92)
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.92"
    for bad in (15000, 0, -1, 1.5):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            engines.vllm_argv("Qwen/Qwen3-8B", port=11499,
                              gpu_memory_utilization=bad)


def test_auto_context_uses_vllms_own_sentinel():
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499, max_model_len=-1)
    assert argv[argv.index("--max-model-len") + 1] == "-1"
    with pytest.raises(ValueError, match="max_model_len"):
        engines.vllm_argv("Qwen/Qwen3-8B", port=11499, max_model_len=0)


def test_a_gguf_model_is_refused_unless_the_plugin_is_declared():
    """Rigma's whole model library is gguf, so handing vLLM the same path
    llama.cpp gets is the tempting mistake. vLLM's own page calls GGUF support
    'highly experimental and under-optimized' and moved it to a plugin."""
    with pytest.raises(ValueError, match="gguf"):
        engines.vllm_argv("/models/Qwen3-8B-Q4_K_M.gguf", port=11499)
    argv = engines.vllm_argv("/models/Qwen3-8B-Q4_K_M.gguf", port=11499,
                             gguf_plugin=True)
    assert argv[2] == "/models/Qwen3-8B-Q4_K_M.gguf"


def test_an_out_of_range_port_is_refused():
    for bad in (0, -1, 70000):
        with pytest.raises(ValueError, match="port"):
            engines.vllm_argv("Qwen/Qwen3-8B", port=bad)


# --- choosing one -----------------------------------------------------------

def test_the_default_is_llamacpp_without_being_asked(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201")
    d = engines.detect_engine_runtime()
    assert d.runtime == engines.LLAMACPP
    assert d.requested == ""


def test_vllm_is_never_chosen_unless_asked(monkeypatch):
    """Even when it would work. llama.cpp stays the default; this is the whole
    'llama.cpp's behaviour must not change' requirement, in one assertion."""
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201")
    for asked in (None, "", "llamacpp", "  "):
        assert engines.detect_engine_runtime(asked).runtime == engines.LLAMACPP


def test_vllm_is_not_chosen_when_asked_but_unavailable(monkeypatch):
    """Falling back is allowed; falling back SILENTLY is not. The decision
    records the request, the reason, and the availability that produced it."""
    _probe(monkeypatch, os_name="windows", python=(3, 12))
    d = engines.detect_engine_runtime("vllm")
    assert d.runtime == engines.LLAMACPP
    assert d.requested == "vllm"
    assert "WSL" in d.reason
    assert d.availability is not None
    assert d.availability.state == "unsupported-os"


def test_vllm_is_chosen_when_asked_and_available(monkeypatch):
    _probe(monkeypatch, os_name="linux", python=(3, 12), executable="/usr/bin/vllm",
           arch="gfx1201", rocm=(7, 0), glibc=(2, 39))
    d = engines.detect_engine_runtime("vLLM")     # case-insensitive, deliberately
    assert d.runtime == engines.VLLM
    assert d.availability is not None and d.availability.available is True


def test_an_unknown_engine_runtime_is_an_error_not_a_fallback(monkeypatch):
    """A typo'd preference must not quietly become llama.cpp — the user would
    be benchmarking the wrong engine and never know."""
    _probe(monkeypatch)
    with pytest.raises(ValueError, match="unknown engine runtime"):
        engines.detect_engine_runtime("tensorrt")


def test_the_registry_lists_both_runtimes_with_a_verdict(monkeypatch):
    monkeypatch.setattr(engines, "llamacpp_availability", lambda registry=None: (
        engines.EngineAvailability(engines.LLAMACPP, True, "runnable", "ok", {})))
    _probe(monkeypatch, os_name="windows", python=(3, 12))
    rows = engines.engine_runtimes()
    assert [r.engine for r in rows] == [engines.LLAMACPP, engines.VLLM]
    assert [r.available for r in rows] == [True, False]
    # every row must survive a JSON round-trip: the UI reads these
    for r in rows:
        d = r.as_dict()
        assert set(d) == {"engine", "available", "state", "reason", "evidence"}


# --- failure classification and the launch loop ----------------------------

def test_the_cuda_wheel_on_an_amd_host_is_named_as_the_cause():
    hint = engines._vllm_failure_hint(
        "ImportError: libcudart.so.12: cannot open shared object file")
    assert "libcudart.so" in hint or "CUDA build" in hint
    assert "3.12" in hint
    assert "WITHOUT saying so" in hint


def test_an_oom_says_rigmas_fit_math_does_not_transfer():
    """A user who has just been OOM'd will reach for the llama.cpp knobs that
    fit their last model. The message has to say those numbers are not this."""
    hint = engines._vllm_failure_hint(
        "ValueError: No available memory for the cache blocks")
    assert "gpu-memory-utilization" in hint
    assert "does not transfer" in hint


def test_an_unrecognised_failure_gets_no_invented_cause():
    assert engines._vllm_failure_hint("some novel traceback") == ""


class _FakeProc:
    def __init__(self, code, pid=4242):
        self._code, self.pid, self.terminated = code, pid, False

    def poll(self):
        return self._code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self._code


def _popen_writing(text, code):
    def popen(argv, stdout=None, stderr=None):
        if stdout is not None:
            stdout.write(text)
        return _FakeProc(code)
    return popen


def test_the_launch_loop_returns_a_handle_when_health_answers(tmp_path, monkeypatch):
    # RIGMA_HOME redirected so the launch cannot create ~/.rigma/logs on the
    # machine running the tests.
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499)
    sp = engines.launch_vllm_server(
        argv, 11499, timeout=5.0, log_path=tmp_path / "vllm.log",
        popen=_popen_writing("", None), is_healthy=lambda: True)
    assert sp.port == 11499
    assert sp.url == "http://127.0.0.1:11499"
    assert sp.proc.pid == 4242


def test_a_crashing_launch_reports_the_log_tail_and_the_real_cause(tmp_path, monkeypatch):
    """The readiness failure has to carry the engine's own words plus the one
    sentence that turns them into a diagnosis — the llama.cpp path learned this
    the hard way (AUDIT F15-3's neighbours)."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    argv = engines.vllm_argv("Qwen/Qwen3-8B", port=11499)
    with pytest.raises(RuntimeError) as e:
        engines.launch_vllm_server(
            argv, 11499, timeout=5.0, log_path=tmp_path / "vllm.log",
            popen=_popen_writing("ImportError: libcudart.so.12\n", 1),
            is_healthy=lambda: False)
    msg = str(e.value)
    assert "libcudart.so" in msg            # the raw tail is preserved
    assert "3.12" in msg                    # and the diagnosis is attached
    assert "0x00000001" in msg              # exit status, as llama.cpp reports it
