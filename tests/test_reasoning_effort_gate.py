"""C3: `--reasoning-effort` is FORK-ONLY and must never reach the pinned engine.

The fork defines `--reasoning-effort LEVEL` (PrismML-Eng/llama.cpp 87268f77
`common/arg.cpp:3678`); mainline `ggml-org/llama.cpp` b9867 — the PINNED build in
`src/rigma/data/engines.json` — has no such string anywhere in its `common/arg.cpp`
(verified by grep at the pin). `llama-server` exits in argparse on an unknown
argument (`error: invalid argument: --reasoning-effort`), so emitting it on the pin
breaks every launch, not just the ones that asked for the lever.

THE SAFETY PROPERTY: the answer used to build the argv is the identity of the
engine that will actually launch. That answer is frozen onto the plan at the
moment the binary is chosen (`server_ops.engine_binary_for_plan`), and
`RunPlan.server_args` reads only the frozen value — it never re-asks. Re-asking is
the bug these tests exist for: `_registered_engine_for` needs the model on disk,
and on a first run `ensure_model` puts it there AFTER the binary was chosen, so a
re-ask names the fork for a model that was absent when the pinned binary was
picked. The launch then handed `--reasoning-effort` to mainline and died.

These tests assert on the built argv and never launch an engine.
"""
from __future__ import annotations

import subprocess

import pytest
from pydantic import ValidationError

from rigma import engine_compat, engine_registry, runtime, server_ops
from rigma.models import ComboFlags, GgufFile, RunPlan

# Ternary-Bonsai-2-27B-PQ2_0: needs the fork's private type 142.
BONSAI = {0: 1, 30: 1, 142: 100}
# An ordinary mainline histogram.
SMOL = {0: 1, 8: 12, 11: 40, 20: 5}


class _Idx:
    def __init__(self, counts, complete=True):
        self.type_counts = counts
        self.types_complete = complete


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A RIGMA_HOME whose models/ dir starts EMPTY — a genuine first run."""
    h = tmp_path / "rigma"
    (h / "models").mkdir(parents=True)
    monkeypatch.setattr(server_ops, "rigma_home", lambda: h)
    return h


def _pinned(monkeypatch, tmp_path):
    """The pin fallback must not download: stub `ensure_engine`."""
    pinned = tmp_path / "pinned-llama-server.exe"
    pinned.write_bytes(b"fake")
    monkeypatch.setattr(runtime, "ensure_engine", lambda backend, os_name: pinned)
    return pinned


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="model.gguf", bytes=1,
                                 quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096, **fl),
                   origin="calculator")


def _register(monkeypatch, tmp_path, *, name="prism", types=range(144),
              known=True, backend="vulkan"):
    monkeypatch.setattr(engine_registry, "registry_path",
                        lambda home=None: tmp_path / "engines-custom.json")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"fake")
    engine_registry.register(engine_registry.CustomEngine(
        name=name, path=str(exe), backend=backend,
        source="PrismML-Eng/llama.cpp branch 'prism'",
        types=list(types), types_known=known))


def _types(monkeypatch, counts, complete=True):
    import rigma.gguf_meta as gm
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(counts, complete))


def _on_disk(home):
    (home / "models" / "model.gguf").write_bytes(b"not a real gguf")


def test_effort_emitted_for_the_registered_prism_fork(home, monkeypatch, tmp_path):
    """The launching engine really is the fork, so the flag is emitted."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["kind"] == "registered" and rec["is_prism_fork"] is True
    assert plan.engine_is_prism_fork is True
    args = plan.server_args("/m", 11500)
    assert args[args.index("--reasoning-effort") + 1] == "high"


def test_effort_never_emitted_on_the_pinned_engine(home, monkeypatch, tmp_path):
    """No registered fork -> the pin is mainline b9867 -> the flag must be absent,
    even though the plan asked for a level and even for a fork-only model."""
    _pinned(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["kind"] == "pinned"
    assert "--reasoning-effort" not in plan.server_args("/m", 11500)


def test_first_run_pin_does_not_get_the_fork_flag_when_the_file_appears_later(
        home, monkeypatch, tmp_path):
    """THE VERIFIER'S REPRODUCTION, as a regression test.

    model absent -> engine chosen (the pin, because `select` cannot read a header
    that is not there) -> `ensure_model` makes the file appear -> argv built. The
    argv must still describe the PIN, not the fork the file would now select. The
    middle assertion proves the scenario is real: a re-ask after the download
    answers "fork", which is exactly the stale answer the old code used.
    """
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)          # the fork IS registered
    _types(monkeypatch, BONSAI)
    plan = _plan(reasoning_effort="high")

    # 1. first run: the model is not on disk, so the pinned mainline build is chosen
    assert not (home / "models" / "model.gguf").exists()
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["kind"] == "pinned"
    assert plan.engine_is_prism_fork is False

    # 2. ensure_model downloads it...
    _on_disk(home)
    # ...and a re-ask NOW would name the fork — the stale answer.
    assert engine_compat.engine_is_prism_fork(
        server_ops._registered_engine_for(plan.gguf, plan.backend)) is True

    # 3. the argv must follow the engine that will actually launch (the pin).
    args = plan.server_args("/m", 11500)
    assert "--reasoning-effort" not in args


def test_first_run_after_the_download_the_fork_flag_is_emitted(
        home, monkeypatch, tmp_path):
    """The other half of the same scenario: once the model IS on disk, the fork is
    chosen and the flag belongs on the argv."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["kind"] == "registered"
    assert "--reasoning-effort" in plan.server_args("/m", 11500)


def test_effort_absent_by_default_even_on_the_fork(home, monkeypatch, tmp_path):
    """No level asked for -> no flag -> the engine's own template default."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan()
    server_ops.engine_binary_for_plan(plan, "windows")
    assert "--reasoning-effort" not in plan.server_args("/m", 11500)


def test_default_means_keep_the_template_default(home, monkeypatch, tmp_path):
    """`default` is the engine's own sentinel for 'erase the kwarg'; Rigma must
    translate it to omission rather than passing the literal through."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="default")
    server_ops.engine_binary_for_plan(plan, "windows")
    assert "--reasoning-effort" not in plan.server_args("/m", 11500)


def test_a_registered_non_fork_engine_does_not_get_the_flag(home, monkeypatch,
                                                            tmp_path):
    """A registered MAINLINE-mirroring build is selected for this model (it
    accepts every type), but it declares no PrismML-private type, so it is not
    the fork and must not be handed a fork-only flag."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path, name="mainline-mirror", types=range(43))
    _types(monkeypatch, SMOL)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["is_prism_fork"] is False
    assert "--reasoning-effort" not in plan.server_args("/m", 11500)


def test_an_engine_with_unknown_capabilities_is_not_taken_for_the_fork(
        home, monkeypatch, tmp_path):
    """Unknown is not a licence to emit: a build nobody vouched for might be
    mainline, where the flag is fatal. `select` will not even choose it, so the
    pin runs and the flag stays off."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path, name="mystery", types=[], known=False)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    exe, rec = server_ops.engine_binary_for_plan(plan, "windows")
    assert rec["kind"] == "pinned" and rec["is_prism_fork"] is False
    assert "--reasoning-effort" not in plan.server_args("/m", 11500)


def test_a_plan_that_never_chose_an_engine_omits_the_flag(home, monkeypatch):
    """No freeze means no engine has been chosen yet — a `--dry-run` preview, or
    any caller that skipped the seam. That is not a licence to emit: the flag is
    fork-only and fatal on mainline, so an unanswered question means 'off'."""
    _types(monkeypatch, BONSAI)
    args = _plan(reasoning_effort="high").server_args("/m", 11500)
    assert "--reasoning-effort" not in args


def test_the_frozen_identity_survives_a_calibration_copy(home, monkeypatch,
                                                         tmp_path):
    """`auto_calibrate` and `run_sweep` rebuild the plan with `model_copy`, which
    must carry the frozen identity — otherwise the sweep's argv would lose the
    flag the launch will have."""
    _pinned(monkeypatch, tmp_path)
    _register(monkeypatch, tmp_path)
    _types(monkeypatch, BONSAI)
    _on_disk(home)
    plan = _plan(reasoning_effort="high")
    server_ops.engine_binary_for_plan(plan, "windows")
    trial = plan.model_copy(update={
        "flags": plan.flags.model_copy(update={"ctx": 8192})})
    assert trial.engine_is_prism_fork is True
    assert "--reasoning-effort" in trial.server_args("/m", 11500)


def test_unknown_level_rejected():
    with pytest.raises(ValidationError):
        ComboFlags(ctx=4096, reasoning_effort="ultra")


# --- the version-blind false positive must not be fatal ----------------------
#
# `engine_is_prism_fork` reads the build's declared types. A PrismML build older
# than 87268f77 declares the same private types but predates the flag, and
# registration carries no version, so the gate cannot tell. `launch_server` must
# survive that: the engine names the argument it rejected, and Rigma relaunches
# ONCE without it. No engine is launched here — `subprocess.Popen` is faked and
# the process dies immediately, exactly as argparse would.

class _Proc:
    pid = 4242

    def __init__(self, dead: bool):
        self._dead = dead

    def poll(self):
        return 1 if self._dead else None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_a_version_blind_fork_false_positive_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma"))
    argv_seen = []

    def fake_popen(argv, **kw):
        argv_seen.append(list(argv))
        if len(argv_seen) == 1:
            # argparse rejecting the option the build predates
            kw["stdout"].write(
                "error: invalid argument: --reasoning-effort\n")
            kw["stdout"].flush()
            return _Proc(dead=True)
        return _Proc(dead=False)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime.ServerProcess, "is_healthy", lambda self: True)

    plan = _plan(reasoning_effort="high")
    plan.engine_is_prism_fork = True     # the gate's (wrong) answer for that build
    sp = runtime.launch_server(tmp_path / "srv.exe", plan,
                               tmp_path / "m.gguf", 11601, timeout=0.2)

    assert sp is not None
    assert len(argv_seen) == 2, "must retry exactly once"
    assert "--reasoning-effort" in argv_seen[0]
    assert "--reasoning-effort" not in argv_seen[1]


def test_a_genuine_engine_failure_is_still_fatal(tmp_path, monkeypatch):
    """The retry must not swallow an unrelated death: with no argparse rejection
    naming the flag, `launch_server` raises as before."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma"))
    argv_seen = []

    def fake_popen(argv, **kw):
        argv_seen.append(list(argv))
        kw["stdout"].write("error: failed to load model\n")
        kw["stdout"].flush()
        return _Proc(dead=True)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    plan = _plan(reasoning_effort="high")
    plan.engine_is_prism_fork = True
    with pytest.raises(RuntimeError):
        runtime.launch_server(tmp_path / "srv.exe", plan,
                              tmp_path / "m.gguf", 11601, timeout=0.2)
    assert len(argv_seen) == 1
