"""A launch must record the cache fingerprint it launched under.

The bug this pins: `rigma up` wrote state without `kv_fp`, and `write_state`
reverts unnamed fields to their defaults BY DESIGN, so it wrote an empty string.
Nothing failed. `serve._prefix_ctx` returns None on a falsy fingerprint, so
prefix warm/snapshot became silent no-ops and `perform_unload` skipped its KV
save — both caches were off for every CLI-started run, and the only symptom was
that long conversations were mysteriously slow to start.

An empty fingerprint is the SAFE default (fail closed: no cache is restored
rather than the wrong one), which is exactly why the mistake is invisible. So
the invariant is checked structurally: any `write_state` call that records a LIVE
engine must name `kv_fp`. `engine_pid=-1` means "no engine" and is exempt.

Parsed with `ast` rather than matched with a regex, because the argument lists
are multi-line and keyword-based, and a regex that only sometimes sees the whole
call is worse than no guard.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import rigma.cli as cli
from rigma import state as st
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, ModelSpec)
from rigma.registry import Registry

SRC = Path(__file__).resolve().parents[1] / "src" / "rigma"


def _write_state_calls():
    """Every `write_state(...)` call in the package, as (file, node)."""
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name == "write_state":
                yield path, node


def _kw(node: ast.Call, key: str):
    for k in node.keywords:
        if k.arg == key:
            return k.value
    return None


def test_every_call_that_records_a_live_engine_names_its_cache_fingerprint():
    offenders = []
    for path, node in _write_state_calls():
        pid = _kw(node, "engine_pid")
        if pid is None:
            continue
        # engine_pid=-1 is the explicit "no engine" record (the UI-up-without-a-
        # model path); there is no cache to name and blank is correct.
        if isinstance(pid, ast.UnaryOp) and isinstance(pid.operand, ast.Constant):
            if pid.operand.value == 1:
                continue
        if _kw(node, "kv_fp") is None:
            offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these write_state calls record a running engine but no kv_fp, which "
        "silently disables prefix caching AND restore-on-unload: "
        + ", ".join(offenders))


def test_both_launch_paths_agree_by_construction():
    """`rigma up` and the UI reload must derive the fingerprint the SAME way.

    Two copies of the expression is how they drift, and a drifted fingerprint
    does not error — it restores a cache taken under a different configuration,
    which generates fluent text from a history that never happened.
    """
    for name in ("cli.py", "server_ops.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert "launch_fingerprint(" in src, (
            f"{name} launches an engine but does not use the shared "
            "kvcache.launch_fingerprint")
        assert "kvcache.fingerprint(kvcache.config_of(" not in src, (
            f"{name} derives the fingerprint inline again — use the shared "
            "helper so the two launch paths cannot disagree")


def test_the_guard_would_have_caught_the_original_bug():
    """The guard is only worth having if it fails on the code that caused it.

    Reproduces the pre-fix call verbatim, minus `kv_fp`.
    """
    bug = ast.parse(
        "st.write_state(rp.model_slug, rp.gguf.quant, port,\n"
        "               engine_pid=sp.proc.pid, ui_pid=os.getpid(),\n"
        "               backend=rp.backend, ctx=rp.flags.ctx)\n")
    call = next(n for n in ast.walk(bug) if isinstance(n, ast.Call))
    assert _kw(call, "kv_fp") is None
    assert _kw(call, "engine_pid") is not None


# --- A8b: recording the key is only half — the launch must OFFER the slot -----
#
# A8 made a refused restore loud on the UI's switch path. The CLI's `up` path
# wrote the same fingerprint and never called `kvcache.restore` at all, so a
# cache saved by a previous launch was never offered to the engine: every
# `rigma up` re-prefilled from zero no matter what sat on disk. These tests run
# the real `up` command against a fake world (no engine is started) and assert
# the slot is offered for a matching fingerprint, ignored for a mismatched one,
# and that a refusal is printed while the key is still recorded.

_runner = CliRunner()
_FP = "deadbeefdeadbeef"


def _profile():
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _up_world(tmp_path, monkeypatch, fingerprint):
    """Everything `up` touches, stubbed. `fingerprint` is what the launch will
    ask the cache for; `seen["calls"]` records the slot actions the engine was
    asked to perform (it is never actually started)."""
    gguf = GgufFile(repo="r", file="m.gguf", bytes=6 * 2**30, quant="Q4")
    spec = ModelSpec(slug="m", family="f", kind="dense", n_layers=40,
                     full_attn_layers=40, kv_heads=8, head_dim=128,
                     native_ctx=131072, ggufs=[gguf], use_cases=["general"],
                     cache_type_policy=CachePolicy())
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "m.gguf").write_text("x")
    reg = Registry([], {"m": spec}, {})
    seen: dict = {"calls": [], "refuse": None}
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls: reg))
    monkeypatch.setattr(cli, "probe_hardware",
                        lambda gpus, raw_gpus=None: _profile())
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    monkeypatch.setattr("rigma.runtime.ensure_model",
                        lambda g: tmp_path / "models" / g.file)
    monkeypatch.setattr("rigma.bench.is_calibrated", lambda *a, **k: True)
    # The fake engine pid must never be signalled on the way out.
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)

    def fake_launch(exe, plan, mp, port=0, timeout=300.0, extra_args=None):
        seen["plan"] = plan
        return SimpleNamespace(proc=SimpleNamespace(pid=4242))
    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    monkeypatch.setattr("rigma.serve.run_ui",
                        lambda port, eport: seen.update(state=st.read_state()))
    # Pure function of the plan; pin it so the test can plant the blob whose
    # name the launch will ask for.
    monkeypatch.setattr("rigma.kvcache.launch_fingerprint",
                        lambda rp, exe: fingerprint)

    def fake_slot_action(port, slot, action, filename, timeout=120.0):
        seen["calls"].append(action)
        return seen["refuse"]
    monkeypatch.setattr("rigma.kvcache.slot_action", fake_slot_action)
    return seen


def _run_up():
    return _runner.invoke(cli.app, ["up", "--model", "m", "--yes",
                                    "--no-browser", "--no-calibrate"])


def test_up_offers_the_saved_cache_for_a_matching_fingerprint(tmp_path,
                                                              monkeypatch):
    """The defect: `rigma up` wrote `kv_fp` but never called `restore`, so a
    cache saved under exactly this configuration was never offered. Before the
    fix `seen["calls"]` is empty and the engine is left to re-prefill."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"kv-{_FP}.bin").write_bytes(b"a saved cache")
    res = _run_up()
    assert res.exit_code == 0, res.output
    assert seen["calls"] == ["restore"], (
        "a matching fingerprint must offer the saved slot to the engine")
    assert seen["state"]["kv_fp"] == _FP


def test_up_does_not_offer_a_cache_saved_under_another_fingerprint(
        tmp_path, monkeypatch):
    """A blob exists, but under a different key. Offering it would restore a
    history taken under another configuration — the silent-corruption case
    `kvcache` exists to prevent — so the engine must never be asked."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "kv-0000000000000000.bin").write_bytes(b"another config")
    res = _run_up()
    assert res.exit_code == 0, res.output
    assert seen["calls"] == [], "a mismatched fingerprint must not restore"
    assert "re-prefilled from zero" not in res.output
    assert seen["state"]["kv_fp"] == _FP


def test_a_refused_restore_is_reported_and_still_keys_the_unload_save(
        tmp_path, monkeypatch):
    """`restore` RETURNS `(False, reason)` on the ordinary refusal; it does not
    raise. The reason must reach the user, and `kv_fp` must still be recorded:
    it is the key the unload save writes under, and that save is the only thing
    that overwrites the blob the engine could not read (A8's rule)."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"kv-{_FP}.bin").write_bytes(b"a saved cache")
    seen["refuse"] = "engine refused the restore (500)"
    res = _run_up()
    assert res.exit_code == 0, res.output
    assert seen["calls"] == ["restore"]
    assert "engine refused the restore (500)" in res.output, res.output
    assert "re-prefilled from zero" in res.output, res.output
    # A8b-n2: the note is deliberately NOT appended to `rp.explain`. The explain
    # loop at the top of `up` has already run by the time the engine is up,
    # nothing after this reads `rp.explain`, and `write_state` does not persist
    # it — so the old append was dead code that only looked like a report. The
    # user-facing half is the echo; the diagnostic half is the WARNING (below).
    assert not any("re-prefilled from zero" in w for w in seen["plan"].explain)
    assert seen["state"]["kv_fp"] == _FP, (
        "clearing kv_fp would skip the unload save, leave the unreadable file "
        "in place forever, and re-prefill on every later restart")


def test_a_refused_restore_is_logged_at_warning_with_the_reason(
        tmp_path, monkeypatch, caplog):
    """A8b-n1. A8's stated motivation was "no trace to diagnose"; the server
    path logs a refusal at WARNING (server_ops.perform_switch) and the CLI path
    did not, so the launch a CLI user actually runs was the one refusal with no
    trace anywhere. The log line must carry the engine's reason."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"kv-{_FP}.bin").write_bytes(b"a saved cache")
    seen["refuse"] = "engine refused the restore (500)"
    with caplog.at_level(logging.WARNING, logger="rigma.cli"):
        res = _run_up()
    assert res.exit_code == 0, res.output
    hits = [r for r in caplog.records
            if r.levelno == logging.WARNING
            and "kv-cache restore failed" in r.getMessage()]
    assert hits, caplog.text
    assert "engine refused the restore (500)" in hits[0].getMessage(), caplog.text


def test_a_refused_restore_prints_its_note_exactly_once(tmp_path, monkeypatch):
    """The note is the user-facing half. It must be visible and appear exactly
    once: a fix that handed the note to the logger as well, or that echoed it
    after the explain loop had already printed it, would put the same paragraph
    on the terminal — and in the detached log — twice."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"kv-{_FP}.bin").write_bytes(b"a saved cache")
    seen["refuse"] = "engine refused the restore (500)"
    res = _run_up()
    assert res.exit_code == 0, res.output
    emitted = res.output + res.stderr
    assert emitted.count("would not load") == 1, emitted


def test_a_successful_restore_logs_no_warning_and_prints_no_note(
        tmp_path, monkeypatch, caplog):
    """Negative control. A slot that loads is the ordinary case; warning about
    it would put a four-minute notice on every warm launch, and echoing a note
    would tell the user a cache failed to load when it did not."""
    seen = _up_world(tmp_path, monkeypatch, _FP)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"kv-{_FP}.bin").write_bytes(b"a saved cache")
    seen["refuse"] = None
    with caplog.at_level(logging.WARNING, logger="rigma.cli"):
        res = _run_up()
    assert res.exit_code == 0, res.output
    assert seen["calls"] == ["restore"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
        caplog.text
    assert "would not load" not in res.output
    assert "re-prefilled from zero" not in res.output
    assert seen["state"]["kv_fp"] == _FP
