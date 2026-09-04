"""What the launch command line and the state record actually carry.

Three defects lived here because nothing looked at either: a terminal chat that
replaced hours of browser writing with a snapshot taken before it, a fit that
charged VRAM for a projector the same call then omitted, and an unload that
rebuilt the state record from defaults and turned that projector back on.

The launch stubs in the rest of the suite all accept `extra_args` and throw it
away, so `--mmproj` could appear or vanish with nothing to notice. Every stub
here records it.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import rigma.cli as cli
from rigma import server_ops, sessions
from rigma import state as st
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, LaunchDefaults, ModelSpec)
from rigma.registry import Registry
from rigma.resolve import draft_cache_mb

runner = CliRunner()


def _profile(vram_mb: int = 16368, ram_free_mb: int = 20000):
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=vram_mb,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=ram_free_mb,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _vision_world(tmp_path, *, launch: LaunchDefaults | None = None,
                  weights_gb: float = 6.0, mmproj_gb: float = 3.0):
    """A vision model with its projector on disk, plus the registry it lives in.

    The projector is deliberately large: 717-888 MiB is what the two shipped
    vision models carry, and the whole point of F18 is that reserving it when
    it will not be loaded moves layers off the card.
    """
    gguf = GgufFile(repo="r", file="v.gguf", bytes=int(weights_gb * 2**30),
                    quant="Q4")
    mm = GgufFile(repo="r", file="v-mmproj.gguf", bytes=int(mmproj_gb * 2**30),
                  quant="F16")
    spec = ModelSpec(slug="seer", family="f", kind="dense", n_layers=40,
                     full_attn_layers=40, kv_heads=8, head_dim=128,
                     native_ctx=131072, ggufs=[gguf], mmproj=mm,
                     use_cases=["general"], capabilities=["vision"],
                     cache_type_policy=CachePolicy(), launch=launch)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "v.gguf").write_text("x")
    (tmp_path / "models" / "v-mmproj.gguf").write_text("x")
    return Registry([], {"seer": spec}, {})


def _record_launch(monkeypatch, tmp_path):
    """Stub the engine launch and KEEP `extra_args`. Every other stub in the
    suite drops it, which is why nothing noticed --mmproj going missing."""
    seen: dict = {}
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    monkeypatch.setattr(server_ops, "_await_port_free", lambda *a, **k: None)
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")

    def fake_launch(exe, plan, mp, port=0, timeout=300.0, extra_args=None):
        seen["extra_args"] = list(extra_args or [])
        seen["plan"] = plan
        return SimpleNamespace(proc=SimpleNamespace(pid=4242))
    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    return seen


def _record_fit(monkeypatch):
    """Capture the ModelSpec each fit is charged against, running the real fit."""
    seen: list = []
    from rigma import resolve as _resolve
    real = _resolve.fit_gguf

    def spy(spec, gguf, profile, ctx, explain):
        seen.append(spec)
        return real(spec, gguf, profile, ctx, explain)
    monkeypatch.setattr(_resolve, "fit_gguf", spy)
    return seen


# --- F18: the fit and the launch must agree about the projector --------------

def test_switch_omits_mmproj_when_the_model_pins_vision_off(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path, launch=LaunchDefaults(vision=False))
    seen = _record_launch(monkeypatch, tmp_path)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    out = server_ops.perform_switch("seer", registry=reg, profile=_profile())
    assert "--mmproj" not in seen["extra_args"]
    assert out["no_vision"] is True


def test_switch_attaches_mmproj_when_vision_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path)
    seen = _record_launch(monkeypatch, tmp_path)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    out = server_ops.perform_switch("seer", registry=reg, profile=_profile())
    assert "--mmproj" in seen["extra_args"]
    assert seen["extra_args"][seen["extra_args"].index("--mmproj") + 1]\
        .endswith("v-mmproj.gguf")
    assert out["no_vision"] is False


def test_the_fit_is_not_charged_for_a_projector_the_launch_omits(tmp_path,
                                                                 monkeypatch):
    """The refusal in the report: `_fit_with_cache` charged spec.mmproj.bytes
    against usable VRAM, then perform_switch resolved vision 30 lines later and
    dropped --mmproj. The fit must see what the launch will actually hold."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path, launch=LaunchDefaults(vision=False))
    _record_launch(monkeypatch, tmp_path)
    fits = _record_fit(monkeypatch)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    server_ops.perform_switch("seer", registry=reg, profile=_profile(),
                              ctx=32768)
    assert fits, "the ctx path must fit before it launches"
    assert fits[-1].mmproj is None, \
        "vision is off — the projector is not resident and must not be reserved"


def test_vision_on_still_reserves_the_projector(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path)
    _record_launch(monkeypatch, tmp_path)
    fits = _record_fit(monkeypatch)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    server_ops.perform_switch("seer", registry=reg, profile=_profile(),
                              ctx=32768)
    assert fits[-1].mmproj is not None
    assert fits[-1].mmproj.bytes == reg.models["seer"].mmproj.bytes


def test_turning_vision_off_buys_gpu_layers(tmp_path, monkeypatch):
    """The user-visible half: 3 GiB of projector is 3 GiB of weights that could
    have stayed on the card. Same model, same context, vision the only change."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    prof = _profile()
    seen_on = _record_launch(monkeypatch, tmp_path)
    # sized so the projector is exactly what pushes it off the card: 11 GiB of
    # weights plus a q4_0 cache at 32K are resident, the same plus 3 GiB of
    # projector are not (measured against this fit, 40 layers -> 37)
    reg_on = _vision_world(tmp_path, weights_gb=11.0)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    server_ops.perform_switch("seer", registry=reg_on, profile=prof, ctx=32768)
    ngl_on = seen_on["plan"].flags.ngl

    seen_off = _record_launch(monkeypatch, tmp_path)
    reg_off = _vision_world(tmp_path, weights_gb=11.0,
                            launch=LaunchDefaults(vision=False))
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    server_ops.perform_switch("seer", registry=reg_off, profile=prof, ctx=32768)
    assert ngl_on < 40, "the projector must be what costs the layers here"
    assert seen_off["plan"].flags.ngl > ngl_on


def test_the_draft_cache_is_budgeted_before_the_fit(tmp_path, monkeypatch):
    """spec_type was applied AFTER the fit, so ~565 MiB of draft cache at 32K
    was budgeted nowhere. The overhead slot must carry it."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path, launch=LaunchDefaults(vision=False,
                                                        spec_type="ngram-simple",
                                                        spec_n_max=1))
    _record_launch(monkeypatch, tmp_path)
    fits = _record_fit(monkeypatch)
    st.write_state("other", "Q0", 18500, engine_pid=999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    server_ops.perform_switch("seer", registry=reg, profile=_profile(),
                              ctx=32768)
    want = draft_cache_mb(reg.models["seer"], 32768, "f16", "ngram-simple", 1)
    assert want > 0
    assert fits[-1].mmproj is not None, "the draft cache needs an overhead slot"
    assert fits[-1].mmproj.bytes == int(want * 2**20)


# --- F18 on the `rigma up` path ----------------------------------------------

def _up_world(tmp_path, monkeypatch, reg):
    """Everything `up` touches, stubbed, with the launch line recorded."""
    seen: dict = {}
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls: reg))
    monkeypatch.setattr(cli, "probe_hardware", lambda gpus, raw_gpus=None:
                        _profile())
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    monkeypatch.setattr("rigma.runtime.ensure_model",
                        lambda g: tmp_path / "models" / g.file)
    monkeypatch.setattr("rigma.bench.is_calibrated", lambda *a, **k: True)

    def fake_launch(exe, plan, mp, port=0, timeout=300.0, extra_args=None):
        seen["extra_args"] = list(extra_args or [])
        seen["plan"] = plan
        return SimpleNamespace(proc=SimpleNamespace(pid=4242))
    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    # run_ui blocks forever in production; here it is the one moment the state
    # record exists, since `up` clears it on the way out.
    monkeypatch.setattr("rigma.serve.run_ui",
                        lambda port, eport: seen.update(state=st.read_state()))
    return seen


def test_up_does_not_attach_a_projector_the_plan_left_out(tmp_path, monkeypatch):
    """cli.up re-fit WITHOUT the projector and then attached --mmproj anyway —
    and `ensure_model` downloads it. Silent overcommit, paged on Windows."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path, launch=LaunchDefaults(vision=False))
    seen = _up_world(tmp_path, monkeypatch, reg)
    res = runner.invoke(cli.app, ["up", "--model", "seer", "--yes",
                                  "--no-browser", "--no-calibrate"])
    assert res.exit_code == 0, res.output
    assert "--mmproj" not in seen["extra_args"]
    assert seen["state"]["no_vision"] is True


def test_up_attaches_the_projector_when_vision_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path)
    seen = _up_world(tmp_path, monkeypatch, reg)
    res = runner.invoke(cli.app, ["up", "--model", "seer", "--yes",
                                  "--no-browser", "--no-calibrate"])
    assert res.exit_code == 0, res.output
    assert "--mmproj" in seen["extra_args"]
    assert seen["state"]["no_vision"] is False


# --- F22: unloading must not rebuild the record from defaults ----------------

def _loaded_state(tmp_path):
    st.write_state("seer", "Q4", 18500, engine_pid=999999, ui_pid=os.getpid(),
                   backend="vulkan", use_case="general", ctx=32768,
                   kv_cache="q8_0", no_vision=True, gguf="v.gguf",
                   kv_fp="cafef00d")


def test_unload_keeps_the_vram_the_user_freed(tmp_path, monkeypatch):
    """no_vision and kv_cache were omitted from perform_unload's write_state,
    so unloading reverted them to False/"" and the next load re-attached the
    projector."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _loaded_state(tmp_path)
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    # kvcache.save talks to the engine over HTTP; there is no engine here.
    monkeypatch.setattr("rigma.kvcache.save", lambda *a, **k: None)
    monkeypatch.setattr("rigma.kvcache.prune", lambda *a, **k: None)
    out = server_ops.perform_unload()
    assert out["unloaded"] is True and out["engine_pid"] == -1
    assert out["no_vision"] is True, "unloading must not switch vision back on"
    assert out["kv_cache"] == "q8_0"
    assert out["kv_fp"] == "cafef00d" and out["gguf"] == "v.gguf"
    assert out["ctx"] == 32768 and out["model"] == "seer"


def test_a_failed_launch_keeps_the_prompt_cache_and_vision_choice(tmp_path,
                                                                  monkeypatch):
    """The launch-failure path dropped kv_fp too, orphaning the saved cache."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _vision_world(tmp_path, launch=LaunchDefaults(vision=False))
    _record_launch(monkeypatch, tmp_path)
    _loaded_state(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("launch failed: boom")
    monkeypatch.setattr("rigma.runtime.launch_server", boom)
    with pytest.raises(RuntimeError, match="boom"):
        server_ops.perform_switch("seer", registry=reg, profile=_profile(),
                                  ctx=16384)
    s = st.read_state()
    assert s["unloaded"] is True and s["engine_pid"] == -1
    assert s["kv_fp"] == "cafef00d", "the saved prompt cache must not be orphaned"
    assert s["no_vision"] is True and s["kv_cache"] == "q8_0"


def test_update_state_changes_only_what_it_is_handed(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _loaded_state(tmp_path)
    out = st.update_state(ctx=8192)
    assert out["ctx"] == 8192
    for k in ("model", "quant", "public_port", "backend", "use_case",
              "kv_cache", "no_vision", "gguf", "kv_fp"):
        assert out[k] == st.read_state()[k]
    assert out["no_vision"] is True and out["kv_cache"] == "q8_0"


# --- F2: the terminal chat must not replace an evening of browser writing -----

def _running(tmp_path):
    st.write_state("seer", "Q4", 18500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)


def _typed(monkeypatch, lines, before=None):
    """Feed the REPL fixed lines. `before` runs while the user is still typing —
    that is the window the browser writes in."""
    it = iter(lines)
    fired = []

    def fake_prompt(text, **kw):
        nxt = next(it)
        if before is not None and not fired:
            fired.append(True)      # the browser writes once, not every turn
            before()
        return nxt
    monkeypatch.setattr("typer.prompt", fake_prompt)


def _browser_writes(sid, text, title=None):
    """A second writer landing a message the way serve.py does."""
    s = sessions.load(sid)
    s["messages"].append({"role": "user", "content": text})
    s["messages"].append({"role": "assistant", "content": text + " reply"})
    if title is not None:
        s["title"] = title
    sessions.save(s, base_rev=s[sessions.REV_KEY])


def test_a_terminal_turn_keeps_what_the_browser_wrote_first(tmp_path,
                                                            monkeypatch):
    """Session loaded once before the REPL, browser writes for an evening, one
    terminal line replaced the lot."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _running(tmp_path)
    sess = sessions.create()
    sid = sess["id"]
    _typed(monkeypatch, ["hello from the terminal", "exit"],
           before=lambda: _browser_writes(sid, "chapter four",
                                          title="The Long Evening"))
    monkeypatch.setattr(cli, "_stream_chat",
                        lambda port, history, params=None: "terminal reply")
    res = runner.invoke(cli.app, ["chat", "--session", sid])
    assert res.exit_code == 0, res.output
    texts = [m["content"] for m in sessions.load(sid)["messages"]]
    assert texts == ["chapter four", "chapter four reply",
                     "hello from the terminal", "terminal reply"]
    assert sessions.load(sid)["title"] == "The Long Evening"


def test_a_write_during_the_generation_survives_too(tmp_path, monkeypatch):
    """The other half of the window: the browser lands a turn while the model
    is still streaming. reload_and_extend grafts this turn onto what is stored
    NOW rather than onto an hours-old list."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _running(tmp_path)
    sid = sessions.create()["id"]
    _typed(monkeypatch, ["terminal question", "exit"])

    def slow_reply(port, history, params=None):
        _browser_writes(sid, "landed mid-generation")
        return "terminal reply"
    monkeypatch.setattr(cli, "_stream_chat", slow_reply)
    res = runner.invoke(cli.app, ["chat", "--session", sid])
    assert res.exit_code == 0, res.output
    texts = [m["content"] for m in sessions.load(sid)["messages"]]
    assert texts == ["landed mid-generation", "landed mid-generation reply",
                     "terminal question", "terminal reply"]


def test_an_unreachable_engine_writes_nothing_at_all(tmp_path, monkeypatch):
    """The save happened BEFORE the model call, so a dead engine still landed
    the stale snapshot — twice, counting the error handler's second save."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _running(tmp_path)
    sid = sessions.create()["id"]
    _browser_writes(sid, "the evening's work", title="Kept")
    before = sessions.load(sid)
    _typed(monkeypatch, ["are you there", "exit"])

    def dead(port, history, params=None):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(cli, "_stream_chat", dead)
    res = runner.invoke(cli.app, ["chat", "--session", sid])
    assert res.exit_code == 0
    after = sessions.load(sid)
    assert [m["content"] for m in after["messages"]] == \
        [m["content"] for m in before["messages"]]
    assert after["title"] == "Kept"
    assert after[sessions.REV_KEY] == before[sessions.REV_KEY], \
        "a failed turn must not write at all"


def test_a_session_deleted_mid_chat_is_not_resurrected(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _running(tmp_path)
    sid = sessions.create()["id"]
    _typed(monkeypatch, ["still here?", "exit"], before=lambda:
           sessions.delete(sid))
    monkeypatch.setattr(cli, "_stream_chat",
                        lambda port, history, params=None: "reply")
    res = runner.invoke(cli.app, ["chat", "--session", sid])
    assert res.exit_code == 0, res.output
    assert sessions.load(sid) is None
