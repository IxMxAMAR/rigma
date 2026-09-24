"""A model remembers how it should be launched.

Owner, 2026-08-19: "I have to change context then KV and model reloads 2 times
to get me to my idle config." Applying them together was fixed; REMEMBERING them
was not, so every plain load still landed on whatever the resolver picked and
had to be corrected by hand.

Measured on this machine 2026-08-21, the difference between the resolver's
choice and the right one is large: the same model and quant ran at 9.95 tok/s in
one configuration and 38.27 in another. That is not a preference, it is the
whole performance of the machine, and it should not have to be re-entered.

Precedence, weakest last: explicit request > model default > resolver.
"""
import pytest

from rigma.models import GgufFile, LaunchDefaults, ModelSpec


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "home"


def _spec(**over):
    base = dict(slug="m", family="qwen35", kind="dense", n_layers=64,
                full_attn_layers=16, kv_heads=4, head_dim=256,
                native_ctx=262144, custom=True,
                ggufs=[GgufFile(repo="a/b", file="q.gguf", bytes=100,
                                quant="Q3_K_M")])
    base.update(over)
    return ModelSpec(**base)


def test_a_spec_with_no_defaults_changes_nothing():
    assert _spec().launch is None


def test_defaults_round_trip_through_json():
    """They live in the spec file, so they must survive being written and read
    like every other stored field."""
    s = _spec(launch=LaunchDefaults(ctx=65536, kv="q5_1", quant="Q3_K_M",
                                    vision=False, spec_type="draft-mtp",
                                    spec_n_max=1))
    back = ModelSpec.model_validate_json(s.model_dump_json())
    assert back.launch is not None
    assert back.launch.ctx == 65536
    assert back.launch.kv == "q5_1"
    assert back.launch.vision is False
    assert back.launch.spec_type == "draft-mtp"
    assert back.launch.spec_n_max == 1


@pytest.mark.parametrize("field,value", [
    ("ctx", 0), ("kv", ""), ("quant", ""), ("spec_type", ""),
])
def test_unset_fields_mean_no_opinion(field, value):
    """Zero and empty are 'let the resolver decide', not 'force this'. A model
    that only wants to pin its context must not thereby also pin its cache."""
    d = LaunchDefaults(**{field: value})
    assert getattr(d, field) == value
    assert d.is_set(field) is False


def test_a_set_field_is_reported_as_set():
    d = LaunchDefaults(ctx=65536, kv="q5_1")
    assert d.is_set("ctx") and d.is_set("kv")
    assert not d.is_set("quant")


def test_vision_off_is_an_opinion_but_unset_is_not():
    """vision is tri-state: None means "keep whatever the last launch used",
    which is not the same as False. Conflating them would silently reload a
    600MB projector the user turned off."""
    assert LaunchDefaults().is_set("vision") is False
    assert LaunchDefaults(vision=False).is_set("vision") is True
    assert LaunchDefaults(vision=True).is_set("vision") is True


def test_perform_switch_applies_the_stored_defaults(home, monkeypatch):
    """A plain load — no ctx, no kv, no quant — must land on the model's own
    configuration rather than the resolver's guess.

    AUDIT F60: the body used to build a LaunchDefaults, assert on a local dict
    it had just written itself, and never call perform_switch at all — so the
    one thing the test is named for was the one thing it did not check.
    """
    import os
    from types import SimpleNamespace

    from rigma import server_ops, state
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    from rigma.registry import Registry

    spec = _spec(launch=LaunchDefaults(ctx=8192, kv="q5_1", quant="Q3_K_M",
                                       vision=False))
    reg = Registry([], {"m": spec}, {})
    gpu = GpuInfo(vendor="amd", name="X", vram_mb=24576, backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=24000,
                              cpu=CpuInfo(cores=16), os="windows",
                              disk_free_gb=200.0)
    (home / "models").mkdir(parents=True, exist_ok=True)
    (home / "models" / "q.gguf").write_text("x", encoding="utf-8")
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    monkeypatch.setattr("rigma.state.kill_pid", lambda pid: None)
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: home / "llama-server.exe")
    seen = {}

    def _launch(exe, plan, mp, port=0, timeout=300.0, extra_args=None):
        seen["plan"] = plan
        return SimpleNamespace(proc=SimpleNamespace(pid=4242))

    monkeypatch.setattr("rigma.runtime.launch_server", _launch)
    new = server_ops.perform_switch("m", registry=reg, profile=profile)
    assert seen["plan"].flags.ctx == 8192, seen["plan"].flags
    assert seen["plan"].flags.cache_type_k == "q5_1", seen["plan"].flags
    assert seen["plan"].gguf.quant == "Q3_K_M", seen["plan"].gguf
    assert new["ctx"] == 8192 and new["quant"] == "Q3_K_M", new


def test_an_explicit_request_beats_the_stored_default():
    """Precedence: what the user just asked for wins. Otherwise changing
    context once would be impossible on a model that pins it."""
    from rigma.models import LaunchDefaults
    d = LaunchDefaults(ctx=65536, kv="q5_1")
    request = {"ctx": 16384}
    merged = {**d.as_overrides(), **{k: v for k, v in request.items()
                                     if v is not None}}
    assert merged["ctx"] == 16384      # explicit wins
    assert merged["kv"] == "q5_1"      # unspecified falls back to the default
