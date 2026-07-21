"""Phase 2 of the field-parity audit: use what the engine and model offer.

MTP detection from real gguf metadata, draft-mtp calibration trials, family
default inheritance for custom imports, 2-slot cache hygiene, reasoning
budget plumbing, and reasoning carry-over for runs.
"""
import pytest

from rigma import bench, sessions
from rigma.models import CachePolicy, ComboFlags, GgufFile, ModelSpec, RunPlan


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


# --- MTP capability drives the spec trial -------------------------------------
def test_quick_configs_add_mtp_trials_only_with_capability():
    base = ComboFlags(ctx=8192, n_cpu_moe=4)
    labels = [k for k, _ in bench.quick_configs(base, moe=True, caps=("mtp",))]
    assert "spec-mtp-2" in labels and "spec-mtp-4" in labels
    labels_no = [k for k, _ in bench.quick_configs(base, moe=True)]
    assert not any("spec" in x for x in labels_no)


def test_mtp_trial_flags_are_launchable():
    cfgs = dict(bench.quick_configs(ComboFlags(ctx=8192), moe=True,
                                    caps=["mtp"]))
    flags = ComboFlags(ctx=8192).model_copy(update=cfgs["spec-mtp-2"])
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=flags, origin="test")
    args = plan.server_args("m.gguf", 11500)
    i = args.index("--spec-type")
    assert args[i + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-n-max") + 1] == "2"


def test_gguf_meta_detects_nextn_layers(monkeypatch):
    # the real key observed live: qwen35moe.nextn_predict_layers = 1 on the
    # MTP-preserved gguf, absent on the plain quant
    from rigma import gguf_meta

    def fake_meta(src):
        return {"general.architecture": "qwen35moe",
                "general.name": "test-mtp",
                "qwen35moe.block_count": 4,
                "qwen35moe.attention.head_count": 8,
                "qwen35moe.attention.head_count_kv": 2,
                "qwen35moe.attention.key_length": 64,
                "qwen35moe.context_length": 4096,
                "qwen35moe.expert_count": 8,
                "qwen35moe.nextn_predict_layers": 1,
                "tokenizer.chat_template": "tool <think>"}

    monkeypatch.setattr(gguf_meta, "read_metadata", fake_meta)
    info = gguf_meta.inspect_gguf("x.gguf")
    assert "mtp" in info.capabilities
    assert "tools" in info.capabilities     # existing detection untouched


# --- custom imports inherit the registry sibling's defaults -------------------
def _spec(slug, custom, family, dp=None, ctp=None):
    return ModelSpec(
        slug=slug, family=family, kind="moe", n_layers=48,
        full_attn_layers=12, kv_heads=4, head_dim=128, native_ctx=262144,
        ggufs=[GgufFile(repo="r", file=f"{slug}.gguf", bytes=1, quant="Q4")],
        custom=custom, default_params=dp or {},
        cache_type_policy=ctp or CachePolicy())


def test_custom_import_inherits_by_geometry(monkeypatch):
    from rigma import hangar

    donor = _spec("qwen3.6-35b-a3b", custom=False, family="qwen3.6",
                  dp={"temperature": 0.75, "dry_multiplier": 0.8},
                  ctp=CachePolicy(k="q8_0", v="q8_0", reason="deltanet"))

    class _FakeReg:
        models = {donor.slug: donor}

    monkeypatch.setattr("rigma.registry.Registry", type(
        "R", (), {"load": staticmethod(lambda path=None: _FakeReg())}))
    # same geometry, different family string — the real situation: gguf arch
    # is "qwen35moe", registry family is "qwen3.6"
    fresh = _spec("heretic-finetune", custom=True, family="qwen35moe")
    out = hangar.inherit_family_defaults(fresh)
    assert out.default_params.get("temperature") == 0.75
    assert out.cache_type_policy.k == "q8_0"


def test_inheritance_never_overwrites_explicit_values(monkeypatch):
    from rigma import hangar

    donor = _spec("base", custom=False, family="f",
                  dp={"temperature": 0.7})

    class _FakeReg:
        models = {donor.slug: donor}

    monkeypatch.setattr("rigma.registry.Registry", type(
        "R", (), {"load": staticmethod(lambda path=None: _FakeReg())}))
    fresh = _spec("mine", custom=True, family="x",
                  dp={"temperature": 1.1})
    out = hangar.inherit_family_defaults(fresh)
    assert out.default_params["temperature"] == 1.1   # explicit wins


def test_no_geometry_match_returns_unchanged(monkeypatch):
    from rigma import hangar

    class _FakeReg:
        models = {}

    monkeypatch.setattr("rigma.registry.Registry", type(
        "R", (), {"load": staticmethod(lambda path=None: _FakeReg())}))
    fresh = _spec("orphan", custom=True, family="x")
    assert hangar.inherit_family_defaults(fresh) is fresh


# --- cache hygiene + reasoning budget in server args --------------------------
def test_server_args_two_unified_slots():
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192),
                   origin="test")
    s = " ".join(plan.server_args("m.gguf", 11500))
    assert "--parallel 2" in s
    assert "--kv-unified" in s
    assert "--checkpoint-min-step 4096" in s


def test_server_args_reasoning_budget():
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan",
                   flags=ComboFlags(ctx=8192, reasoning_budget=2048),
                   origin="test")
    args = plan.server_args("m.gguf", 11500)
    assert args[args.index("--reasoning-budget") + 1] == "2048"
    default = RunPlan(model_slug="m",
                      gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                      backend="vulkan", flags=ComboFlags(ctx=8192),
                      origin="test")
    assert "--reasoning-budget" not in default.server_args("m.gguf", 11500)


# --- reasoning carry-over for autonomous runs ---------------------------------
def _run_session(n_turns):
    return {"one_action": True, "effort": "on", "use_tools": False,
            "messages": [
                {"role": "assistant", "content": f"turn {i}",
                 "thinking": f"thought {i}"}
                for i in range(n_turns)]}


def test_run_sessions_carry_recent_reasoning():
    msgs = sessions.build_messages(_run_session(2))
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert all(m.get("reasoning_content") for m in assistants)
    assert assistants[-1]["reasoning_content"] == "thought 1"


def test_reasoning_carry_capped_to_last_four():
    msgs = sessions.build_messages(_run_session(7))
    assistants = [m for m in msgs if m["role"] == "assistant"]
    with_r = [m for m in assistants if "reasoning_content" in m]
    assert len(with_r) == 4
    assert with_r[-1]["reasoning_content"] == "thought 6"   # newest kept


def test_chat_sessions_never_carry_reasoning():
    s = _run_session(2)
    s["one_action"] = False
    msgs = sessions.build_messages(s)
    assert all("reasoning_content" not in m for m in msgs)


def test_effort_off_never_carries_reasoning():
    s = _run_session(2)
    s["effort"] = "off"
    msgs = sessions.build_messages(s)
    assert all("reasoning_content" not in m for m in msgs)
