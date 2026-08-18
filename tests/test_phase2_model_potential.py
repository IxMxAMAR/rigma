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


# --- MTP capability drives the spec trial (EXHAUSTIVE sweep only — live
# 2026-07-21: the short predictable bench inflates draft acceptance; a
# crowned 44.7 bench ran at 36.8 live, so auto first-load never tries spec) --
def test_sweep_configs_add_mtp_trials_only_with_capability():
    base = ComboFlags(ctx=8192, n_cpu_moe=4)
    labels = [k for k, _ in bench.sweep_configs(base, moe=True, caps=("mtp",))]
    assert "spec-mtp-2" in labels and "spec-mtp-4" in labels
    labels_no = [k for k, _ in bench.sweep_configs(base, moe=True)]
    assert not any("spec" in x for x in labels_no)


def test_quick_configs_never_include_spec_trials():
    base = ComboFlags(ctx=8192, n_cpu_moe=4)
    labels = [k for k, _ in bench.quick_configs(base, moe=True, caps=("mtp",))]
    assert not any("spec" in x for x in labels)


def test_mtp_trial_flags_are_launchable():
    cfgs = dict(bench.sweep_configs(ComboFlags(ctx=8192), moe=True,
                                    caps=["mtp"]))
    flags = ComboFlags(ctx=8192).model_copy(update=cfgs["spec-mtp-2"])
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=flags, origin="test")
    args = plan.server_args("m.gguf", 11500)
    i = args.index("--spec-type")
    assert args[i + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-n-max") + 1] == "2"


def test_spec_crown_needs_decisive_margin(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    class _FakeSrv:
        def stop(self):
            pass

    results = {"baseline": 40.0, "spec-mtp-4": 44.0}   # +10% < 15% margin
    seq = {"i": 0}
    labels_in_order = []

    def fake_bench(port, **k):
        label = labels_in_order[seq["i"]]
        seq["i"] += 1
        return bench.BenchResult(pp_tps=100, tg_tps=results[label],
                                 prompt_tokens=8, gen_tokens=8)

    cfgs = [("baseline", {}),
            ("spec-mtp-4", {"spec_type": "draft-mtp", "spec_n_max": 4})]
    labels_in_order.extend(k for k, _ in cfgs)
    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", fake_bench)
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192),
                   origin="test")
    bench.run_sweep(plan, tmp_path / "s.exe", tmp_path / "m.gguf",
                    port=11601, configs=cfgs, mark_calibrated=True)
    crowned = bench.load_calibration()["m:Q4:vulkan"]["flags"]
    assert "spec_type" not in crowned      # narrow bench win never crowns spec


def test_speculation_is_only_trialled_when_the_file_carries_the_head(tmp_path,
                                                                     monkeypatch):
    """A sweep runs real llama-server launches, so offering draft-mtp against a
    quant without the nextn tensors is not a wasted row — it is the documented
    Vulkan driver reset. The model's capability list cannot answer this: MTP is
    kept or dropped per artefact, so the FILE has to be read (parser coverage
    lives in test_gguf_meta.py)."""
    from rigma import bench, hangar
    from rigma.models import ComboFlags, GgufFile
    from rigma.resolve import RunPlan

    monkeypatch.setattr(hangar, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(bench, "_capabilities", lambda slug: ("tools", "mtp"))

    def plan_for(name):
        return RunPlan(model_slug="m",
                       gguf=GgufFile(repo="r", file=name, bytes=1, quant="Q4"),
                       backend="vulkan", flags=ComboFlags(ctx=8192),
                       origin="test")

    def labels(name):
        caps = bench._sweepable_caps(plan_for(name))
        return [lb for lb, _ in bench.sweep_configs(ComboFlags(ctx=8192), True,
                                                    caps)]

    _write_qwen35_gguf(tmp_path / "with.gguf", nextn=True)
    _write_qwen35_gguf(tmp_path / "without.gguf", nextn=False)
    assert any(lb.startswith("spec-mtp") for lb in labels("with.gguf"))
    assert not any(lb.startswith("spec-mtp") for lb in labels("without.gguf"))
    # never downloaded => unverifiable => not offered
    assert not any(lb.startswith("spec-mtp") for lb in labels("absent.gguf"))


def _write_qwen35_gguf(path, *, nextn: bool):
    """Minimal real gguf: a header the parser accepts plus a tensor table that
    either does or does not carry the MTP projections."""
    import struct

    def s(b):
        return struct.pack("<Q", len(b)) + b

    def kv_u32(k, v):
        return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)

    def tensor(name, dims):
        return (s(name) + struct.pack("<I", len(dims))
                + b"".join(struct.pack("<Q", d) for d in dims)
                + struct.pack("<I", 0) + struct.pack("<Q", 0))

    kvs = [s(b"general.architecture") + struct.pack("<I", 8) + s(b"qwen35moe"),
           kv_u32(b"qwen35moe.block_count", 5),
           kv_u32(b"qwen35moe.context_length", 4096),
           kv_u32(b"qwen35moe.embedding_length", 512),
           kv_u32(b"qwen35moe.attention.head_count", 8),
           kv_u32(b"qwen35moe.attention.head_count_kv", 2),
           kv_u32(b"qwen35moe.attention.key_length", 64),
           kv_u32(b"qwen35moe.expert_count", 8),
           kv_u32(b"qwen35moe.nextn_predict_layers", 1)]
    tensors = [tensor(b"blk.0.attn_q.weight", [8, 8])]
    if nextn:
        tensors.append(tensor(b"blk.4.nextn.eh_proj.weight", [8, 8]))
    path.write_bytes(b"GGUF" + struct.pack("<I", 3)
                     + struct.pack("<Q", len(tensors))
                     + struct.pack("<Q", len(kvs))
                     + b"".join(kvs) + b"".join(tensors))


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


def test_no_geometry_match_keeps_the_card_but_still_gets_repetition_control(
        monkeypatch):
    """This used to assert the spec came back UNCHANGED. That contract was the
    bug: a model matching no curated geometry inherited nothing and ran on
    llama-server's bare defaults — repeat_penalty 1.0 and DRY off. Live
    2026-08-18 that produced 81,544 characters of the same stanza repeated
    ~200 times. No match now means "your own sampling plus repetition
    control", never "nothing"."""
    from rigma import hangar

    class _FakeReg:
        models = {}

    monkeypatch.setattr("rigma.registry.Registry", type(
        "R", (), {"load": staticmethod(lambda path=None: _FakeReg())}))
    fresh = _spec("orphan", custom=True, family="x")
    out = hangar.inherit_family_defaults(fresh)
    assert out.default_params.get("dry_multiplier")
    # nothing ELSE is invented — the cache policy is still left alone
    assert out.cache_type_policy == fresh.cache_type_policy
    assert out.slug == fresh.slug and out.n_layers == fresh.n_layers


def test_a_declared_sampling_block_is_carried_into_the_card(monkeypatch):
    """Qwen3.8 ggufs state their own preset in general.sampling.* (top_k 20,
    top_p 0.95, temp 1.0 — the published thinking-mode values). Rigma was not
    reading it, so a model shipping its own recommendation ran on the engine's
    generic defaults instead."""
    from rigma import hangar

    class _FakeReg:
        models = {}

    monkeypatch.setattr("rigma.registry.Registry", type(
        "R", (), {"load": staticmethod(lambda path=None: _FakeReg())}))
    probe = {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0}}
    out = hangar.inherit_family_defaults(_spec("o2", custom=True, family="x"),
                                         probe)
    assert out.default_params["top_k"] == 20.0
    assert out.default_params["top_p"] == 0.95
    assert out.default_params.get("dry_multiplier")   # plus what it omits


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
