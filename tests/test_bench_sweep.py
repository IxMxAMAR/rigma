import inspect

from rigma import bench
from rigma.models import ComboFlags, GgufFile, RunPlan


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192, **fl),
                   origin="calculator")


def _only_entry():
    """The single entry a sweep wrote.

    R3-CAL-1 put the hardware identity in the key, so the literal
    "m:Q4:vulkan" is no longer the key on any machine — it is
    "m:Q4:vulkan:<digest>", and the digest differs per machine. A sweep writes
    exactly one entry, so taking the only one keeps these tests deterministic
    instead of making them depend on which GPU happens to be installed.
    """
    cal = bench.load_calibration()
    assert len(cal) == 1, cal
    return next(iter(cal.values()))


def test_sweep_configs_moe_includes_key_axes():
    cfgs = dict(bench.sweep_configs(ComboFlags(ctx=8192, n_cpu_moe=4), moe=True))
    labels = " ".join(cfgs)
    assert "fa-off" in labels and "kv-q8" in labels and "coopmat-off" in labels
    first = bench.sweep_configs(ComboFlags(ctx=8192), moe=True)[0]
    assert first[0] == "baseline" and first[1] == {}


def test_sweep_configs_dense_has_no_moe_axis():
    cfgs = dict(bench.sweep_configs(ComboFlags(ctx=8192), moe=False))
    assert not any("cpu-moe" in k or "gfxqueue" in k for k in cfgs)


def test_run_sweep_picks_best_and_saves(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    seq = iter([
        bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8),
        bench.BenchResult(pp_tps=120, tg_tps=70, prompt_tokens=8, gen_tokens=8),
    ])

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe, caps=(): [
        ("baseline", {}), ("fa-off", {"flash_attn": "off"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    assert rows[0]["label"] == "fa-off" and rows[0]["tg_tps"] == 70
    cal = _only_entry()
    assert cal["flags"]["flash_attn"] == "off"


def test_a_config_with_no_usable_measurement_is_a_loss(monkeypatch, tmp_path):
    """AUDIT F08-7: `run_bench` now raises when the engine reports no timings.
    The sweep must record that as a failed row — never crown it, never write it
    to calibration."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench",
                        lambda port, **k: (_ for _ in ()).throw(
                            RuntimeError("engine returned no timings")))
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe, caps=(): [
        ("baseline", {}), ("fa-off", {"flash_attn": "off"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    assert all(r["ok"] is False for r in rows)
    assert bench.crowned_row(rows) is None
    assert bench.load_calibration() == {}


def test_quick_configs_is_short_and_baseline_first():
    q = bench.quick_configs(ComboFlags(ctx=8192, n_cpu_moe=4), moe=True)
    assert q[0] == ("baseline", {})
    assert len(q) <= 4  # first-load must be fast
    labels = [k for k, _ in q]
    assert "coopmat-off" in labels  # the toggle that can't be defaulted


def test_auto_calibrate_runs_once_then_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    runs = {"n": 0}

    class _FakeSrv:
        def stop(self):
            pass

    def counting_launch(*a, **k):
        runs["n"] += 1
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", counting_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    monkeypatch.setattr(bench, "quick_configs", lambda base, moe, caps=(): [
        ("baseline", {}), ("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}})])

    plan = _plan()
    assert bench.is_calibrated("m", "Q4", "vulkan") is False
    out1 = bench.auto_calibrate(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                                port=11601)
    assert runs["n"] == 2                      # both quick configs launched once
    assert bench.is_calibrated("m", "Q4", "vulkan") is True
    # second call: no new launches, plan returned as-is (already tuned)
    out2 = bench.auto_calibrate(out1, tmp_path / "srv.exe", tmp_path / "m.gguf",
                                port=11601)
    assert runs["n"] == 2                      # unchanged — cached
    assert out2 is not None


def test_auto_calibrate_applies_winning_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    seq = iter([
        bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8),
        bench.BenchResult(pp_tps=100, tg_tps=80, prompt_tokens=8, gen_tokens=8),
    ])

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    monkeypatch.setattr(bench, "quick_configs", lambda base, moe, caps=(): [
        ("baseline", {}), ("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}})])

    out = bench.auto_calibrate(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf")
    assert out.flags.env.get("GGML_VK_DISABLE_COOPMAT") == "1"  # winner applied
    assert out.origin.endswith("+calibrated")


def test_auto_calibrate_cpu_backend_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma.models import ComboFlags, GgufFile, RunPlan
    cpu_plan = RunPlan(model_slug="m",
                       gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                       backend="cpu", flags=ComboFlags(ctx=4096), origin="calculator")

    def boom(*a, **k):
        raise AssertionError("must not launch for CPU")

    monkeypatch.setattr(bench, "launch_server", boom)
    out = bench.auto_calibrate(cpu_plan, tmp_path / "srv.exe", tmp_path / "m.gguf")
    assert out is cpu_plan
    assert bench.is_calibrated("m", "Q4", "cpu") is False


def test_run_sweep_skips_failed_launch(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    class _FakeSrv:
        def stop(self):
            pass

    calls = {"n": 0}

    def flaky_launch(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("failed to become healthy")
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", flaky_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=10, tg_tps=42, prompt_tokens=8, gen_tokens=8))
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe, caps=(): [
        ("baseline", {}), ("kv-q8", {"cache_type_k": "q8_0", "cache_type_v": "q8_0"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf")
    baseline = next(r for r in rows if r["label"] == "baseline")
    assert baseline["ok"] is False
    assert rows[0]["label"] == "kv-q8" and rows[0]["tg_tps"] == 42


def _fake_sweep(monkeypatch, results, configs):
    """Drive run_sweep without launching anything."""
    seq = iter(results)

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): configs)


def _rows_log(home):
    import json
    p = home / "logs" / "bench-rows.jsonl"
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def test_a_sweep_keeps_every_measurement_not_just_the_winner(monkeypatch,
                                                             tmp_path):
    """A sweep launches a real engine per config and measures it. Only the
    winner's two floats survived into calibration.json; the losing rows were
    returned, printed, and dropped. Those are the most expensive numbers Rigma
    ever produces — each one is a real model load."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _fake_sweep(monkeypatch,
                [bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8,
                                   gen_tokens=8),
                 bench.BenchResult(pp_tps=120, tg_tps=70, prompt_tokens=8,
                                   gen_tokens=8)],
                [("baseline", {}), ("fa-off", {"flash_attn": "off"})])
    bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                    port=11601)
    logged = _rows_log(tmp_path)
    assert {r["label"] for r in logged} == {"baseline", "fa-off"}
    by = {r["label"]: r for r in logged}
    assert by["baseline"]["tg_tps"] == 50      # the LOSER is kept
    assert by["fa-off"]["crowned"] is True
    assert by["baseline"]["crowned"] is False
    assert by["baseline"]["model"] == "m" and by["baseline"]["backend"] == "vulkan"


def test_a_sweep_the_baseline_wins_still_records_what_it_measured(monkeypatch,
                                                                  tmp_path):
    """`if best["flags"] or mark_calibrated` means an explicit sweep where the
    baseline wins writes NOTHING — not even the baseline speed it just spent
    two engine loads measuring."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _fake_sweep(monkeypatch,
                [bench.BenchResult(pp_tps=100, tg_tps=70, prompt_tokens=8,
                                   gen_tokens=8),
                 bench.BenchResult(pp_tps=120, tg_tps=50, prompt_tokens=8,
                                   gen_tokens=8)],
                [("baseline", {}), ("fa-off", {"flash_attn": "off"})])
    bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                    port=11601)
    assert bench.load_calibration() == {}          # unchanged: correct
    logged = _rows_log(tmp_path)
    assert len(logged) == 2                        # but the numbers survive
    assert {r["label"] for r in logged} == {"baseline", "fa-off"}


def test_a_failed_config_is_recorded_as_a_loss_with_its_error(monkeypatch,
                                                              tmp_path):
    """A config that OOMs is a measurement too — it says this machine cannot
    run that combination."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    def _boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(bench, "launch_server", _boom)
    # NOT a q4 KV config: run_sweep drops those for tools-capable models on
    # purpose, so using one here would test the filter, not the log
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): [("batch-big", {"batch": 16384})])
    bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                    port=11601)
    logged = _rows_log(tmp_path)
    assert len(logged) == 1
    assert logged[0]["ok"] is False
    assert "out of memory" in logged[0]["error"]


def test_the_rows_log_never_breaks_a_sweep(monkeypatch, tmp_path):
    """Logging is bookkeeping. If the log cannot be written the sweep must
    still return its rows and still save calibration."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _fake_sweep(monkeypatch,
                [bench.BenchResult(pp_tps=120, tg_tps=70, prompt_tokens=8,
                                   gen_tokens=8)],
                [("fa-off", {"flash_attn": "off"})])
    monkeypatch.setattr(bench, "_log_row",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    assert rows and rows[0]["label"] == "fa-off"
    assert _only_entry()["flags"]["flash_attn"] == "off"


def test_a_calibration_entry_says_what_it_was_measured_on(monkeypatch, tmp_path):
    """A calibration entry carried a day-granularity date and nothing else. It
    could not tell you which engine build or context it was measured at, so
    there was no way to know it had gone stale after an engine bump."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _fake_sweep(monkeypatch,
                [bench.BenchResult(pp_tps=120, tg_tps=70, prompt_tokens=8,
                                   gen_tokens=8)],
                [("fa-off", {"flash_attn": "off"})])
    bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                    port=11601)
    entry = _only_entry()
    assert entry["schema"] == 3
    assert entry["ctx"] == 8192
    assert "engine" in entry
    # R3-CAL-1: and now WHICH CARD, so a 3090 cannot silently inherit a 4090's
    # number. The identity is in the key AND recorded in the entry, so a stale
    # entry can explain itself rather than just being absent.
    assert entry["hardware"]["id"]
    assert entry["hardware"]["backend"] == "vulkan"


# The exact baseline argv as produced by the parent commit dbb2110, captured
# before C2 touched anything. `--no-op-offload` is a sweep axis, so a default
# launch must still emit this list byte for byte.
BEFORE_C2_BASELINE_ARGV = [
    "-m", "/tmp/m.gguf", "--port", "11601", "--host", "127.0.0.1",
    "-ngl", "99", "-c", "8192", "--parallel", "2", "--kv-unified",
    "--alias", "m", "-fa", "on", "--cache-type-k", "f16",
    "--cache-type-v", "f16", "--reasoning-format", "deepseek",
    "--cache-reuse", "256", "--checkpoint-min-step", "4096"]


def test_sweep_offers_the_no_op_offload_axis_and_off_emits_nothing():
    """C2: the axis is a sweep config, and it is boolean.

    `--no-op-offload` is `{"--op-offload"}, {"--no-op-offload"}` (bool) in
    common/arg.cpp at both pins, and the compiled default is op-offload ON, so
    the OFF value must add NOTHING to the argv.
    """
    cfgs = dict(bench.sweep_configs(ComboFlags(ctx=8192), moe=False))
    assert cfgs["no-op-offload"] == {"no_op_offload": True}
    # ON: the per-trial value reaches the argv.
    on = _plan(no_op_offload=True).server_args("/tmp/m.gguf", 11601)
    assert "--no-op-offload" in on
    # OFF: the default value, and the baseline trial, emit nothing at all.
    assert "--no-op-offload" not in _plan().server_args("/tmp/m.gguf", 11601)
    assert "--no-op-offload" not in _plan(
        **cfgs["baseline"]).server_args("/tmp/m.gguf", 11601)


def test_a_default_sweep_leaves_every_other_argv_unchanged():
    """C2 must not change an existing launch or an existing measurement.

    Only the new axis may carry the flag; the baseline argv is the one captured
    from the parent commit, and no other axis gained a token.
    """
    carrying = []
    for label, override in bench.sweep_configs(ComboFlags(ctx=8192), moe=False):
        argv = _plan(**override).server_args("/tmp/m.gguf", 11601)
        if "--no-op-offload" in argv:
            carrying.append(label)
        if label == "baseline":
            assert argv == BEFORE_C2_BASELINE_ARGV
    assert carrying == ["no-op-offload"]


def test_quick_configs_does_not_trial_the_no_op_offload_axis():
    """Not a launch default: the first-load calibration sweep (which is a
    default load path) must not start trialling it."""
    labels = [k for k, _ in bench.quick_configs(ComboFlags(ctx=8192), moe=False)]
    assert "no-op-offload" not in labels


def test_a_no_op_offload_row_records_the_trialled_value(monkeypatch, tmp_path):
    """A stored row is self-describing: it says which value was trialled.

    Same shape as the env toggles — the override dict IS the recorded flags, in
    the returned row, in bench-rows.jsonl and in calibration.json — and the row
    corresponds to an argv that really carried the flag.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    seq = iter([bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8,
                                  gen_tokens=8),
                bench.BenchResult(pp_tps=120, tg_tps=70, prompt_tokens=8,
                                  gen_tokens=8)])
    seen = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        from rigma import runtime
        seen.append((bool(getattr(plan.flags, "no_op_offload", False)),
                     runtime.server_argv(exe, plan, model_path,
                                         k.get("port", 11601))))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("no-op-offload", {"no_op_offload": True})])
    # the trial really launched the flag; the baseline really did not.
    assert "--no-op-offload" in seen[1][1] and seen[1][0] is True
    assert "--no-op-offload" not in seen[0][1] and seen[0][0] is False
    by_label = {r["label"]: r for r in rows}
    assert by_label["no-op-offload"]["flags"] == {"no_op_offload": True}
    assert by_label["baseline"]["flags"] == {}
    logged = {e["label"]: e for e in _rows_log(tmp_path)}
    assert logged["no-op-offload"]["flags"] == {"no_op_offload": True}
    assert logged["baseline"]["flags"] == {}
    # the winner (no-op-offload) is what calibration stores, with its value.
    assert _only_entry()["flags"] == {"no_op_offload": True}


# --- C11: the attention-rotation env toggle ---------------------------------
#
# `LLAMA_ATTN_ROT_DISABLE` has NO argv spelling at either pin, so unlike C2's
# `--no-op-offload` the only way to trial it is through the engine-spawn env
# that `runtime.launch_server` merges over `os.environ`. It is the same class
# of lever as the GGML_VK_* driver toggles already in the sweep.
#
# Provenance (raw.githubusercontent.com, fetched 2026-09-30):
#   PrismML-Eng/llama.cpp@87268f77 src/llama-kv-cache.cpp L316-326
#   ggml-org/llama.cpp@b9867     src/llama-kv-cache.cpp L329-339
#     const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
#     const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(...) : false;
#     attn_rot_k = !attn_rot_disable && ... && ggml_is_quantized(type_k) && ...
# The rotation is therefore ON BY DEFAULT for a quantized KV cache, and this
# machine's own load runs it: `.scratch/prism-v.log:4576-4578` shows
# `K (q8_0): 1088.00 MiB` with `attn_rot_k = 1, attn_rot_v = 1`.

def test_sweep_offers_the_attn_rot_axis_and_off_sets_nothing():
    """C11: the axis is a sweep config, and its OFF/default case emits NOTHING.

    `LLAMA_ATTN_ROT_DISABLE` is read with `getenv`, so an unset variable is the
    engine's untouched default (rotation ON where it applies). The baseline
    trial must therefore carry no `env` key at all — exactly C2's acceptance
    criterion for argv, applied to the child environment.
    """
    cfgs = dict(bench.sweep_configs(ComboFlags(ctx=8192), moe=False))

    assert cfgs["attn-rot-off"] == {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}
    # OFF: no env key anywhere it did not already exist, so the child inherits
    # the parent environment unchanged (runtime.launch_server only builds a
    # `Popen` env when `plan.flags.env` is non-empty).
    assert "env" not in cfgs["baseline"]
    assert _plan(**cfgs["baseline"]).flags.env == {}


def test_a_default_sweep_leaves_every_pre_existing_trial_env_unchanged():
    """C11 must not change an existing launch's environment.

    Only the new axis may carry `LLAMA_ATTN_ROT_DISABLE`; every axis that had
    no env before still has no env, so `runtime.launch_server` takes the
    `if plan.flags.env:` false branch and the child inherits `os.environ`
    byte for byte.
    """
    carrying = []
    for label, override in bench.sweep_configs(ComboFlags(ctx=8192), moe=False):
        env = override.get("env") or {}
        if "LLAMA_ATTN_ROT_DISABLE" in env:
            carrying.append(label)
            continue
        # every OTHER axis must not gain the variable, in its override or in a
        # plan built from it
        assert "LLAMA_ATTN_ROT_DISABLE" not in _plan(**override).flags.env
        if label in ("baseline", "fa-off", "kv-q8", "kv-q4", "batch-big",
                     "no-op-offload"):
            # these axes carried no engine env before C11 and still carry none
            assert "env" not in override
    assert carrying == ["attn-rot-off"]


def test_the_attn_rot_axis_is_the_childs_environment_only(monkeypatch):
    """The ON case is a CHILD-only environment.

    `runtime.launch_server` merges `plan.flags.env` over `os.environ` into the
    `Popen` call, so constructing or running the trial must never write the
    parent's `os.environ` — the lever would otherwise leak into every later
    launch in this process.
    """
    import os

    monkeypatch.delenv("LLAMA_ATTN_ROT_DISABLE", raising=False)
    override = dict(bench.sweep_configs(ComboFlags(ctx=8192),
                                        moe=False))["attn-rot-off"]

    assert _plan(**override).flags.env == {"LLAMA_ATTN_ROT_DISABLE": "1"}
    assert "LLAMA_ATTN_ROT_DISABLE" not in os.environ


def test_a_tools_capable_default_sweep_does_not_trial_the_attn_rot_axis(
        monkeypatch, tmp_path):
    """The blast-radius fix. `sweep_configs` still OFFERS the axis (that is
    where the lever lives), but the default `run_sweep` path for a tools-capable
    model drops it exactly as it drops the q4_0 KV axis — so it is never
    trialled, never crowned, and never persisted into calibration.json, which
    `resolve()` would then merge into EVERY later launch's child environment.

    The sweep still runs and still crowns a flagged row, so this is not a
    vacuous "nothing was saved" pass."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append(dict(plan.flags.env))
        return _FakeSrv()

    n = {"i": 0}

    def fake_bench(port, **k):
        n["i"] += 1
        return bench.BenchResult(pp_tps=100, tg_tps=float(n["i"]),
                                 prompt_tokens=8, gen_tokens=8)

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", fake_bench)
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)

    labels = [r["label"] for r in rows]
    assert "attn-rot-off" not in labels
    assert "kv-q4" not in labels          # the pre-existing guard still holds
    assert all("LLAMA_ATTN_ROT_DISABLE" not in env for env in launched)
    # the sweep did crown and persist something — just not a quality lever
    assert _only_entry()["flags"] == {"no_op_offload": True}


def test_crowned_row_refuses_a_quality_lever_row_without_an_opt_in():
    """The single rule both readers share (`run_sweep` saves it, `rigma sweep`
    prints it): a tokens/sec-only score may not crown a quality-degrading env
    lever. The opt-in is explicit, so the CLI — which never passes it — and the
    saved calibration cannot disagree."""
    rows = [{"label": "attn-rot-off", "tg_tps": 99.0, "ok": True,
             "flags": {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}},
            {"label": "fa-off", "tg_tps": 50.0, "ok": True,
             "flags": {"flash_attn": "off"}}]

    assert bench.crowned_row(rows)["label"] == "fa-off"
    assert bench.crowned_row(rows, allow_quality_levers=True)["label"] == \
        "attn-rot-off"


def test_quick_configs_does_not_trial_the_attn_rot_axis():
    """Not a launch default: the first-load calibration sweep (a default load
    path) must not start trialling it."""
    labels = [k for k, _ in bench.quick_configs(ComboFlags(ctx=8192), moe=False)]
    assert "attn-rot-off" not in labels


def test_a_tools_capable_sweep_drops_the_axis_even_when_it_is_the_fastest(
        monkeypatch, tmp_path):
    """A caller that hands `run_sweep` the rotation config explicitly cannot get
    it crowned either: the config is dropped before it launches, so it never
    even costs a model load."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    seq = iter([bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8,
                                  gen_tokens=8),
                bench.BenchResult(pp_tps=120, tg_tps=99, prompt_tokens=8,
                                  gen_tokens=8)])
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append(dict(plan.flags.env))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("attn-rot-off",
                                     {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}})])
    assert [r["label"] for r in rows] == ["baseline"]
    assert launched == [{}]
    assert bench.load_calibration() == {}


def test_a_tools_capable_sweep_drops_a_lever_the_plan_already_carries(
        monkeypatch, tmp_path):
    """The trial gate must read the EFFECTIVE child environment, not the
    per-trial override.

    A plan that already carries the lever on `plan.flags.env` — the state
    `rigma sweep` reaches when a `calibration.json` entry is merged by
    `resolve._apply_calibration` — launched EVERY trial with it while the gate,
    looking only at the override, believed the axis had been dropped. The saved
    flags are still the override, so this contaminates the measurement rather
    than reopening the write path; the point is that the guard's stated intent
    ("does not trial it on a tools-capable model") is not met in that state.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append(dict(plan.flags.env))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan(env={"LLAMA_ATTN_ROT_DISABLE": "1"})
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("kv-q8", {"cache_type_k": "q8_0",
                                               "cache_type_v": "q8_0"}),
                                    ("coopmat-off",
                                     {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}})])
    # No launched trial may inherit the lever: the two that would have (their
    # override names no env of its own) are dropped before they cost a load.
    assert all("LLAMA_ATTN_ROT_DISABLE" not in env for env in launched)
    assert [r["label"] for r in rows] == ["coopmat-off"]
    # ...and the survivor does so because its OWN env REPLACES the plan's
    # (`model_copy(update=...)`), so it genuinely runs lever-free. It is then
    # crowned and persisted — with the coopmat toggle, never the lever.
    assert launched == [{"GGML_VK_DISABLE_COOPMAT": "1"}]
    assert _only_entry()["flags"] == {
        "env": {"GGML_VK_DISABLE_COOPMAT": "1"}}


def test_a_plan_carrying_an_unrelated_env_var_is_still_trialled(
        monkeypatch, tmp_path):
    """The effective-env gate must refuse only a quality lever.

    A plan that carries some OTHER engine env var (a driver toggle, say) must
    still be trialled normally — the fix must not turn any pre-existing
    `plan.flags.env` into a refusal.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append(dict(plan.flags.env))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan(env={"GGML_VK_DISABLE_COOPMAT": "1"})
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601, configs=[("baseline", {})])
    assert [r["label"] for r in rows] == ["baseline"]
    assert launched == [{"GGML_VK_DISABLE_COOPMAT": "1"}]


def test_a_non_tools_sweep_trials_the_attn_rot_axis_but_does_not_crown_it(
        monkeypatch, tmp_path):
    """C11's lever is not deleted: a non-tools model still MEASURES the axis, so
    its direction is recorded in bench-rows.jsonl for an operator to read. But
    the row cannot win — and so cannot be persisted — on a tokens/sec-only
    score, because rotation is what keeps a quantized KV cache's loss down."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: False)
    seq = iter([bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8,
                                  gen_tokens=8),
                bench.BenchResult(pp_tps=120, tg_tps=99, prompt_tokens=8,
                                  gen_tokens=8)])
    seen = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        seen.append(dict(plan.flags.env))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("attn-rot-off",
                                     {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}})])
    # the trial really launched with the variable, and its row is recorded...
    assert seen[1] == {"LLAMA_ATTN_ROT_DISABLE": "1"}
    assert seen[0] == {}
    by_label = {r["label"]: r for r in rows}
    assert by_label["attn-rot-off"]["flags"] == {
        "env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}
    logged = {e["label"]: e for e in _rows_log(tmp_path)}
    assert logged["attn-rot-off"]["flags"] == {
        "env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}
    assert logged["baseline"]["flags"] == {}
    # ...but it is not crowned, and nothing at all was persisted.
    assert bench.crowned_row(rows)["label"] == "baseline"
    assert bench.load_calibration() == {}


def test_the_attn_rot_axis_is_crowned_only_with_an_explicit_opt_in(
        monkeypatch, tmp_path):
    """The escape hatch: an operator who accepts the quality trade can let the
    rotation row win. `rigma sweep` never passes this, so the winner the CLI
    announces and the flags saved to calibration stay in step."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: False)
    seq = iter([bench.BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=8,
                                  gen_tokens=8),
                bench.BenchResult(pp_tps=120, tg_tps=99, prompt_tokens=8,
                                  gen_tokens=8)])

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: next(seq))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("attn-rot-off",
                                     {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}})],
                           allow_quality_levers=True)
    assert bench.crowned_row(rows, allow_quality_levers=True)["label"] == \
        "attn-rot-off"
    assert _only_entry()["flags"] == {"env": {"LLAMA_ATTN_ROT_DISABLE": "1"}}


def test_the_quality_lever_match_is_case_insensitive():
    """The gate must match the spelling the ENGINE honours.

    llama.cpp reads the lever with the C library's `getenv`
    (ggml-org/llama.cpp@b9867 src/llama-kv-cache.cpp:
    `getenv("LLAMA_ATTN_ROT_DISABLE")`), and on Windows that lookup ignores
    case — Microsoft's own reference: "getenv isn't case sensitive in the
    Windows operating system"
    (learn.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv).
    A lowercase `llama_attn_rot_disable` therefore really does disable the
    rotation there, so the gate must see it. On POSIX `getenv` IS
    case-sensitive, so this over-matches there — the safe direction (a dropped
    trial, never a crowned or persisted lever).
    """
    assert bench._carries_quality_env_lever(
        {"env": {"llama_attn_rot_disable": "1"}}) is True
    assert bench._carries_quality_env_lever(
        {"env": {"Llama_Attn_Rot_Disable": "1"}}) is True
    # the exact spelling still matches, and an unrelated var still does not
    assert bench._carries_quality_env_lever(
        {"env": {"LLAMA_ATTN_ROT_DISABLE": "0"}}) is True
    assert bench._carries_quality_env_lever(
        {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}}) is False


def test_a_tools_capable_sweep_drops_a_lowercase_lever_too(monkeypatch,
                                                           tmp_path):
    """The gate runs on the spelling the engine will actually honour on this
    platform: a lowercase `llama_attn_rot_disable` config is dropped before it
    costs a model load, exactly like the canonical one."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append(dict(plan.flags.env))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("attn-rot-off",
                                     {"env": {"llama_attn_rot_disable": "1"}})])
    assert [r["label"] for r in rows] == ["baseline"]
    assert launched == [{}]


# ---------------------------------------------------------------------------
# DR3-2: the KV-cache guard must read the EFFECTIVE flags, not the override.
#
# `cache_type_k`/`cache_type_v` are in `resolve._CALIBRATION_PLACEMENT_KEYS`, so
# a plan can already carry q4_0 (a row written when the slug was not yet known
# to be tools-capable, hand-edited, or written by an older build). The old check
# looked only at the override, so every config that did not itself name a cache
# type was trialled with the very cache the guard exists to keep off a
# tools-capable model — the same asymmetry `_effective_env` closes next door.
# ---------------------------------------------------------------------------

def test_a_tools_capable_sweep_drops_a_q4_plan_it_already_carries(
        monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append((plan.flags.cache_type_k, plan.flags.cache_type_v))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan(cache_type_k="q4_0", cache_type_v="q4_0")
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("kv-q8", {"cache_type_k": "q8_0",
                                               "cache_type_v": "q8_0"}),
                                    ("kv-q4", {"cache_type_k": "q4_0",
                                               "cache_type_v": "q4_0"})])
    # Only the config that REPLACES the cache type may run: `baseline` inherits
    # the plan's q4_0 and `kv-q4` asks for it outright.
    assert [r["label"] for r in rows] == ["kv-q8"]
    assert launched == [("q8_0", "q8_0")]


def test_a_tools_capable_q4_plan_that_drops_everything_says_so(
        monkeypatch, tmp_path, caplog):
    """Dropping trials cannot fix a plan that already carries q4_0, and an
    empty sweep must not look like "nothing worth measuring"."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan(cache_type_k="q4_0", cache_type_v="q4_0")
    with caplog.at_level("WARNING", logger="rigma.bench"):
        rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                               port=11601,
                               configs=[("baseline", {}),
                                        ("kv-q4", {"cache_type_k": "q4_0",
                                                   "cache_type_v": "q4_0"})])
    assert rows == []
    assert "already quantizes the KV cache to q4_0" in caplog.text
    assert "every config was dropped" in caplog.text


def test_a_calibration_merged_one_sided_q4_plan_is_dropped_too(
        monkeypatch, tmp_path):
    """The shape the CONSTRUCTOR cannot make but a calibration merge can.

    `ComboFlags` normalizes a mismatched cache pair to the safer side
    (`q4_0`/`q8_0` -> `q8_0`/`q8_0`), so a plan built through the constructor
    carries q4_0 only when BOTH sides ask for it. `model_copy(update=...)`
    bypasses that validation, and `cache_type_k`/`cache_type_v` are both in
    `resolve._CALIBRATION_PLACEMENT_KEYS` — so `_apply_calibration` merging an
    older or hand-edited row CAN put q4_0 on one side only. The guard therefore
    ORs the two sides rather than checking one.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: True)
    launched = []

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, plan, model_path, **k):
        launched.append((plan.flags.cache_type_k, plan.flags.cache_type_v))
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan()
    # exactly what a calibration merge does: model_copy, no validation
    plan = plan.model_copy(update={
        "flags": plan.flags.model_copy(update={"cache_type_k": "q4_0"})})
    assert plan.flags.cache_type_k == "q4_0"
    assert plan.flags.cache_type_v != "q4_0"      # one side only
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("kv-q8", {"cache_type_k": "q8_0",
                                               "cache_type_v": "q8_0"})])
    assert [r["label"] for r in rows] == ["kv-q8"]
    assert launched == [("q8_0", "q8_0")]


def test_a_non_tools_sweep_still_trials_a_q4_plan(monkeypatch, tmp_path):
    """The non-tools path is unchanged by DR3-2: q4_0 is a legitimate
    measurement there (it just cannot be crowned), so a plan carrying it must
    not silently lose its baseline."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(bench, "_tools_capable", lambda slug: False)

    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    plan = _plan(cache_type_k="q4_0", cache_type_v="q4_0")
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601,
                           configs=[("baseline", {}),
                                    ("kv-q8", {"cache_type_k": "q8_0",
                                               "cache_type_v": "q8_0"})])
    assert [r["label"] for r in rows] == ["baseline", "kv-q8"]


def test_the_sweep_and_its_guards_build_a_trial_the_same_way(monkeypatch, tmp_path):
    """W13B-1: one construction, not three copies of it.

    `run_sweep`'s launch loop and the two trial guards (`_effective_env`,
    `_effective_flags`) each used to spell out
    `plan.flags.model_copy(update=override)` for themselves. They agreed, but
    nothing kept them agreeing — and a guard that reads a different flag set from
    the one the child is launched with is worse than no guard, because it reports
    on a trial that never ran. This pins the single source on all three paths:
    every construction records its CALLER, so re-inlining either guard fails, and
    the flags the child is launched with are the helper's RETURN value, so a
    wrong return fails.

    W16A-1: the previous version inspected only the TAIL of the recorded calls
    and never looked at what was launched, so inlining either guard, or making
    `_trial_flags` return the wrong flags, left it green.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    configs = [("baseline", {}), ("fa-off", {"flash_attn": "off"})]
    real = bench._trial_flags
    calls: list[tuple[str, dict]] = []
    launched: list = []

    def spy(plan_flags, override):
        # The immediate caller's name: `run_sweep` for the launch loop,
        # `_effective_flags`/`_effective_env` for the guards. A guard re-inlined
        # to `plan_flags.model_copy(...)` stops calling here.
        caller = inspect.currentframe().f_back.f_code.co_name
        calls.append((caller, dict(override)))
        return real(plan_flags, override)

    monkeypatch.setattr(bench, "_trial_flags", spy)

    class _FakeSrv:
        def stop(self):
            pass

    def fake_launch(exe, trial, model_path, **kw):
        launched.append(trial)
        return _FakeSrv()

    monkeypatch.setattr(bench, "launch_server", fake_launch)
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=100, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): list(configs))

    bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf", port=11602)

    expected = [o for _, o in configs]
    by_caller: dict[str, list[dict]] = {}
    for caller, override in calls:
        by_caller.setdefault(caller, []).append(override)

    # (1) All three paths — the launch loop and BOTH guards — read the single
    # source, once per config, in config order.
    assert by_caller.get("run_sweep") == expected, (
        "run_sweep must build every launched trial through bench._trial_flags")
    assert by_caller.get("_effective_flags") == expected, (
        "_effective_flags must read the trial through bench._trial_flags, not "
        "re-inline plan.flags.model_copy(update=override)")
    assert by_caller.get("_effective_env") == expected, (
        "_effective_env must read the trial through bench._trial_flags, not "
        "re-inline plan.flags.model_copy(update=override)")

    # (2) The helper's RETURN VALUE is what reaches the child, so a wrong return
    # cannot silently launch flags the guards never saw.
    assert [t.flags for t in launched] == [
        _plan().flags.model_copy(update=o) for o in expected], (
        "the flags the child was launched with are not bench._trial_flags's "
        "return value")

