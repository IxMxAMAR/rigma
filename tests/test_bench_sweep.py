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
