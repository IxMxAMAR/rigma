from rigma import bench
from rigma.models import ComboFlags, GgufFile, RunPlan


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192, **fl),
                   origin="calculator")


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
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe: [
        ("baseline", {}), ("fa-off", {"flash_attn": "off"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    assert rows[0]["label"] == "fa-off" and rows[0]["tg_tps"] == 70
    cal = bench.load_calibration()["m:Q4:vulkan"]
    assert cal["flags"]["flash_attn"] == "off"


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
    monkeypatch.setattr(bench, "quick_configs", lambda base, moe: [
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
    monkeypatch.setattr(bench, "quick_configs", lambda base, moe: [
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
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe: [
        ("baseline", {}), ("kv-q8", {"cache_type_k": "q8_0", "cache_type_v": "q8_0"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf")
    baseline = next(r for r in rows if r["label"] == "baseline")
    assert baseline["ok"] is False
    assert rows[0]["label"] == "kv-q8" and rows[0]["tg_tps"] == 42
