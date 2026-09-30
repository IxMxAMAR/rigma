from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()
RAW = [{"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT", "vram_mb": 16368}]


def _fake_probe(gpu_table, raw_gpus=None):
    # Fully canonical profile — REAL probe reads this machine's RAM, and a
    # hardware upgrade (16->32GB, 2026-07-14) changed combo resolution and
    # broke these tests. Never let real hardware leak into assertions.
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16368,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def test_doctor(monkeypatch):
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["doctor"])
    assert res.exit_code == 0 and "rx-9070-xt" in res.output.lower()


def test_session_exec_grants_and_revokes(tmp_path, monkeypatch):
    """AUDIT 13-3 regression: execution needs an explicit per-session grant,
    and there was no way to give one. `rigma session exec` is that way."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import sessions
    sid = sessions.create()["id"]
    # the safe direction: a fresh chat is NOT granted
    assert sessions.load(sid)["confirm_exec"] is False

    res = runner.invoke(cli.app, ["session", "exec", sid, "--on"])
    assert res.exit_code == 0 and "granted" in res.output
    assert sessions.load(sid)["confirm_exec"] is True

    res = runner.invoke(cli.app, ["session", "exec", sid, "--off"])
    assert res.exit_code == 0 and "revoked" in res.output
    assert sessions.load(sid)["confirm_exec"] is False


def test_session_exec_without_a_flag_only_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import sessions
    sid = sessions.create()["id"]
    res = runner.invoke(cli.app, ["session", "exec", sid])
    assert res.exit_code == 0 and "off" in res.output
    assert sessions.load(sid)["confirm_exec"] is False


def test_session_exec_refuses_an_unknown_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["session", "exec", "nope", "--on"])
    assert res.exit_code == 1 and "no such session" in res.output


def test_session_exec_rejects_a_non_boolean_grant(tmp_path, monkeypatch):
    """`bool("false")` is True: a stringly-typed grant must be refused, not
    read as a yes."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import sessions
    sessions.create()
    try:
        sessions.validate_field_types({"confirm_exec": "false"})
        raise AssertionError("a string grant was accepted")
    except ValueError as e:
        assert "confirm_exec" in str(e)


def test_memory_list_and_forget(tmp_path, monkeypatch):
    """IMP-6: the CLI can see what Rigma learned and delete one rule."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma.memory import MemoryStore
    store = MemoryStore(tmp_path / "memory" / "memories.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.")

    res = runner.invoke(cli.app, ["memory", "list"])
    assert res.exit_code == 0
    assert m["id"] in res.output and "Never type filenames." in res.output

    res = runner.invoke(cli.app, ["memory", "forget", m["id"]])
    assert res.exit_code == 0 and "forgot" in res.output
    assert store.all() == []

    res = runner.invoke(cli.app, ["memory", "forget", "nope"])
    assert res.exit_code == 1 and "no such memory" in res.output


def test_plan_explain(monkeypatch):
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["plan", "--use-case", "coding", "--explain"])
    assert res.exit_code == 0
    assert "UD-Q3_K_XL" in res.output and "combo:" in res.output


def _profile_with(disk=400.0, gpus=True, backends=("vulkan", "rocm")):
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    gpu = []
    if gpus:
        gpu = [GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT",
                       vram_mb=16368, arch="rdna4",
                       slug="amd-radeon-rx-9070-xt-16g",
                       backends=list(backends))]
    return HardwareProfile(gpus=gpu, ram_mb=16234, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=disk)


def _run_doctor(monkeypatch, args=(), **kw):
    monkeypatch.setattr(cli, "probe_hardware", lambda gpus: _profile_with(**kw))
    monkeypatch.setattr(cli, "_port_status", lambda port: "")
    return runner.invoke(cli.app, ["doctor", *args])


def test_doctor_runs_every_check_and_only_warns(monkeypatch):
    res = _run_doctor(monkeypatch)
    assert res.exit_code == 0
    for check in ("gpu", "engine", "model", "port", "disk", "mcp"):
        assert check in res.output
    assert "rx-9070-xt" in res.output.lower()
    assert "fail" not in res.output


def test_doctor_fails_without_a_gpu(monkeypatch):
    res = _run_doctor(monkeypatch, gpus=False)
    assert res.exit_code == 1
    assert "fail" in res.output and "GPU" in res.output
    assert "fix:" in res.output


def test_doctor_fails_on_a_gpu_with_no_backend(monkeypatch):
    res = _run_doctor(monkeypatch, backends=())
    assert res.exit_code == 1
    assert "no compute backend" in res.output


def test_doctor_warns_when_disk_is_tight(monkeypatch):
    res = _run_doctor(monkeypatch, disk=6.0)
    assert res.exit_code == 0
    assert "warn" in res.output and "disk" in res.output


def test_doctor_warns_on_a_broken_mcp_json(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "mcp.json").write_text("{oops", encoding="utf-8")
    res = _run_doctor(monkeypatch)
    assert res.exit_code == 0
    assert "mcp.json does not parse" in res.output
    assert "fix:" in res.output


def test_doctor_reports_a_valid_mcp_json(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "mcp.json").write_text(
        '{"mcpServers": {"a": {"command": "x"}}}', encoding="utf-8")
    res = _run_doctor(monkeypatch)
    assert res.exit_code == 0
    assert "1 MCP server(s) configured" in res.output


def test_doctor_never_binds_a_port(monkeypatch):
    """The port check is read-only: a doctor that probed by binding would
    collide with the port of a server that is already running."""
    monkeypatch.setattr(cli, "probe_hardware", lambda gpus: _profile_with())

    def _boom(port):
        raise AssertionError("doctor bound a port")

    monkeypatch.setattr(cli, "_port_holder", _boom)
    monkeypatch.setattr(cli, "_port_status", lambda port: "")
    res = runner.invoke(cli.app, ["doctor"])
    assert res.exit_code == 0


def test_doctor_json_is_machine_readable(monkeypatch):
    import json as _json
    res = _run_doctor(monkeypatch, ["--json"])
    rows = _json.loads(res.output)
    assert {r["id"] for r in rows} >= {"gpu", "engine", "model", "port",
                                       "disk", "mcp"}
    assert all(r["state"] in ("ok", "warn", "fail") for r in rows)


def test_status_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0 and "not running" in res.output.lower()


def test_status_ui_only_reports_no_model(tmp_path, monkeypatch):
    """AUDIT F15-2: the record `rigma up` (no model) writes has model="" and
    unloaded=True; status used to print `running:  ()`, which reads as broken."""
    import os
    from rigma import state as st
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("", "", 11500, engine_pid=-1, ui_pid=os.getpid(),
                   backend="", use_case="general", ctx=0, unloaded=True)
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0
    assert "no model loaded" in res.output
    assert "running:  ()" not in res.output


def test_stop_when_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["stop"])
    assert res.exit_code == 0 and "not running" in res.output.lower()


def test_stop_says_so_in_its_exit_code_when_it_stopped_nothing(
        tmp_path, monkeypatch):
    """A stale record is not a successful stop.

    The message was honest — "stale state — nothing was killed" — and the exit
    code was 0 either way, so `rigma stop && rigma up` walked straight past a
    stop that did not happen, and a script could not tell the two apart at all.

    The record here names a pid that IS alive (this test process) with a
    create time that does not match it, which is exactly what a recycled pid
    looks like to `kill_recorded`. Nothing is killed: the identity check
    refuses, and that refusal is what has to reach the exit code.
    """
    import json as _json
    import os as _os

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import state as st
    st.write_state("m", "q4", 11500, engine_pid=-1, ui_pid=-1, backend="",
                   use_case="general", ctx=0, unloaded=True)
    path = st.state_path()
    rec = _json.loads(path.read_text(encoding="utf-8"))
    rec["ui_pid"] = _os.getpid()          # alive, but not the recorded process
    rec["ui_started_at"] = 1.0
    path.write_text(_json.dumps(rec), encoding="utf-8")
    assert st._is_recorded_process(_os.getpid(), 1.0) is False

    res = runner.invoke(cli.app, ["stop"])
    assert "stale state" in res.output.lower(), res.output
    assert res.exit_code == 1, res.output


def test_stop_that_stopped_something_exits_zero(tmp_path, monkeypatch):
    """The other direction: the honest success must stay a success."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import state as st
    st.write_state("m", "q4", 11500, engine_pid=-1, ui_pid=-1, backend="",
                   use_case="general", ctx=0, unloaded=True)
    killed = []
    monkeypatch.setattr(st, "kill_recorded",
                        lambda s, key: killed.append(key) or True)
    res = runner.invoke(cli.app, ["stop"])
    assert res.exit_code == 0, res.output
    assert killed


def test_harness_exits_nonzero_when_the_build_drifted(monkeypatch):
    """`rigma harness` exists to be the thing that notices. Reporting drift in
    its output while exiting 0 makes `rigma harness && deploy` a green light on
    a backend whose event schema nobody has checked."""
    import dataclasses

    from rigma import harness as seam
    from rigma import harness_mcode

    monkeypatch.setattr(harness_mcode, "backend_version",
                        lambda exe=None: "0.5.4")
    monkeypatch.setitem(seam.BACKENDS, "mcode", dataclasses.replace(
        seam.BACKENDS["mcode"], verified="0.5.1", probe=lambda: True))
    res = runner.invoke(cli.app, ["harness", "-b", "mcode"])
    assert "DRIFTED" in res.output, res.output
    assert res.exit_code == 1, res.output


def test_harness_exits_zero_when_nothing_drifted(monkeypatch):
    import dataclasses

    from rigma import harness as seam
    from rigma import harness_mcode

    monkeypatch.setattr(harness_mcode, "backend_version",
                        lambda exe=None: "0.5.1")
    monkeypatch.setitem(seam.BACKENDS, "mcode", dataclasses.replace(
        seam.BACKENDS["mcode"], verified="0.5.1", probe=lambda: True))
    res = runner.invoke(cli.app, ["harness", "-b", "mcode"])
    assert res.exit_code == 0, res.output


def test_up_dry_run_ui_only(tmp_path, monkeypatch):
    # bare `up` now starts the UI with no model — dry-run says exactly that
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--dry-run"])
    assert res.exit_code == 0 and "no model" in res.output.lower()


def test_up_dry_run_with_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--dry-run"])
    assert res.exit_code == 0
    assert "qwen3.6-35b-a3b" in res.output and "-fa on" in res.output


def test_ui_only_up_does_not_probe_hardware(tmp_path, monkeypatch):
    """AUDIT F15-1: README says `rigma up` (no --model) does not probe; the
    probe ran first, above the UI-only branch, and its result was discarded."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    calls = []

    def fake_profile(reg):
        calls.append(reg)
        return _fake_probe(reg.gpus)

    monkeypatch.setattr(cli, "_profile", fake_profile)
    res = runner.invoke(cli.app, ["up", "--dry-run"])
    assert res.exit_code == 0 and "no model" in res.output.lower()
    assert calls == []
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--dry-run"])
    assert res.exit_code == 0 and len(calls) == 1


def test_chat_requires_running_server(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["chat"])
    assert res.exit_code == 1 and "not running" in res.output.lower()


def test_up_refuses_double_start(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import state as st
    st.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--use-case", "coding"])
    assert res.exit_code == 1 and "already running" in res.output.lower()


def test_up_ctx_override_and_clamp(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--ctx", "4096"])
    assert res.exit_code == 0 and "-c 4096" in res.output
    assert "+ctx-override" in res.output
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--ctx", "99999999"])
    assert res.exit_code == 0 and "-c 262144" in res.output  # qwen native cap


def test_up_surfaces_a_clamped_default_ngl(tmp_path, monkeypatch):
    """C10-nits N3: the resolver records a clamp in `explain` ("launch default
    ngl 999 exceeds what fits ...: using 99"), but `rigma up` never printed it —
    so a launch could run at a clamped ngl with no notice while `rigma plan
    --explain` showed the note. The dry-run preview must show it too."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    from rigma.models import LaunchDefaults
    real_load = cli.Registry.load

    def _load(*a, **k):
        reg = real_load(*a, **k)
        models = dict(reg.models)
        spec = models["qwen3.6-35b-a3b"]
        models["qwen3.6-35b-a3b"] = spec.model_copy(
            update={"launch": LaunchDefaults(ngl=999)})
        return cli.Registry(reg.gpus, models, reg.combos)

    monkeypatch.setattr(cli.Registry, "load", staticmethod(_load))
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "launch default ngl 999 exceeds what fits" in res.output


def test_up_reasoning_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--reasoning", "off"])
    assert res.exit_code == 0 and "--reasoning off" in res.output
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--reasoning", "sideways"])
    assert res.exit_code != 0


def test_up_fa_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--fa", "auto"])
    assert res.exit_code == 0 and "-fa auto" in res.output
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--fa", "maybe"])
    assert res.exit_code != 0


def test_up_spec_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--spec", "ngram-simple"])
    assert res.exit_code == 0 and "--spec-type ngram-simple" in res.output
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b", "--use-case", "coding", "--dry-run",
                                  "--spec", "warp-drive"])
    assert res.exit_code != 0


def test_unknown_model_clean_cli_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["plan", "--model", "not-a-model"])
    assert res.exit_code == 1 and "rigma update" in res.output
    res = runner.invoke(cli.app, ["up", "--model", "not-a-model", "--dry-run"])
    assert res.exit_code == 1 and "rigma update" in res.output


def _running_state(tmp_path, monkeypatch):
    import os
    from rigma import state as st
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())


def test_bench_reports_a_dead_engine_without_a_traceback(tmp_path, monkeypatch):
    """AUDIT F08-8: a refused connection escaped as a raw httpx traceback."""
    import httpx
    _running_state(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    res = runner.invoke(cli.app, ["bench"])
    assert res.exit_code == 1 and "benchmark failed" in res.output


def test_bench_reports_a_no_timings_engine_cleanly(tmp_path, monkeypatch):
    """The RuntimeError 08-7 raises must not become a new traceback either."""
    import httpx
    _running_state(tmp_path, monkeypatch)

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    res = runner.invoke(cli.app, ["bench"])
    assert res.exit_code == 1 and "benchmark failed" in res.output
    assert "timings" in res.output


def _argv_line(output):
    return next(line for line in output.splitlines()
                if line.startswith("argv: "))


def test_dry_run_preview_shows_every_launch_flag(tmp_path, monkeypatch):
    """AUDIT F15-3: the preview printed only `plan.server_args`, so it omitted
    the projector, a repaired chat template and the always-present KV slot dir."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    tmpl = tmp_path / "templates"
    tmpl.mkdir(parents=True)
    (tmpl / "qwen3-vl-8b.jinja").write_text("{{ x }}", encoding="utf-8")
    res = runner.invoke(cli.app, ["up", "--model", "qwen3-vl-8b", "--dry-run"])
    assert res.exit_code == 0
    line = _argv_line(res.output)
    assert "--mmproj" in line
    assert "--chat-template-file" in line
    assert "--slot-save-path" in line


def test_dry_run_preview_comes_from_the_shared_argv_helper(tmp_path, monkeypatch):
    """The printed line must be `_launch_argv`'s output, not a hand-built
    string, so the preview cannot drift from the launch again."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    monkeypatch.setattr(cli, "_launch_argv",
                        lambda cand, reg, port, model_label="<model>":
                        ["llama-server", "--sentinel"])
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--dry-run"])
    assert res.exit_code == 0
    assert _argv_line(res.output) == "argv: llama-server --sentinel"


def test_server_argv_appends_the_slot_save_path():
    """`runtime.server_argv` is the one place the launch argv is assembled."""
    from rigma import runtime
    from rigma.models import ComboFlags, GgufFile, RunPlan
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192),
                   origin="calculator")
    argv = runtime.server_argv("llama-server", plan, "<model>", 11499)
    assert argv[0] == "llama-server"
    assert argv[-2:] == ["--slot-save-path",
                         str(runtime.rigma_home() / "sessions")]


def test_up_failure_message_names_the_real_log_dir(tmp_path, monkeypatch):
    """AUDIT F15-7: `~/.rigma/logs/` is not expanded by Explorer or cmd, so the
    one message that says where a failed launch logged pointed nowhere."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    from rigma import runtime

    def boom(*a, **k):
        raise RuntimeError("engine download failed")

    monkeypatch.setattr(runtime, "ensure_engine", boom)
    res = runner.invoke(cli.app, ["up", "--model", "qwen3.6-35b-a3b",
                                  "--use-case", "coding", "--yes"])
    assert res.exit_code == 1
    assert str(tmp_path / "logs") in res.output
    assert "~/.rigma" not in res.output


def test_bench_calibration_message_names_the_real_file(tmp_path, monkeypatch):
    """AUDIT F15-7: same shorthand in the calibration confirmation."""
    import httpx
    _running_state(tmp_path, monkeypatch)

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"timings": {"prompt_per_second": 650.0,
                                "predicted_per_second": 55.5}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    res = runner.invoke(cli.app, ["bench"])
    assert res.exit_code == 0
    assert str(tmp_path / "calibration.json") in res.output
    assert "~/.rigma" not in res.output
