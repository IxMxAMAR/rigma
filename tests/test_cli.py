from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()
RAW = [{"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT", "vram_mb": 16368}]


def _fake_probe(gpu_table, raw_gpus=None):
    from rigma.probe import probe_hardware
    return probe_hardware(gpu_table, raw_gpus=RAW)


def test_doctor(monkeypatch):
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["doctor"])
    assert res.exit_code == 0 and "rx-9070-xt" in res.output.lower()


def test_plan_explain(monkeypatch):
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["plan", "--use-case", "coding", "--explain"])
    assert res.exit_code == 0
    assert "UD-Q3_K_XL" in res.output and "combo:" in res.output


def test_status_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0 and "not running" in res.output.lower()


def test_stop_when_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["stop"])
    assert res.exit_code == 0 and "not running" in res.output.lower()


def test_up_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    res = runner.invoke(cli.app, ["up", "--use-case", "coding", "--dry-run"])
    assert res.exit_code == 0
    assert "--n-cpu-moe 10" in res.output and "-fa on" in res.output


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
