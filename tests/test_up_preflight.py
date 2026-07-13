import socket

from typer.testing import CliRunner

import rigma.cli as cli

runner = CliRunner()
RAW = [{"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT", "vram_mb": 16368}]


def _fake_probe(gpu_table, raw_gpus=None):
    from rigma.probe import probe_hardware
    return probe_hardware(gpu_table, raw_gpus=RAW)


def test_version_flag():
    import rigma
    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0 and rigma.__version__ in res.output


def test_up_fails_fast_when_port_taken(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "probe_hardware", _fake_probe)
    from rigma import runtime
    called = []
    monkeypatch.setattr(runtime, "ensure_engine",
                        lambda *a, **k: called.append(1))
    monkeypatch.setattr(runtime, "ensure_model",
                        lambda *a, **k: called.append(1))
    monkeypatch.setattr(runtime, "launch_server",
                        lambda *a, **k: called.append(1))
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 11596))
    blocker.listen(1)
    try:
        res = runner.invoke(cli.app, ["up", "--use-case", "coding",
                                      "--yes", "--port", "11596"])
    finally:
        blocker.close()
    assert res.exit_code == 1
    assert "in use" in res.output.lower()
    assert not called  # failed BEFORE any download/model load
