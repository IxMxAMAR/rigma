import sys
from pathlib import Path

from rigma.models import ComboFlags, GgufFile, RunPlan
from rigma.runtime import launch_server


def test_launch_health_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = RunPlan(model_slug="x",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q8_0"),
                   backend="cpu", flags=ComboFlags(ctx=8192), origin="calculator")
    fake = Path(__file__).parent / "fake_server.py"
    # exe = python; extra args make it run our fake, which ignores the plan args
    sp = launch_server(Path(sys.executable), plan, Path("model.gguf"),
                       port=11599, timeout=30, extra_args=[str(fake)])
    try:
        assert sp.is_healthy()
        assert sp.url == "http://127.0.0.1:11599"
    finally:
        sp.stop()
    assert not sp.is_healthy()
