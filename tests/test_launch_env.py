import subprocess

from rigma import runtime
from rigma.models import ComboFlags, GgufFile, RunPlan


class _Boom(RuntimeError):
    pass


def _plan(env):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan",
                   flags=ComboFlags(ctx=4096, env=env), origin="calculator")


def test_launch_merges_flag_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    captured = {}

    def fake_popen(argv, **kw):
        captured["env"] = kw.get("env")
        raise _Boom()  # stop before the health loop

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    try:
        runtime.launch_server(tmp_path / "srv.exe",
                              _plan({"GGML_VK_DISABLE_COOPMAT": "1"}),
                              tmp_path / "m.gguf", 11601)
    except _Boom:
        pass
    assert captured["env"]["GGML_VK_DISABLE_COOPMAT"] == "1"
    assert "PATH" in captured["env"]  # inherited base env preserved


def test_launch_no_env_field_leaves_popen_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    captured = {}

    def fake_popen(argv, **kw):
        captured["has_env"] = "env" in kw
        raise _Boom()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    try:
        runtime.launch_server(tmp_path / "srv.exe", _plan({}),
                              tmp_path / "m.gguf", 11601)
    except _Boom:
        pass
    assert captured["has_env"] is False  # empty env => inherit, don't pass env=
