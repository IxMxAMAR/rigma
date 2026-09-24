"""13-6: the mcode child does not inherit Rigma's whole environment.

Calls the real `_env` builder; no child process is started.
"""
import os

from rigma import harness_mcode


def test_mcode_child_does_not_inherit_rigmas_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gem_secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tv_secret")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    env = harness_mcode._env()
    assert "HF_TOKEN" not in env
    assert "GEMINI_API_KEY" not in env
    assert "TAVILY_API_KEY" not in env
    # what a CLI needs to start and find its temp dir is still there
    assert env.get("PATH")
    assert env["MINIMAX_DATA_DIR"] == str(harness_mcode.data_home())
    assert env[harness_mcode.API_KEY_ENV] == harness_mcode._API_KEY


def test_mcode_env_passthrough_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("MY_MCODE_EXTRA", "yes")
    assert "MY_MCODE_EXTRA" not in harness_mcode._env()
    monkeypatch.setenv("RIGMA_MCODE_ENV_PASSTHROUGH", "MY_MCODE_EXTRA")
    assert harness_mcode._env()["MY_MCODE_EXTRA"] == "yes"


def test_mcode_env_passthrough_supports_a_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("MY_MCODE_A", "1")
    monkeypatch.setenv("MY_MCODE_B", "2")
    monkeypatch.setenv("RIGMA_MCODE_ENV_PASSTHROUGH", "MY_MCODE_*")
    env = harness_mcode._env()
    assert env["MY_MCODE_A"] == "1" and env["MY_MCODE_B"] == "2"
