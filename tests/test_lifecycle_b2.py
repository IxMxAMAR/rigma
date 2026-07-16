"""B2 lifecycle: orphan recovery, unloaded survival, port wait, list/rm."""
import os
import struct

import pytest
from typer.testing import CliRunner

import rigma.cli as cli
from rigma import hangar
from rigma import state as st


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_orphan_engine_reaped_when_ui_dead(home, monkeypatch):
    # engine 'alive', UI dead -> reap engine + clear (terminal-closed case)
    st.write_state("m", "Q4", 11500, engine_pid=4242, ui_pid=999999999)
    killed = []
    monkeypatch.setattr(st, "kill_pid", lambda pid: killed.append(pid))
    monkeypatch.setattr(st, "pid_alive", lambda pid: pid == 4242)
    assert st.server_running() is None
    assert killed == [4242]
    assert st.read_state() is None


def test_unloaded_survives_dead_engine_if_ui_alive(home):
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=os.getpid(),
                   unloaded=True)
    assert st.server_running() is not None


def test_await_port_free_returns_when_free(home):
    from rigma.server_ops import _await_port_free
    _await_port_free(0, tries=1)   # port 0 is always bindable


def _tiny_gguf(path):
    def _s(b):
        return struct.pack("<Q", len(b)) + b
    kvs = [_s(b"general.architecture") + struct.pack("<I", 8) + _s(b"qwen3"),
           _s(b"general.name") + struct.pack("<I", 8) + _s(b"Tiny Tune 1B"),
           _s(b"qwen3.block_count") + struct.pack("<II", 4, 4),
           _s(b"qwen3.context_length") + struct.pack("<II", 4, 8192),
           _s(b"qwen3.embedding_length") + struct.pack("<II", 4, 512),
           _s(b"qwen3.attention.head_count") + struct.pack("<II", 4, 8),
           _s(b"qwen3.attention.head_count_kv") + struct.pack("<II", 4, 2)]
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", len(kvs)) + b"".join(kvs)
                     + b"\x00" * 64)
    return path


def test_list_and_rm_roundtrip(home, tmp_path):
    hangar.install_model(_tiny_gguf(tmp_path / "Tiny-Q4_K_M.gguf"))
    runner = CliRunner()
    r = runner.invoke(cli.app, ["list"])
    assert r.exit_code == 0 and "tiny-tune-1b" in r.output
    r = runner.invoke(cli.app, ["rm", "tiny-tune-1b", "-y"])
    assert r.exit_code == 0 and "deleted" in r.output
    assert "tiny-tune-1b" not in runner.invoke(cli.app, ["list"]).output


def test_rm_refuses_running_model(home, tmp_path):
    hangar.install_model(_tiny_gguf(tmp_path / "Tiny-Q4_K_M.gguf"))
    st.write_state("tiny-tune-1b", "Q4_K_M", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    r = CliRunner().invoke(cli.app, ["rm", "tiny-tune-1b", "-y"])
    assert r.exit_code == 1 and "running" in r.output


def test_rag_base_url_follows_state_port(home):
    from rigma import rag
    st.write_state("m", "Q4", 8080, engine_pid=os.getpid(), ui_pid=os.getpid())
    assert rag._llm_base_url() == "http://127.0.0.1:8080/v1"
