"""R3-ORPHAN: an engine whose UI died, and the two commands that recover.

The failure this covers is not "the port is busy". It is that the port is held by
**Rigma's own engine** while `state.json` — the only thing that could stop it — is
gone, so `rigma up` refused with advice the user could not act on ("free it") and
`rigma stop` answered "not running" while a model sat in VRAM.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from rigma import cli, orphan

PROPS = {
    "model_path": r"C:\Users\amren\.rigma\models\Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "build_info": "b10709-9a9394a89",
    "default_generation_settings": {"n_ctx": 65536, "params": {}},
}
EXE = r"C:\Users\amren\.rigma\engines\b9867\rocm\llama-server.exe"

# The holder is on the ENGINE port (11500 - 1), which is the only port that can
# carry an engine. A mock that answers for the UI port instead would agree with
# the bug this file exists to catch — it did, in the first version.
ENGINE_PORT = 11499


def _fake_holder(p):
    return " (held by pid 4242: llama-server.exe)" if p == ENGINE_PORT else ""


def _fake_pid(p):
    return 4242 if p == ENGINE_PORT else None



# --------------------------------------------------------------------------
# what the engine says about itself


def test_the_record_is_built_from_the_engine_not_from_memory():
    rec = orphan.record_from_props(PROPS, 16908, EXE, 11500)
    assert rec["engine_pid"] == 16908
    assert rec["public_port"] == 11500
    assert rec["ctx"] == 65536
    assert rec["gguf"] == "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    assert rec["model_slug"] == "Ternary-Bonsai-2-27B-PQ2_0"
    assert rec["engine"] == "llamacpp"


def test_the_backend_is_read_from_the_binary_path():
    """`rocm` is the directory Rigma itself creates, and the only place the
    running build recorded which backend it is."""
    assert orphan._backend_from_exe(EXE) == "rocm"
    assert orphan._backend_from_exe(
        r"C:\x\.rigma\engines\b9867\vulkan\llama-server.exe") == "vulkan"
    # a layout Rigma did not create: empty, not a guess
    assert orphan._backend_from_exe(r"D:\build\llama-server.exe") == ""


def test_a_field_the_engine_does_not_report_is_left_empty():
    """`quant` is a derived label that moves as a repo gains siblings, so an
    empty one is better than a wrong one."""
    rec = orphan.record_from_props({}, 1, "", 11500)
    assert rec["quant"] == ""
    assert rec["ctx"] == 0
    assert rec["gguf"] == ""


def test_describe_names_the_model_and_the_window():
    s = orphan.describe(11499, 16908, PROPS)
    assert "Ternary-Bonsai-2-27B-PQ2_0.gguf" in s
    assert "65536" in s
    assert "16908" in s


def test_a_foreign_binary_is_not_rigmas_engine():
    """The check that keeps adoption honest: a llama-server the USER built is
    not Rigma's to adopt, and a name check alone would claim it."""
    import os

    assert not orphan.is_rigma_engine(os.getpid())
    assert not orphan.is_rigma_engine(0)
    assert not orphan.is_rigma_engine(-1)


# --------------------------------------------------------------------------
# the refusal, and the way out


class _FakeEngine(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(PROPS if self.path == "/props" else {"status": "ok"})
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_engine():
    srv = HTTPServer(("127.0.0.1", 0), _FakeEngine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    (tmp_path / "rigma-home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "rigma-home"


def test_props_are_read_from_a_real_http_engine(fake_engine):
    got = orphan.engine_props(fake_engine)
    assert got and got["default_generation_settings"]["n_ctx"] == 65536


def test_props_returns_none_for_something_that_is_not_an_engine():
    """Absence is the answer to "can we adopt this", not an error."""
    assert orphan.engine_props(1) is None          # nothing there
    assert orphan.engine_props(0) is None


def test_without_reattach_the_refusal_names_rigmas_own_engine(home, monkeypatch):
    """The whole point of the finding: the old sentence said "free it or pass a
    different --port" about a process that is Rigma's, holding the user's model.
    """
    monkeypatch.setattr(cli, "_port_holder", _fake_holder)
    monkeypatch.setattr(cli, "_listening_pid", _fake_pid)
    monkeypatch.setattr(orphan, "is_rigma_engine", lambda pid: True)
    with pytest.raises(cli.typer.Exit) as e:
        cli._adopt_or_refuse(11500, reattach=False, dry_run=False)
    assert e.value.exit_code == 1


def test_the_refusal_reaches_the_user_with_the_remedy(home, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_port_holder", _fake_holder)
    monkeypatch.setattr(cli, "_listening_pid", _fake_pid)
    monkeypatch.setattr(orphan, "is_rigma_engine", lambda pid: True)
    with pytest.raises(cli.typer.Exit):
        cli._adopt_or_refuse(11500, reattach=False, dry_run=False)
    out = capsys.readouterr().out
    assert "RIGMA'S OWN engine" in out, out
    assert "rigma up --reattach" in out, out
    assert "rigma stop" in out, out


def test_a_foreign_holder_still_gets_the_old_sentence(home, monkeypatch, capsys):
    """A genuinely foreign process must not be described as Rigma's."""
    monkeypatch.setattr(cli, "_port_holder", lambda p: " (held by pid 9: chrome.exe)")
    monkeypatch.setattr(cli, "_listening_pid", lambda p: 9)
    monkeypatch.setattr(orphan, "is_rigma_engine", lambda pid: False)
    with pytest.raises(cli.typer.Exit):
        cli._adopt_or_refuse(11500, reattach=False, dry_run=False)
    out = capsys.readouterr().out
    assert "free it or pass a different --port" in out, out
    assert "RIGMA'S OWN" not in out, out


def test_reattach_writes_a_record_the_ui_can_stop(home, monkeypatch, fake_engine,
                                                  capsys):
    """The recovered engine must be stoppable. `_adopt_or_refuse` writes a record
    naming the pid, which is what `up`'s finally block reads to stop it — without
    that, adopting would leave the same orphan behind a working UI."""
    from rigma import state as st

    monkeypatch.setattr(cli, "_port_holder", _fake_holder)
    monkeypatch.setattr(cli, "_listening_pid", _fake_pid)
    monkeypatch.setattr(orphan, "is_rigma_engine", lambda pid: True)
    monkeypatch.setattr(orphan, "engine_props", lambda port, timeout=5.0: PROPS)
    cli._adopt_or_refuse(11500, reattach=True, dry_run=False)
    rec = st.read_state()
    assert rec is not None, "reattach must leave a record behind"
    assert rec["engine_pid"] == 4242
    assert rec["ctx"] == 65536
    assert rec["gguf"] == "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    assert "reattached" in capsys.readouterr().out


def test_reattach_refuses_an_engine_that_does_not_answer_health(home, monkeypatch,
                                                               capsys):
    """Rigma's binary, but not answering: adopting it would produce a UI pointed
    at a dead engine, which is worse than the refusal."""
    monkeypatch.setattr(cli, "_port_holder", _fake_holder)
    monkeypatch.setattr(cli, "_listening_pid", _fake_pid)
    monkeypatch.setattr(orphan, "is_rigma_engine", lambda pid: True)
    monkeypatch.setattr(orphan, "engine_props", lambda port, timeout=5.0: None)
    with pytest.raises(cli.typer.Exit) as e:
        cli._adopt_or_refuse(11500, reattach=True, dry_run=False)
    assert e.value.exit_code == 1
    assert "not answering /health" in capsys.readouterr().out


def test_a_free_port_is_left_alone(home, monkeypatch):
    monkeypatch.setattr(cli, "_port_holder", lambda p: "")
    monkeypatch.setattr(cli, "_listening_pid", lambda p: None)
    cli._adopt_or_refuse(11500, reattach=True, dry_run=False)   # must not raise


# --------------------------------------------------------------------------
# stop: the absent record was not the whole truth


def test_stop_finds_the_orphan_before_saying_not_running(home, monkeypatch,
                                                         capsys):
    from rigma import state as st

    assert st.read_state() is None
    monkeypatch.setattr(orphan, "find_engines", lambda port: [(4242, PROPS)])
    monkeypatch.setattr(orphan, "record_from_props",
                        lambda *a, **k: {"model_slug": "m", "quant": "",
                                         "backend": "rocm", "ctx": 65536,
                                         "gguf": "m.gguf", "engine": "llamacpp"})
    monkeypatch.setattr(st, "kill_recorded", lambda s, key: key == "engine_pid")
    result = CliRunner().invoke(cli.app, ["stop"])
    out = result.output
    assert "not running" not in out, out
    assert "stopped" in out, out


def test_stop_still_says_not_running_when_there_is_nothing(home, monkeypatch):
    monkeypatch.setattr(orphan, "find_engines", lambda port: [])
    result = CliRunner().invoke(cli.app, ["stop"])
    assert "not running" in result.output


def test_stop_refuses_to_guess_between_two_engines(home, monkeypatch):
    """Two healthy Rigma engines on one port is not something to adopt."""
    monkeypatch.setattr(orphan, "find_engines",
                        lambda port: [(1, PROPS), (2, PROPS)])
    result = CliRunner().invoke(cli.app, ["stop"])
    assert result.exit_code == 1
    assert "stopping none of them" in result.output
