"""R3-ORPHAN: an engine whose UI died, and the two commands that recover.

The failure this covers is not "the port is busy". It is that the port is held by
**Rigma's own engine** while `state.json` — the only thing that could stop it — is
gone, so `rigma up` refused with advice the user could not act on ("free it") and
`rigma stop` answered "not running" while a model sat in VRAM.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from rigma import cli, orphan

# The engine reports the path it was LAUNCHED with, in the running platform's own
# spelling, so the fixture must be spelled that way too. A Windows-spelled path
# is not a path on POSIX: `Path(r"C:\models\x.gguf").name` is the WHOLE string
# there, because `\` is an ordinary filename character — so a hardcoded `C:\...`
# fixture silently turns "take the basename" into "return the input" and stops
# testing anything on Linux. That is what CI caught.
_HOME = r"C:\Users\dev\.rigma" if os.name == "nt" else "/home/dev/.rigma"
# The pin's binary is named for the platform: `runtime.ensure_engine` writes
# `llama-server.exe` on Windows and `llama-server` everywhere else
# (runtime.py:301), and `server_ops.engine_version` looks for exactly that name
# (server_ops.py:54).
_SERVER = "llama-server.exe" if os.name == "nt" else "llama-server"
# Deliberately NOT a registry model: this file pins the STEM fallback.
MODEL_FILE = "My-Model-Q4_K_M.gguf"
MODEL_SLUG = "My-Model-Q4_K_M"


def _native(*parts) -> str:
    """`parts` under Rigma's home, in the running platform's spelling."""
    return str(Path(_HOME, *parts))


PROPS = {
    "model_path": _native("models", MODEL_FILE),
    "build_info": "b10709-9a9394a89",
    "default_generation_settings": {"n_ctx": 65536, "params": {}},
}
EXE = _native("engines", "b9867", "rocm", _SERVER)

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
    assert rec["gguf"] == MODEL_FILE
    assert rec["model_slug"] == MODEL_SLUG
    assert rec["engine"] == "llamacpp"


def test_the_backend_is_read_from_the_binary_path():
    """`rocm` is the directory Rigma itself creates, and the only place the
    running build recorded which backend it is."""
    assert orphan._backend_from_exe(EXE) == "rocm"
    assert orphan._backend_from_exe(
        _native("engines", "b9867", "vulkan", _SERVER)) == "vulkan"
    # a layout Rigma did not create: empty, not a guess
    foreign = Path(r"D:\build" if os.name == "nt" else "/build", _SERVER)
    assert orphan._backend_from_exe(str(foreign)) == ""


def test_a_field_the_engine_does_not_report_is_left_empty():
    """`quant` is a derived label that moves as a repo gains siblings, so an
    empty one is better than a wrong one."""
    rec = orphan.record_from_props({}, 1, "", 11500)
    assert rec["quant"] == ""
    assert rec["ctx"] == 0
    assert rec["gguf"] == ""


def test_describe_names_the_model_and_the_window():
    s = orphan.describe(11499, 16908, PROPS)
    assert MODEL_FILE in s
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
    assert rec["gguf"] == MODEL_FILE
    # kv_fp is empty ON PURPOSE and must be present as empty. It hashes the
    # launch fields in kvcache.FINGERPRINT_FIELDS; the launch that computed it
    # is gone and /props reports only the window, so any value here would be a
    # different hash than the original recorded — and a cache restored under a
    # mismatched fingerprint generates fluent text from a history that never
    # happened. Empty disables save and restore. The suite's own guard
    # (test_launch_records_fingerprint) is what caught this being absent rather
    # than empty, which is why it is asserted here: absent and empty read the
    # same downstream, and only one is a decision.
    assert rec["kv_fp"] == ""
    assert "prompt caching is OFF" in capsys.readouterr().out


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


# --------------------------------------------------------------------------
# R3-CTX-1 — the record the recovery writes must be one the registry resolves


def _a_model_whose_stem_differs_from_its_slug():
    """(slug, gguf) for a model where the file stem is NOT the registry slug.

    The bug only shows up on those, and which models they are depends on the
    registry in play — so this finds one rather than naming the owner's.
    """
    from pathlib import Path

    from rigma import registry as R

    for slug, spec in R.Registry.load().models.items():
        for g in (spec.ggufs or []):
            if Path(g.file).stem != slug:
                return slug, g
    return None


def test_the_record_names_the_registry_slug_not_the_file_stem():
    """The difference is not cosmetic and it is not small.

    `server_ops._free_current` looks the slug up to credit the outgoing engine's
    VRAM back to the budget, and does NOTHING when the lookup fails — so a file
    stem written where a slug belongs makes the card look full of someone else's
    memory, and the next context change is refused while the engine is running
    fine. Measured on the owner's machine: usable VRAM 6,254 MiB (refused at
    8,192) versus 13,726 MiB (accepted at 65,536), differing only in this string.
    """
    found = _a_model_whose_stem_differs_from_its_slug()
    if found is None:
        pytest.skip("this registry has no model whose stem differs from its slug")
    slug, g = found
    rec = orphan.record_from_props(
        {"model_path": "C:/models/" + g.file,
         "default_generation_settings": {"n_ctx": 8192}}, 1, "", 11500)
    assert rec["model_slug"] == slug, (
        "the record must carry the registry slug; a stem makes _free_current a "
        "no-op and the next ctx change is refused")
    assert rec["quant"] == g.quant

def test_a_file_outside_the_registry_falls_back_to_the_stem(monkeypatch):
    """A custom install is not in the registry, and a label is better than
    nothing — but it must not be mistaken for an identity."""
    custom = _native("models", MODEL_FILE)
    monkeypatch.setattr(orphan, "engine_props", lambda port, timeout=5.0: {
        "model_path": custom,
        "default_generation_settings": {"n_ctx": 4096}})
    rec = orphan.record_from_props(
        {"model_path": custom,
         "default_generation_settings": {"n_ctx": 4096}}, 1, "", 11500)
    assert rec["model_slug"] == MODEL_SLUG
    assert rec["gguf"] == MODEL_FILE


def test_free_current_asks_the_engine_when_the_record_cannot_say(monkeypatch):
    """`_free_current` credited nothing when the record was empty, which is
    exactly the state a hard kill leaves behind. It must fall back to what the
    engine reports, and it must do so BEFORE the `not state` early return — the
    first version of the fallback sat after it and credited nothing. The repro
    caught that, not the reasoning.

    Built on a SYNTHETIC registry so it does not depend on which models this
    machine happens to have, and on the REAL `HardwareProfile` because
    `_free_current` calls `profile.model_copy(...)`. A `SimpleNamespace` stub
    failed on exactly that, which is the same trap as the port mocks earlier in
    this file: a stub that does not behave like the thing it replaces cannot
    fail on the bug it is meant to catch.

    The GPU is STATED rather than probed. `probe_hardware` finds no GPU on a
    headless CI runner, and `_budgets` then answers (0, ...) for BOTH profiles —
    so the budget assertion below failed `0 > 0` on ubuntu while the crediting
    under test was perfectly correct. What this test measures is `_free_current`,
    not the machine CI happens to run on.
    """
    from types import SimpleNamespace

    from rigma import probe, registry as R, server_ops
    from rigma.models import GpuInfo
    from rigma.resolve import _budgets

    gguf = SimpleNamespace(file="Tiny-Model-Q4_K_M.gguf", bytes=4 * 2**30,
                           quant="Q4_K_M")
    spec = SimpleNamespace(ggufs=[gguf], mmproj=None)
    reg = SimpleNamespace(models={"tiny-model": spec})
    gpu = GpuInfo(vendor="amd", name="Test GPU 16G", vram_mb=16368,
                  arch="rdna4", slug="test-gpu-16g", backends=["vulkan"])
    base = probe.probe_hardware(R.Registry.load().gpus)
    prof = base.model_copy(update={"ram_free_mb": 4000,
                                   "vram_used_mb": 9900.0, "gpus": [gpu]})

    monkeypatch.setattr(orphan, "running_gguf_file",
                        lambda ui_port: "Tiny-Model-Q4_K_M.gguf")
    credited = server_ops._free_current(prof, {"public_port": 11500}, reg)
    assert credited.ram_free_mb > prof.ram_free_mb, (
        "the outgoing engine's RAM was not credited from an empty record")
    assert credited.vram_used_mb < prof.vram_used_mb, (
        "the outgoing engine's VRAM was not credited back to the budget")
    assert _budgets(credited)[0] > _budgets(prof)[0]

    # and with no engine answering, nothing is credited
    monkeypatch.setattr(orphan, "running_gguf_file", lambda ui_port: "")
    plain = server_ops._free_current(prof, {"public_port": 11500}, reg)
    assert plain.vram_used_mb == prof.vram_used_mb

def test_adopt_writes_a_record_the_registry_can_resolve(home, monkeypatch):
    """The end the user sees: `rigma up` no longer says "no model loaded" while
    a model is answering /health."""
    from rigma import state as st

    monkeypatch.setattr(orphan, "find_engines",
                        lambda port: [(4242, PROPS)])
    rec = orphan.adopt(11500)
    assert rec is not None
    s = st.read_state()
    assert s["model"] == rec["model_slug"]
    assert s["unloaded"] is False, "an adopted engine is loaded, not unloaded"
    assert s["engine_pid"] == 4242
    assert s["gguf"] == MODEL_FILE
    assert s["kv_fp"] == ""


def test_adopt_declines_when_there_is_nothing_or_too_much(home, monkeypatch):
    from rigma import state as st

    monkeypatch.setattr(orphan, "find_engines", lambda port: [])
    assert orphan.adopt(11500) is None
    monkeypatch.setattr(orphan, "find_engines",
                        lambda port: [(1, PROPS), (2, PROPS)])
    assert orphan.adopt(11500) is None, "two engines is not something to adopt"
    assert st.read_state() is None, "declining must not write a record"


def test_reattach_with_nothing_to_adopt_is_an_error(home, monkeypatch, capsys):
    """Asking to reattach to nothing must fail, not start a UI with no model.

    That quiet start is what produces "no model loaded" and then a harness turn
    dying with the adapter's own "at least one --model <id> is required" — a
    failure two screens away from its cause.
    """
    monkeypatch.setattr(cli, "_port_holder", lambda p: "")
    monkeypatch.setattr(cli, "_listening_pid", lambda p: None)
    monkeypatch.setattr(orphan, "find_engines", lambda port: [])
    result = CliRunner().invoke(cli.app, ["up", "--reattach", "--port", "11700"])
    assert result.exit_code == 1, result.output
    assert "nothing to reattach to" in result.output, result.output


def test_an_empty_model_is_refused_before_the_harness_is_spawned(home, monkeypatch):
    """An external harness is handed `model=str(state.get("model") or "")`. With
    no model loaded that reaches mcode as "at least one --model <id> is required"
    and DSH as a runner exiting 1 with an empty stderr tail. Both are true;
    neither names the cause. The turn must say the cause instead of spawning.
    """
    from rigma import serve as _serve

    spawned = []

    class _Adapter:
        name = "dsh"
        label = "stub"

        def drive_turn(self, **kw):
            spawned.append(kw)
            return iter(())

    monkeypatch.setattr(_serve._harness, "resolve",
                        lambda name, **kw: _Adapter())
    monkeypatch.setattr(_serve._harness, "endpoint_for",
                        lambda port: "http://127.0.0.1:1/v1")
    app = _serve.build_app(upstream_port=1)
    c = TestClient(app)
    sid = c.post("/api/sessions", json={}).json()["id"]
    c.post(f"/api/sessions/{sid}", json={"harness": "dsh", "use_tools": True})
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200, r.text
    assert "no model is loaded" in r.text, r.text[:400]
    assert not spawned, "the adapter was spawned with no model to run against"
