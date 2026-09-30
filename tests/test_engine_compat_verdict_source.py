"""A5: `_engine_compat_note` must follow the ENGINE's verdict and name its source.

The engine's own refusal is authoritative: a registered fork may load a file the
model-side heuristic calls unloadable. When the engine cannot be asked at all, the
model-side heuristic is the only evidence left — and the note must admit that,
instead of reusing the model-side guess as if the engine had said it.

The refuse gate (`--verify --refuse`) must act on the ENGINE's refusal only. A
model-side note is shown but never blocks, because the model-side heuristic is known
to be wrong for fork quant types and an engine that cannot be asked (a fork build with
no `llama-fit-params`) is exactly that case.
"""
from __future__ import annotations

import types

import pytest
import typer

from rigma import cli, engine_compat, hangar, memtruth

BONSAI = {0: 1, 30: 1, 142: 100}   # fork type 142 (PQ2_0)


class _Idx:
    def __init__(self, counts, complete=True):
        self.type_counts = counts
        self.types_complete = complete


def _model_side_says_cannot_load(monkeypatch):
    monkeypatch.setattr(engine_compat, "check_gguf",
                        lambda p: engine_compat.check_types(BONSAI))


def test_engine_loadable_overrides_the_model_side_verdict(monkeypatch):
    """The engine ran and did NOT refuse the file, so it loads — authoritative, even
    though the model-side heuristic flags fork type 142."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: None)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_engine_unavailable_uses_the_model_side_check_and_says_so(monkeypatch):
    """No oracle / failed probe: the model-side heuristic is all there is, and the
    note must name it as the source rather than implying the engine confirmed it. It
    must also be marked so the refuse gate does NOT act on it."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count",
                        lambda e, m: cli._ENGINE_UNASKED)
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert "model-side" in note
    assert "could not be asked" in note
    assert "UNVERIFIED" in note
    assert cli._compat_note_refuses(note) is False


def test_engine_refusal_names_the_engine_as_the_source(monkeypatch):
    """The engine refused, so the note is the engine's verdict and says so."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    import rigma.gguf_meta as gm
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(BONSAI))
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert "CANNOT LOAD" in note
    assert "source:" in note
    assert "type-table bound" in note
    assert cli._compat_note_refuses(note) is True


def test_a_failed_engine_explanation_still_refuses_without_the_model_side_reason(
        monkeypatch):
    """A5's stale-`c` bug: the engine refused, but the histogram re-read failed, so the
    old code printed the model-side `check_gguf` result under the engine's banner.

    The REFUSAL must stand (base behaviour: it exited 1), but the reason must be the
    engine's own, never the model-side guess — which here would name PQ2_0."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    import rigma.gguf_meta as gm

    def boom(p):
        raise OSError("histogram read failed")

    monkeypatch.setattr(gm, "read_tensor_index", boom)
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert cli._compat_note_refuses(note) is True
    assert "type table ends at 42" in note
    assert "PQ2_0" not in note


def _rp(tmp_path):
    return types.SimpleNamespace(
        backend="rocm", gguf=types.SimpleNamespace(file="m.gguf"))


def _stub_verify_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_engine_server_exe",
                        lambda rp: tmp_path / "llama-server.exe")
    monkeypatch.setattr(hangar, "models_dir", lambda: tmp_path)
    (tmp_path / "m.gguf").write_bytes(b"x")
    stub = types.SimpleNamespace(primary=None, reason="stubbed", ok=False)
    monkeypatch.setattr(memtruth, "verify_plan", lambda *a, **k: (stub, None))


def test_refuse_does_not_fire_on_a_model_side_note(monkeypatch, tmp_path, capsys):
    """Direction (a): engine unaskable + model-side cannot-load must NOT exit 1 (base
    printed nothing and exited 0). The note is still shown, marked unverified."""
    _stub_verify_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_engine_compat_note",
                        lambda *a: cli._MODEL_SIDE_BANNER + "\n         model-side")
    cli._verify_plan_or_explain(_rp(tmp_path), refuse=True)   # must not raise
    assert "UNVERIFIED" in capsys.readouterr().out


def test_refuse_fires_on_the_engines_own_refusal(monkeypatch, tmp_path):
    """Direction (b): an engine refusal must still exit 1 even when the explanation
    could not be reconstructed."""
    _stub_verify_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "_engine_compat_note",
        lambda *a: cli._ENGINE_REFUSAL_BANNER + "\n         the engine refused it")
    with pytest.raises(typer.Exit):
        cli._verify_plan_or_explain(_rp(tmp_path), refuse=True)
