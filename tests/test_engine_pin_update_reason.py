"""A6: `rigma update` must not call every engine-pin failure "already current".

`update_engines_manifest` returned a bare False for offline, an HTTP error and a
corrupt payload alike, so the CLI told an offline user their pin was up to date.
The result now distinguishes the cases, and each prints a different line.
"""
from __future__ import annotations

import json

import httpx

from rigma import cli, runtime


def _manifest(version: str) -> dict:
    packaged = runtime._engines_manifest()
    return json.loads(json.dumps(packaged).replace(packaged["version"], version))


def _response(payload=None, *, content: bytes | None = None) -> httpx.Response:
    kw = {"content": content} if content is not None else {"json": payload}
    return httpx.Response(200, request=httpx.Request("GET", "https://x/"), **kw)


def _result(status: str, version: str = "") -> runtime.ManifestUpdate:
    return runtime.ManifestUpdate(status, version)


# --- the CLI line -----------------------------------------------------------

def test_up_to_date_says_already_current():
    line = cli._engine_pin_line("b9867", _result(runtime.MANIFEST_CURRENT, "b9867"))
    assert line == "engine pin: b9867 (already current)"


def test_a_network_failure_says_so():
    line = cli._engine_pin_line("b9867", _result(runtime.MANIFEST_NETWORK))
    assert "could not reach the network" in line
    assert "already current" not in line


def test_a_corrupt_manifest_says_it_was_unusable():
    line = cli._engine_pin_line("b9867", _result(runtime.MANIFEST_UNUSABLE))
    assert "unusable" in line
    assert "could not reach the network" not in line


def test_the_three_cases_print_three_different_lines():
    lines = {
        cli._engine_pin_line("b9867", _result(runtime.MANIFEST_CURRENT, "b9867")),
        cli._engine_pin_line("b9867", _result(runtime.MANIFEST_NETWORK)),
        cli._engine_pin_line("b9867", _result(runtime.MANIFEST_UNUSABLE)),
    }
    assert len(lines) == 3


def test_a_newer_pin_still_prints_the_arrow():
    line = cli._engine_pin_line("b9867", _result(runtime.MANIFEST_UPDATED, "b9900"))
    assert line == "engine pin: b9867 -> b9900"


# --- the result itself ------------------------------------------------------

def test_a_failed_request_is_network_not_unusable(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(runtime.httpx, "get", boom)
    assert runtime.update_engines_manifest_result().status == runtime.MANIFEST_NETWORK


def test_a_non_json_200_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: _response(content=b"<html>not a manifest"))
    assert runtime.update_engines_manifest_result().status == runtime.MANIFEST_UNUSABLE


def test_a_structurally_invalid_manifest_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: _response({"version": "b9900"}))
    assert runtime.update_engines_manifest_result().status == runtime.MANIFEST_UNUSABLE


def test_the_pin_in_use_reports_current(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: _response(runtime._engines_manifest()))
    res = runtime.update_engines_manifest_result()
    assert res.status == runtime.MANIFEST_CURRENT
    assert res.version == runtime._engines_manifest()["version"]


def test_a_newer_pin_reports_updated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("RIGMA_HOME", str(home))
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: _response(_manifest("b9900")))
    res = runtime.update_engines_manifest_result()
    assert res.status == runtime.MANIFEST_UPDATED
    assert res.version == "b9900"
    assert json.loads((home / "engines.json").read_text())["version"] == "b9900"


def test_the_boolean_door_still_works(tmp_path, monkeypatch):
    """The old bool API is kept for callers that only want yes/no."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: _response(_manifest("b9900")))
    assert runtime.update_engines_manifest() is True
    monkeypatch.setattr(runtime.httpx, "get",
                        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    assert runtime.update_engines_manifest() is False


def test_the_update_command_prints_the_reason(tmp_path, monkeypatch):
    """The wiring, not just the formatter: `rigma update` must report the result."""
    from typer.testing import CliRunner

    from rigma import registry

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(registry, "update_registry", lambda *a, **k: tmp_path / "reg")
    monkeypatch.setattr(runtime, "_engines_manifest", lambda: {"version": "b9867"})
    monkeypatch.setattr(
        runtime, "update_engines_manifest_result",
        lambda *a, **k: runtime.ManifestUpdate(runtime.MANIFEST_NETWORK))
    res = CliRunner().invoke(cli.app, ["update"])
    assert res.exit_code == 0, res.output
    assert "engine pin: b9867 (could not reach the network" in res.output

