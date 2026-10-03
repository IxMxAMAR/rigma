"""raggity is a Python child, and on Windows a Python child writing to a pipe
uses a strict cp1252 unless told otherwise: a file name outside it crashed the
ingest, and its UTF-8 output was mis-read. These run a REAL child — the other RAG
tests stub `subprocess.run`, which is why the encoding was never exercised."""
import json
import sys

from rigma import rag

NAME = "notes/\u015dkola \u4e2d.md"      # "ŝ" and CJK are not in cp1252


def _fake_raggity(tmp_path, monkeypatch, body):
    script = tmp_path / "fake_raggity.py"
    script.write_text(body, encoding="utf-8")
    monkeypatch.setattr(rag, "raggity_cmd", lambda: [sys.executable, str(script)])


def test_ingest_survives_a_file_name_outside_cp1252(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    _fake_raggity(tmp_path, monkeypatch, f"print('indexed: ' + {NAME!r})\n")
    assert NAME in rag.ingest()


def test_discover_reads_a_candidate_outside_cp1252(tmp_path, monkeypatch):
    payload = json.dumps({"complete": True,
                          "candidates": [{"path": "C:/" + NAME, "why": "vault"}]},
                         ensure_ascii=False)
    _fake_raggity(tmp_path, monkeypatch, f"print({payload!r})\n")
    got = rag.discover()
    assert got["available"] is True
    assert got["candidates"][0]["path"] == "C:/" + NAME
