"""`remember` / `recall` share `model_memory.json` between the native loop and
every arm chat's MCP server — several processes. A write in place could be read
half-done by another process (or truncated by a power cut), after which every
call raised; and two writers lost each other's facts."""
import json
import os
import subprocess
import sys
import textwrap

from rigma import tools


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path / "model_memory.json"


def test_a_damaged_store_is_reported_and_never_overwritten(tmp_path, monkeypatch):
    f = _store(tmp_path, monkeypatch)
    f.write_text('{"name": "Ada", "colour": "bl', encoding="utf-8")   # torn write

    recalled = tools._recall({}, {})
    saved = tools._remember({"key": "pet", "value": "cat"}, {})

    assert recalled.startswith("error") and "unreadable" in recalled
    assert saved.startswith("error") and "not overwritten" in saved
    assert f.read_text(encoding="utf-8") == '{"name": "Ada", "colour": "bl'


def test_remember_then_recall_round_trips(tmp_path, monkeypatch):
    f = _store(tmp_path, monkeypatch)
    assert tools._remember({"key": "name", "value": "Ada"}, {}) == "remembered 'name'"
    assert tools._recall({"key": "name"}, {}) == "Ada"
    assert json.loads(f.read_text(encoding="utf-8")) == {"name": "Ada"}


def test_concurrent_writers_in_separate_processes_lose_nothing(tmp_path, monkeypatch):
    f = _store(tmp_path, monkeypatch)
    worker = textwrap.dedent("""
        import sys
        from rigma import tools
        who = sys.argv[1]
        for i in range(25):
            out = tools._remember({"key": f"{who}-{i}", "value": "x"}, {})
            assert out.startswith("remembered"), out
    """)
    env = {**os.environ, "RIGMA_HOME": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, "-c", worker, f"w{n}"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for n in range(4)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode("utf-8", "replace")[-800:]

    mem = json.loads(f.read_text(encoding="utf-8"))
    assert len(mem) == 100, f"lost {100 - len(mem)} of 100 facts"
