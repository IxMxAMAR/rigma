from rigma import presets
from rigma.models import UseCase
from rigma.registry import Registry


def _fake_reg(**prompts):
    return Registry([], {}, {}, {k: UseCase(name=k, system_prompt=v)
                                 for k, v in prompts.items()})


def test_create_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    p = presets.create("Noir narrator", "You write noir.",
                       params={"temperature": 1.2})
    assert len(p["id"]) == 12 and p["builtin"] is False
    got = presets.load(p["id"])
    assert got["name"] == "Noir narrator"
    assert got["params"] == {"temperature": 1.2} and got["greeting"] == ""


def test_delete_and_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    p = presets.create("x", "y")
    assert presets.delete(p["id"]) is True
    assert presets.load(p["id"]) is None
    assert presets.delete(p["id"]) is False


def test_list_builtins_first_then_files_skips_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    presets.create("Zebra", "z")
    presets.create("Alpha", "a")
    (presets.presets_dir() / "junk.json").write_text("{oops", encoding="utf-8")
    out = presets.list_presets(_fake_reg(general="G", creative="C"))
    assert [p["id"] for p in out[:2]] == ["usecase:creative", "usecase:general"]
    assert all(p["builtin"] for p in out[:2])
    assert [p["name"] for p in out[2:]] == ["Alpha", "Zebra"]


def test_resolve_builtin_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    reg = _fake_reg(creative="C-PROMPT")
    b = presets.resolve("usecase:creative", reg)
    assert b["system_prompt"] == "C-PROMPT" and b["builtin"] is True
    f = presets.create("mine", "M")
    assert presets.resolve(f["id"], reg)["system_prompt"] == "M"
    assert presets.resolve("usecase:nope", reg) is None
    assert presets.resolve("", reg) is None


def test_is_builtin():
    assert presets.is_builtin("usecase:general") is True
    assert presets.is_builtin("a1b2c3d4e5f6") is False


def test_save_rejects_builtin_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    import pytest as _pytest
    with _pytest.raises(ValueError, match="read-only"):
        presets.save({"id": "usecase:general", "name": "hax"})
    assert list(presets.presets_dir().glob("*")) == []  # nothing written, no ADS orphan
