"""R3-ENG-3 — a non-stock engine must be expressible, not smuggled in.

The situation this exists for, from the owner's machine. With no way to represent a
fork, the owner installed one BY HAND into `~/.rigma/engines/b9867/rocm/` and left an
`ENGINE-PROVENANCE.txt` beside it:

    This is NOT the mainline llama.cpp ROCm build rigma pins.
      PrismML-Eng/llama.cpp, branch 'prism', release prism-b10709-9a9394a
    ...
    rigma's ENGINE_URL_ALLOWLIST is untouched; this rides the .ready short-circuit
    in runtime.ensure_engine rather than the download path.

It worked, and that is the problem: Rigma could not name it, verify it, select it, or
restore the build it displaced, and it would have mislabelled calibration forever.
"""
from __future__ import annotations

import json

from rigma import engine_registry as er

BONSAI = [0, 30, 142]        # what Ternary-Bonsai-2-27B-PQ2_0 actually uses
SMOL = [0, 8, 11, 20]        # an ordinary mainline model


def _fork(home, **kw):
    e = er.CustomEngine(
        name=kw.pop("name", "prism-rocm"),
        path=kw.pop("path", str(home / "prism" / "llama-server.exe")),
        backend=kw.pop("backend", "rocm"),
        source=kw.pop("source", "PrismML-Eng/llama.cpp branch 'prism'"),
        types=kw.pop("types", [0, 30, 142, 143]),
        types_known=kw.pop("types_known", True),
        **kw)
    er.register(e, home)
    return e


# --- the registry round-trips ------------------------------------------------

def test_a_registered_engine_survives_a_round_trip(tmp_path):
    _fork(tmp_path)
    got = er.load(tmp_path)["prism-rocm"]
    assert got.backend == "rocm"
    assert got.types_known is True
    assert 142 in got.types
    assert got.exe.name == "llama-server.exe"


def test_forgetting_an_engine_works(tmp_path):
    _fork(tmp_path)
    assert er.forget("prism-rocm", tmp_path) is True
    assert er.load(tmp_path) == {}
    assert er.forget("prism-rocm", tmp_path) is False


def test_a_corrupt_registry_reads_as_empty_rather_than_raising(tmp_path):
    """A registry that cannot be parsed must not stop Rigma starting. The worst case
    of ignoring it is that the pin is used — which is the default anyway."""
    p = er.registry_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert er.load(tmp_path) == {}


def test_a_structurally_wrong_registry_is_ignored(tmp_path):
    p = er.registry_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"engines": "nope"}), encoding="utf-8")
    assert er.load(tmp_path) == {}


def test_an_entry_with_no_path_is_skipped(tmp_path):
    """A registration with no binary is not usable, and keeping it would produce a
    selection that fails later instead of a clean 'not registered'."""
    p = er.registry_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"engines": {"broken": {"backend": "rocm"}}}),
                 encoding="utf-8")
    assert er.load(tmp_path) == {}


def test_saving_is_atomic_so_a_crash_cannot_tear_the_registry(tmp_path):
    """Same reasoning as the engine lock file: a torn registry would take out every
    later launch, and the cure would be finding and deleting it by hand."""
    _fork(tmp_path)
    _fork(tmp_path, name="second", path=str(tmp_path / "s" / "llama-server.exe"))
    assert set(er.load(tmp_path)) == {"prism-rocm", "second"}
    assert not er.registry_path(tmp_path).with_suffix(".tmp").exists()


# --- accepting and refusing, honestly ---------------------------------------

def test_an_engine_accepts_a_model_it_has_every_type_for():
    e = er.CustomEngine(name="x", path="p", types=[0, 8, 11, 20, 30, 142],
                        types_known=True)
    assert e.accepts(SMOL) is True
    assert e.accepts(BONSAI) is True


def test_an_engine_rejects_a_model_with_a_type_it_lacks():
    e = er.CustomEngine(name="x", path="p", types=[0, 8, 11], types_known=True)
    assert e.accepts(BONSAI) is False


def test_an_engine_with_no_recorded_types_is_unknown_not_a_refusal():
    """The single most important rule here. `types_known=False` means nobody recorded
    what this build supports, NOT that it supports nothing. Reporting it as a refusal
    would refuse models it can load perfectly well."""
    e = er.CustomEngine(name="x", path="p", types=[], types_known=False)
    assert e.accepts(BONSAI) is None


# --- selection ---------------------------------------------------------------

def test_selection_picks_the_engine_that_can_load_the_model(tmp_path):
    _fork(tmp_path)
    got, why = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is not None
    assert got.name == "prism-rocm"
    assert "accepts" in why


def test_selection_ignores_an_engine_for_another_backend(tmp_path):
    """A Vulkan build cannot serve a ROCm plan."""
    _fork(tmp_path, backend="vulkan")
    got, why = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is None
    assert "no engine is registered" in why


def test_an_engine_with_no_declared_backend_is_a_candidate_anywhere(tmp_path):
    """The owner installed it knowing what it was for; refusing to consider it
    because a field was left blank would be unhelpful."""
    _fork(tmp_path, backend="")
    got, _ = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is not None


def test_selection_reports_when_every_engine_rejects_the_model(tmp_path):
    _fork(tmp_path, types=[0, 8, 11], name="mainline-ish")
    got, why = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is None
    assert "rejects" in why


def test_selection_says_so_when_capabilities_are_unknown(tmp_path):
    """It must not silently pick an engine nobody can vouch for, and it must not
    claim the engine is unusable either. It says which engines are undecidable."""
    _fork(tmp_path, types=[], types_known=False)
    got, why = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is None
    assert "no recorded type table" in why
    assert "prism-rocm" in why


def test_selection_prefers_the_engine_that_knows_more_types(tmp_path):
    """A fork carrying mainline's whole table as well as its own additions is a safer
    default than one that only knows its own."""
    _fork(tmp_path, name="narrow", types=[0, 142])
    _fork(tmp_path, name="broad", path=str(tmp_path / "b" / "llama-server.exe"),
          types=[0, 8, 11, 20, 30, 142, 143])
    got, _ = er.select(BONSAI, "rocm", home=tmp_path)
    assert got.name == "broad"


def test_selection_is_deterministic_when_two_engines_tie(tmp_path):
    """Ties must not depend on dict iteration order, or the chosen engine would
    change between runs on the same machine."""
    _fork(tmp_path, name="aaa", types=[0, 30, 142])
    _fork(tmp_path, name="zzz", path=str(tmp_path / "z" / "llama-server.exe"),
          types=[0, 30, 142])
    names = {er.select(BONSAI, "rocm", home=tmp_path)[0].name for _ in range(5)}
    assert names == {"aaa"}


def test_selection_with_nothing_registered_says_so(tmp_path):
    got, why = er.select(BONSAI, "rocm", home=tmp_path)
    assert got is None
    assert "no engine is registered" in why


# --- provenance --------------------------------------------------------------

def test_restore_hint_names_the_source_so_the_owner_can_undo_it(tmp_path):
    """The owner's own note recorded what the fork was and how to restore the build
    it displaced. Registration must carry that, not just a path."""
    _fork(tmp_path)
    hint = er.restore_hint("prism-rocm", tmp_path)
    assert "prism-rocm" in hint
    assert "PrismML-Eng" in hint


def test_restore_hint_is_empty_for_an_unknown_engine(tmp_path):
    assert er.restore_hint("nope", tmp_path) == ""
