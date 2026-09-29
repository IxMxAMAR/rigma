"""R3-ENG-3, the launch half: a registered engine must actually be used.

`engine_registry` could already describe, verify and `select` a non-pinned engine, but
nothing in the launch path called `select`. So the owner installed PrismML's fork,
registered it, and `rigma up` still chose the pinned b9867 build and died with
`invalid ggml type 142. should be in [0, 42)` — the exact failure registration exists to
prevent.

Note the axis: `test_r3_engine_selection.py` covers the ENGINE RUNTIME choice
(llamacpp vs vllm). This covers the ENGINE BINARY choice (the pin vs a registered
third-party build), which is a different question answered by the model's tensor types.
"""
from __future__ import annotations

import pytest

from rigma import engine_registry, server_ops

BONSAI = {0: 1, 30: 1, 142: 100}   # real Ternary-Bonsai-2-27B-PQ2_0 histogram
SMOL = {0: 1, 8: 12, 11: 40, 20: 5}


class _Idx:
    def __init__(self, counts, complete=True):
        self.type_counts = counts
        self.types_complete = complete


class _Gguf:
    def __init__(self, file="model.gguf"):
        self.file = file


@pytest.fixture
def home(tmp_path, monkeypatch):
    """RIGMA_HOME for the state round-trip tests."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A home with the model on disk, so the file check passes."""
    home = tmp_path / "rigma"
    (home / "models").mkdir(parents=True)
    (home / "models" / "model.gguf").write_bytes(b"not a real gguf")
    monkeypatch.setattr(server_ops, "rigma_home", lambda: home)
    return home


def _register(monkeypatch, tmp_path, *, name, path, backend, types, known=True):
    monkeypatch.setattr(engine_registry, "registry_path",
                        lambda home=None: tmp_path / "engines-custom.json")
    engine_registry.register(engine_registry.CustomEngine(
        name=name, path=str(path), backend=backend, source="test",
        types=list(types), types_known=known))


def _exe(tmp_path, name="llama-server.exe"):
    p = tmp_path / name
    p.write_bytes(b"fake")
    return p


def _types(monkeypatch, counts, complete=True):
    import rigma.gguf_meta as gm
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(counts, complete))


def test_a_registered_engine_that_accepts_the_model_is_chosen(wired, monkeypatch, tmp_path):
    """The bug: a fork that can load 142 must be selected over the pin that cannot."""
    exe = _exe(tmp_path)
    _register(monkeypatch, tmp_path, name="prism", path=exe, backend="vulkan",
              types=range(144))
    _types(monkeypatch, BONSAI)
    got = server_ops._registered_engine_for(_Gguf(), "vulkan")
    assert got is not None
    assert got.name == "prism"
    assert got.exe == exe


def test_a_registered_engine_that_cannot_load_the_model_is_not_chosen(
        wired, monkeypatch, tmp_path):
    """A build whose table stops at 43 must not be handed a model using 142."""
    _register(monkeypatch, tmp_path, name="mainline-fork", path=_exe(tmp_path),
              backend="vulkan", types=range(43))
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_an_engine_with_UNKNOWN_capabilities_is_not_chosen(wired, monkeypatch, tmp_path):
    """Unknown is not a licence to pick. `accepts` returns None here, and selecting it
    would put a model on a binary nobody vouched for; `rigma engine-runtimes` offers it
    to a human instead."""
    _register(monkeypatch, tmp_path, name="mystery", path=_exe(tmp_path),
              backend="vulkan", types=[], known=False)
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_an_engine_for_another_backend_is_not_chosen(wired, monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, name="rocm-only", path=_exe(tmp_path),
              backend="rocm", types=range(144))
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_a_missing_model_file_yields_no_opinion(wired, monkeypatch, tmp_path):
    _register(monkeypatch, tmp_path, name="prism", path=_exe(tmp_path),
              backend="vulkan", types=range(144))
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf("absent.gguf"), "vulkan") is None


def test_a_truncated_read_yields_no_opinion(wired, monkeypatch, tmp_path):
    """A partial histogram cannot show an engine accepts the file, so it must not be
    used to move a model onto a third-party binary."""
    _register(monkeypatch, tmp_path, name="prism", path=_exe(tmp_path),
              backend="vulkan", types=range(144))
    _types(monkeypatch, BONSAI, complete=False)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_a_registered_engine_whose_binary_vanished_is_not_chosen(
        wired, monkeypatch, tmp_path):
    """The registry entry can outlive the file it points at. Selecting a missing exe
    would turn a working pinned launch into a failure to spawn."""
    _register(monkeypatch, tmp_path, name="prism", path=tmp_path / "gone.exe",
              backend="vulkan", types=range(144))
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_no_registry_at_all_yields_no_opinion(wired, monkeypatch):
    """The default machine: nothing registered, so the pin is used as before. This is
    the guarantee that registration is purely additive."""
    _types(monkeypatch, BONSAI)
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


# --- the shared seam the launch paths actually call --------------------------
#
# `_registered_engine_for` returning the right engine is not enough. Rigma launches from
# THREE places — `up` (with its own fallback ladder), `sweep`, and `perform_switch` (the
# UI) — and each called `runtime.ensure_engine` directly. Wiring only `perform_switch`
# left `rigma up --model <pq2_0 model>` launching the pin, which refused type 142 and let
# the ladder quietly serve SmolLM2. These pin the seam all three now share.

def test_engine_binary_for_prefers_a_registered_engine(wired, monkeypatch, tmp_path):
    exe = _exe(tmp_path)
    _register(monkeypatch, tmp_path, name="prism", path=exe, backend="vulkan",
              types=range(144))
    _types(monkeypatch, BONSAI)
    got, rec = server_ops.engine_binary_for(_Gguf(), "vulkan", "windows")
    assert got == exe
    assert rec["kind"] == "registered" and rec["name"] == "prism"


def test_engine_binary_for_falls_back_to_the_pin(wired, monkeypatch, tmp_path):
    """With nothing registered, the pin is used and the record says so — the default
    machine must be indistinguishable from before this change."""
    import rigma.runtime as runtime
    pinned = tmp_path / "pinned-llama-server.exe"
    pinned.write_bytes(b"fake")
    monkeypatch.setattr(runtime, "ensure_engine", lambda backend, os_name: pinned)
    _types(monkeypatch, BONSAI)
    got, rec = server_ops.engine_binary_for(_Gguf(), "vulkan", "windows")
    assert got == pinned
    assert rec["kind"] == "pinned" and rec["name"] == "windows/vulkan"


def test_engine_binary_for_prefers_the_registered_engine_over_the_pin(
        wired, monkeypatch, tmp_path):
    """The regression that shipped once already: the pin is available and would be
    returned, but a registered engine that covers the file must win."""
    import rigma.runtime as runtime
    pinned = tmp_path / "pinned-llama-server.exe"
    pinned.write_bytes(b"fake")
    monkeypatch.setattr(runtime, "ensure_engine", lambda backend, os_name: pinned)
    custom = _exe(tmp_path, "prism-llama-server.exe")
    _register(monkeypatch, tmp_path, name="prism", path=custom, backend="vulkan",
              types=range(144))
    _types(monkeypatch, BONSAI)
    got, rec = server_ops.engine_binary_for(_Gguf(), "vulkan", "windows")
    assert got == custom, "the pin was used for a model only the fork can load"
    assert rec["kind"] == "registered"


def test_both_launch_paths_use_the_seam():
    """A wiring test, because this bug was purely one of wiring.

    The CLI's `up` and `sweep` must not call `runtime.ensure_engine` directly any more —
    that is exactly how the registered engine got bypassed on the command line.
    """
    import pathlib
    src = pathlib.Path(server_ops.__file__).parent / "cli.py"
    text = src.read_text(encoding="utf-8")
    assert "engine_binary_for" in text, (
        "cli.py does not use the shared engine-selection seam")
    # The two launch sites inside `up` and `sweep` are the ones that regressed.
    assert text.count("runtime.ensure_engine") == 0, (
        "a CLI path still calls runtime.ensure_engine directly, which bypasses "
        "registered engines")


def test_an_unreadable_header_yields_no_opinion(wired, monkeypatch, tmp_path):
    """A read failure must not be treated as 'no types', which would let an engine with
    an empty type list match vacuously."""
    import rigma.gguf_meta as gm

    def boom(p):
        raise ValueError("bad header")

    monkeypatch.setattr(gm, "read_tensor_index", boom)
    _register(monkeypatch, tmp_path, name="prism", path=_exe(tmp_path),
              backend="vulkan", types=range(144))
    assert server_ops._registered_engine_for(_Gguf(), "vulkan") is None


def test_an_ordinary_model_is_still_offered_a_covering_registered_engine(
        wired, monkeypatch, tmp_path):
    """The helper reports capability, not policy: a registered build whose table covers
    an ordinary model's types IS a valid answer. The pin is protected by the CALLER,
    which consults the registry only when the pin cannot load the file — so that
    ordering is what these tests must not let anyone invert silently.
    """
    _register(monkeypatch, tmp_path, name="prism", path=_exe(tmp_path),
              backend="vulkan", types=range(144))
    _types(monkeypatch, SMOL)
    got = server_ops._registered_engine_for(_Gguf(), "vulkan")
    assert got is not None and got.name == "prism"


def test_state_round_trips_the_engine_binary(home):
    """`rigma status` must be able to say WHICH BINARY is serving, because the owner
    otherwise learned it only by reading the process table."""
    from rigma import state as st
    st.write_state("m", "Q2_0", 11500, engine_pid=-1, ui_pid=1, backend="vulkan",
                   engine="llamacpp",
                   engine_binary={"kind": "registered", "name": "prism-b10743-vulkan",
                                  "path": "C:/x/llama-server.exe", "source": "PrismML"})
    s = st.read_state()
    assert s["engine_binary"]["name"] == "prism-b10743-vulkan"
    # and the two axes stay distinct: `engine` is the RUNTIME, a string that
    # calibration compares for equality, so it must not have been overwritten.
    assert s["engine"] == "llamacpp"


def test_engine_binary_defaults_to_absent_for_an_old_record(home):
    """A state.json written before this field existed must degrade, not explode —
    the same rule `engine` already follows."""
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=1, backend="vulkan")
    assert st.read_state().get("engine_binary") is None


def test_engine_binary_survives_an_unrelated_update(home):
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=1, backend="vulkan",
                   engine_binary={"kind": "pinned", "name": "windows/vulkan",
                                  "path": "p", "source": ""})
    st.update_state(ctx=8192)
    assert st.read_state()["engine_binary"]["kind"] == "pinned"


# --- the UI surface ----------------------------------------------------------

def test_api_server_reports_the_engine_binary(home):
    """Recording it is not enough — the Engine page has to be able to show it, or the
    only way to learn which binary is serving remains the process table."""
    import os

    from fastapi.testclient import TestClient

    from rigma import serve
    from rigma import state as st
    st.write_state("m", "Q2_0", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vulkan", engine="llamacpp",
                   engine_binary={"kind": "registered", "name": "prism-b10743-vulkan",
                                  "path": "C:/x/llama-server.exe", "source": "PrismML"})
    body = TestClient(serve.build_app(upstream_port=11500)).get("/api/server").json()
    assert body["engine_binary"]["name"] == "prism-b10743-vulkan"
    assert body["engine_binary"]["kind"] == "registered"
    # and the runtime axis is untouched by it
    assert body["engine"] == "llamacpp"


def test_api_server_reports_null_when_no_binary_was_recorded(home):
    """An older record must not be made to look like the pin."""
    import os

    from fastapi.testclient import TestClient

    from rigma import serve
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vulkan")
    body = TestClient(serve.build_app(upstream_port=11500)).get("/api/server").json()
    assert body.get("engine_binary") is None
