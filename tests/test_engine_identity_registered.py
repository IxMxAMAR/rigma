"""A model whose tensor types need a REGISTERED engine (the PrismML fork for
PQ2_0) runs that engine, so "which build produced this" must name it. Asking by
backend measured the pin's directory instead: the Engine page showed b9867 while
prism-b10743 served, and calibrations were stamped with the pin's build."""
from types import SimpleNamespace

from rigma import bench, engine_build, server_ops


def _builds(monkeypatch):
    table = {"prism.exe": "b10743+adfffbe", "pin.exe": "b9867+152d337fa"}
    monkeypatch.setattr(engine_build, "cached_build", lambda exe: SimpleNamespace(
        ok=str(exe) in table, identity=table.get(str(exe), "")))
    monkeypatch.setattr(server_ops, "engine_version", lambda backend="": table["pin.exe"])


def test_a_plan_on_a_registered_engine_reports_that_engines_build(monkeypatch):
    _builds(monkeypatch)
    monkeypatch.setattr(server_ops, "_registered_engine_for",
                        lambda gguf, backend: SimpleNamespace(exe="prism.exe"))
    assert server_ops.plan_engine_identity(object(), "vulkan") == "b10743+adfffbe"


def test_a_plan_on_the_pin_reports_the_pins_build(monkeypatch):
    _builds(monkeypatch)
    monkeypatch.setattr(server_ops, "_registered_engine_for", lambda gguf, backend: None)
    assert server_ops.plan_engine_identity(object(), "vulkan") == "b9867+152d337fa"


def test_an_unmeasurable_binary_is_unknown_not_the_pin(monkeypatch):
    _builds(monkeypatch)
    assert server_ops.engine_identity_of("missing.exe") == ""


def test_a_calibration_is_stamped_with_the_engine_that_produced_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _builds(monkeypatch)
    key = bench.calibration_key("bonsai", "Q2_0", "vulkan")
    bench.save_calibration(key, {"tg_tps": 54.5}, backend="vulkan",
                           engine="b10743+adfffbe")
    assert bench.load_calibration()[key]["engine"] == "b10743+adfffbe"
