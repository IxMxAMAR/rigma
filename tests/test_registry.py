from rigma.registry import Registry


def test_bundled_registry_loads():
    r = Registry.load()
    assert "qwen3.6-35b-a3b" in r.models
    assert any(row["match"] == "RX 9070" for row in r.gpus)
    spec = r.models["qwen3.6-35b-a3b"]
    sizes = [g.bytes for g in spec.ggufs]
    assert all(s > 10**9 for s in sizes) and sizes == sorted(sizes, reverse=True)


def test_find_combo_exact_then_class():
    r = Registry.load()
    hit = r.find_combo("amd", "amd-radeon-rx-9070-xt-16g", 16, 16, "coding")
    assert hit and hit[0].flags.n_cpu_moe == 10 and "coding.json" in hit[1]
    cls = r.find_combo("nvidia", "unknown-card-16g", 16, 16, "general")
    assert cls and cls[1].startswith("_class/")
    assert r.find_combo("amd", "nope", 999, 999, "general") is None
