import rigma.runtime as runtime


def test_ensure_model_defaults_to_polite_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    seen = {}

    def fake_download(repo_id, filename, local_dir):
        import os
        seen["conc"] = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS")
        p = tmp_path / "models" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["conc"] == "4"  # polite default, doesn't saturate home connections


def test_turbo_env_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
    seen = {}

    def fake_download(repo_id, filename, local_dir):
        import os
        seen["conc"] = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS")
        p = tmp_path / "models" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["conc"] == "16"  # explicit user choice wins
