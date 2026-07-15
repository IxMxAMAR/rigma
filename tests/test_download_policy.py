import rigma.runtime as runtime


def _capture(tmp_path, monkeypatch):
    seen = {}

    def fake_download(repo_id, filename, local_dir):
        import os
        seen["conc"] = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS")
        seen["xet_off"] = os.environ.get("HF_HUB_DISABLE_XET")
        p = tmp_path / "models" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(runtime, "hf_hub_download", fake_download)
    return seen


def test_ensure_model_defaults_to_full_speed(tmp_path, monkeypatch):
    # Owner decision 2026-07-14: throttled-by-default punished every download
    # and didn't fix gaming packet loss. Full speed is the default now.
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    seen = _capture(tmp_path, monkeypatch)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["xet_off"] == "0" and seen["conc"] == "16"


def test_polite_flag_single_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    seen = _capture(tmp_path, monkeypatch)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"),
                         polite=True)
    assert seen["xet_off"] == "1"  # classic downloader: deterministic resume


def test_explicit_env_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "2")
    seen = _capture(tmp_path, monkeypatch)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["xet_off"] == "1" and seen["conc"] == "2"  # user env respected
