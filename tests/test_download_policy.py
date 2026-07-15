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


def test_default_is_classic_and_uncapped(tmp_path, monkeypatch):
    # Owner decision 2026-07-14: no artificial speed caps. Classic downloader
    # because xet hung mid-download on Windows twice (2026-07-06, 2026-07-14).
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    seen = _capture(tmp_path, monkeypatch)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["xet_off"] == "1"     # classic: line-rate, deterministic resume
    assert seen["conc"] is None       # and NO concurrency throttle env


def test_explicit_env_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")   # user opts back into xet
    seen = _capture(tmp_path, monkeypatch)
    from rigma.models import GgufFile
    runtime.ensure_model(GgufFile(repo="r/x", file="m.gguf", bytes=1, quant="Q8_0"))
    assert seen["xet_off"] == "0"
