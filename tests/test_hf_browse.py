"""Bazaar: HF search / remote header inspect / fit verdicts / add-to-library."""
import struct

import pytest

from rigma import hf_browse
from rigma.hangar import HangarError

T_U32, T_STR = 4, 8


def _s(b):
    return struct.pack("<Q", len(b)) + b


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", T_U32) + struct.pack("<I", v)


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", T_STR) + _s(v)


HEADER = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
          + struct.pack("<Q", 7) + b"".join([
              _kv_str(b"general.architecture", b"llama"),
              _kv_str(b"general.name", b"Web Tune 7B"),
              _kv_u32(b"llama.block_count", 32),
              _kv_u32(b"llama.context_length", 131072),
              _kv_u32(b"llama.embedding_length", 4096),
              _kv_u32(b"llama.attention.head_count", 32),
              _kv_u32(b"llama.attention.head_count_kv", 8),
          ]) + b"\x00" * 64)

TREE = [
    {"path": "WebTune-Q8_0.gguf", "size": 8 * 2**30},
    {"path": "WebTune-Q4_K_M.gguf", "size": 4 * 2**30},
    {"path": "WebTune-Q4_K_M-00001-of-00002.gguf", "size": 2 * 2**30},
    {"path": "mmproj-WebTune-F16.gguf", "size": 800 * 2**20},
    {"path": "README.md", "size": 100},
]


@pytest.fixture
def fake_hf(monkeypatch):
    def _get_json(path, params=None):
        if path == "/api/models":
            return [{"id": "cool/WebTune-GGUF", "downloads": 1234,
                     "likes": 56, "lastModified": "2026-07-01T00:00:00Z"}]
        if path.endswith("/tree/main"):
            return TREE
        raise AssertionError(path)
    monkeypatch.setattr(hf_browse, "_get_json", _get_json)
    monkeypatch.setattr(hf_browse, "_fetch_head",
                        lambda repo, file, cap: HEADER)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_search_shapes_results(fake_hf):
    out = hf_browse.search("webtune")
    assert out == [{"repo": "cool/WebTune-GGUF", "downloads": 1234,
                    "likes": 56, "updated": "2026-07-01"}]


def test_repo_files_skips_split_and_picks_f16_mmproj(fake_hf):
    rf = hf_browse.repo_files("cool/WebTune-GGUF")
    assert [g["file"] for g in rf["ggufs"]] == ["WebTune-Q8_0.gguf",
                                                "WebTune-Q4_K_M.gguf"]
    assert rf["split_skipped"] == 1
    assert rf["mmproj"]["file"] == "mmproj-WebTune-F16.gguf"


def test_inspect_repo_fit_verdicts_and_caps(fake_hf, home):
    d = hf_browse.inspect_repo("cool/WebTune-GGUF")
    assert d["name"] == "web-tune-7b" and not d["already"]
    assert d["native_ctx"] == 131072
    assert "vision" in d["capabilities"]        # mmproj present
    by_q = {g["quant"]: g for g in d["ggufs"]}
    # 16GB card: 4GB file fits (and grows), 8GB+kv is judged by real math
    assert by_q["Q4_K_M"]["fit"]["ok"] is True
    assert by_q["Q4_K_M"]["fit"]["ctx"] >= 8192
    assert set(by_q) == {"Q8_0", "Q4_K_M"}


def test_add_model_registers_pullable_spec(fake_hf, home):
    from rigma.registry import Registry
    spec = hf_browse.add_model("cool/WebTune-GGUF")
    assert spec.slug == "web-tune-7b" and spec.custom
    reg = Registry.load()
    got = reg.models["web-tune-7b"]
    assert got.ggufs[0].repo == "cool/WebTune-GGUF"   # real repo -> pullable
    assert got.mmproj is not None
    with pytest.raises(HangarError, match="already in your library"):
        hf_browse.add_model("cool/WebTune-GGUF")


def test_split_only_repo_is_a_clean_error(monkeypatch, home):
    monkeypatch.setattr(hf_browse, "_get_json", lambda p, params=None: [
        {"path": "Big-00001-of-00009.gguf", "size": 5}])
    with pytest.raises(HangarError, match="split"):
        hf_browse.add_model("cool/Split-GGUF")


def test_range_escalation_on_truncated_header(monkeypatch, home):
    calls = []
    # a header whose big template string extends past the 8MB range boundary
    big = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
           + struct.pack("<Q", 1) + _s(b"tokenizer.chat_template")
           + struct.pack("<I", T_STR) + struct.pack("<Q", 9 * 2**20)
           + b"t" * (9 * 2**20))

    def fetch(repo, file, cap):
        calls.append(cap)
        return big[:8 * 2**20] if cap == 8 else HEADER
    monkeypatch.setattr(hf_browse, "_fetch_head", fetch)
    monkeypatch.setattr(hf_browse, "_get_json", lambda p, params=None: TREE)
    d = hf_browse.inspect_repo("cool/WebTune-GGUF")
    assert d["name"] == "web-tune-7b"
    assert calls == [8, 32]


def test_hf_5xx_and_429_stay_friendly(monkeypatch):
    """Review 2026-07-17: rate limits / HF hiccups must be HangarError (clean
    502 upstream), never a raw 500."""
    import types

    import httpx as _httpx
    for code in (429, 500, 503):
        monkeypatch.setattr(hf_browse.httpx, "get",
                            lambda *a, code=code, **k: types.SimpleNamespace(
                                status_code=code))
        with pytest.raises(HangarError, match=str(code)):
            hf_browse.search("x")
    assert _httpx  # imported to prove no real network path was involved


def test_imatrix_gguf_is_not_a_quant_and_probe_falls_forward(monkeypatch,
                                                             home):
    """Live find 2026-07-17 (bartowski/Cydonia): repos ship imatrix data as
    .gguf — must be excluded from quants, and the header probe must not die
    on a non-model gguf."""
    tree = [{"path": "Model-imatrix.gguf", "size": 10 * 2**20},
            {"path": "Model-Q4_K_M.gguf", "size": 4 * 2**30}]
    monkeypatch.setattr(hf_browse, "_get_json", lambda p, params=None: tree)
    monkeypatch.setattr(hf_browse, "_fetch_head",
                        lambda repo, file, cap: HEADER)
    rf = hf_browse.repo_files("x/y")
    assert [g["file"] for g in rf["ggufs"]] == ["Model-Q4_K_M.gguf"]
    d = hf_browse.inspect_repo("x/y")
    assert d["name"] == "web-tune-7b"


def test_probe_falls_forward_past_metadata_less_gguf(monkeypatch, home):
    """Even without the name filter, a metadata-less smallest gguf must not
    kill the repo — the next candidate gets probed."""
    bare = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", 1)
            + _kv_str(b"general.architecture", b"mystery"))
    tree = [{"path": "weird.gguf", "size": 5 * 2**20},
            {"path": "Model-Q4_K_M.gguf", "size": 4 * 2**30}]
    monkeypatch.setattr(hf_browse, "_get_json", lambda p, params=None: tree)
    monkeypatch.setattr(hf_browse, "_fetch_head",
                        lambda repo, file, cap:
                        bare if file == "weird.gguf" else HEADER)
    d = hf_browse.inspect_repo("x/y")
    assert d["name"] == "web-tune-7b"
