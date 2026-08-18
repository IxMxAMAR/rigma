"""Bazaar: HF search / remote header inspect / fit verdicts / add-to-library."""
import pathlib
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
    # fit verdicts must be deterministic: CI runners have no GPU, and the
    # fixture's assertions describe a 16GB card
    from rigma import probe as _probe
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    monkeypatch.setattr(_probe, "probe_hardware", lambda gpus=None: HardwareProfile(
        gpus=[GpuInfo(vendor="amd", name="RX", vram_mb=16368,
                      backends=["vulkan"])],
        ram_mb=32768, ram_free_mb=24000, cpu=CpuInfo(cores=16),
        os="windows", disk_free_gb=400.0))
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


def test_search_result_keys_match_what_the_v2_ui_reads():
    """The repo id ships as `repo`, never `id`.

    Live 2026-07-30: the v2 Models surface read `hit.id` — a key this endpoint
    does not return, because it RENAMES Hugging Face's own `id` on the way
    out. Every result rendered as a blank row with a dead `add` button, and
    HF model adding was broken in the whole v2 UI. TypeScript missed it
    because HfHit carried an index signature; nothing on the Python side
    said which name the UI was entitled to. This does.
    """
    src = (pathlib.Path(__file__).parent.parent / "frontend-v2" / "src")
    api_ts = (src / "lib" / "engineApi.ts").read_text(encoding="utf-8")
    hit = api_ts.split("export interface HfHit", 1)[1].split("}", 1)[0]
    assert "repo:" in hit, "HfHit must declare `repo` — the key search() returns"
    assert "id:" not in hit, "HfHit must not declare `id`; the API renames it"
    # an index signature makes any typo type-check and render as undefined
    assert "[k: string]" not in hit, "HfHit must not carry an index signature"

    ui = (src / "models" / "ModelsSurface.tsx").read_text(encoding="utf-8")
    body = ui.split("function HfSearch", 1)[1]
    assert "h.repo" in body
    assert "h.id" not in body, "the Models surface is reading a key that does not exist"


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
    with pytest.raises(HangarError, match="already added that repo"):
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


def test_fetch_head_caps_when_server_ignores_range(monkeypatch, home):
    """Review 2026-07-18: a mirror ignoring Range and returning the whole
    40GB file must not be pulled into memory — stream + hard cap."""

    class _Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_bytes(self, n):
            for _ in range(100):        # would be 100MB if uncapped
                yield b"x" * (1 << 20)
    monkeypatch.setattr(hf_browse.httpx, "stream",
                        lambda *a, **k: _Resp())
    got = hf_browse._fetch_head("a/b", "big.gguf", 8)
    assert len(got) == 8 * 2**20       # capped at 8MB, not 100MB


def test_repo_files_requests_recursive_tree(monkeypatch, home):
    seen = {}
    def _get(path, params=None):
        seen["params"] = params
        return TREE
    monkeypatch.setattr(hf_browse, "_get_json", _get)
    hf_browse.repo_files("a/b")
    assert seen["params"] == {"recursive": "true"}


def test_mtp_and_draft_ggufs_excluded_from_quants(monkeypatch, home):
    """User-reported 2026-07-18 (HauhauCS Gemma4-26B-A4B ...-MTP): the repo
    ships an mtp-*.gguf draft head. Left in, it shows as a phantom 0.2GB
    'quant' AND — being smallest — becomes the header-probe source, making a
    26B MoE read as 'dense'. Must be filtered like split/imatrix files."""
    tree = [
        {"path": "Gemma4-26B-A4B-...-Q4_K_M.gguf", "size": 16018 * 2**20},
        {"path": "mmproj-Gemma4-26B-A4B-...-BF16.gguf", "size": 1139 * 2**20},
        {"path": "mtp-gemma-4-26B-A4B-it.gguf", "size": 240 * 2**20},
        {"path": "model-imatrix.gguf", "size": 10 * 2**20},
        {"path": "eagle-draft-head.gguf", "size": 50 * 2**20},
    ]
    monkeypatch.setattr(hf_browse, "_get_json", lambda p, params=None: tree)
    rf = hf_browse.repo_files("HauhauCS/x")
    files = [g["file"] for g in rf["ggufs"]]
    assert files == ["Gemma4-26B-A4B-...-Q4_K_M.gguf"]   # only the real model
    assert "mtp" not in " ".join(files).lower()
    assert "eagle" not in " ".join(files).lower()
    assert rf["mmproj"] is not None                       # projector still found


def test_recommend_picks_speed_sweet_spot_not_biggest(monkeypatch, home):
    """User-reported 2026-07-18: the ★ was on the LARGEST quant, which only
    'fits' via heavy RAM offload (slowest). Recommend the best quality that
    still runs at GPU speed."""
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    # big dense model: only small quants fit on the 16GB GPU
    def mk(bytes_):
        return GgufFile(repo="r", file=f"m-{bytes_}.gguf", bytes=bytes_,
                        quant={30: "Q8_0", 16: "Q4_K_M", 10: "IQ3_M"}[
                            bytes_ // 2**30])
    spec = ModelSpec(slug="big", family="f", kind="dense", n_layers=64,
                     full_attn_layers=64, kv_heads=4, head_dim=256,
                     native_ctx=262144,
                     ggufs=[mk(30 * 2**30), mk(16 * 2**30), mk(10 * 2**30)],
                     use_cases=["general"], cache_type_policy=CachePolicy())
    monkeypatch.setattr(hf_browse, "_spec_from_repo",
                        lambda repo: (spec, {"mmproj": None, "split_skipped": 0}))
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    prof = HardwareProfile(
        gpus=[GpuInfo(vendor="amd", name="x", vram_mb=16368, arch="rdna4",
                      slug="x", backends=["vulkan"])],
        ram_mb=49152, ram_free_mb=44000, cpu=CpuInfo(cores=16), os="windows",
        disk_free_gb=400.0)
    d = hf_browse.inspect_repo("x/y", profile=prof)
    speeds = {q["quant"]: q["fit"].get("speed") for q in d["ggufs"]}
    assert speeds["Q8_0"] == "offload"          # 30GB spills to RAM — slow
    assert speeds["IQ3_M"] == "gpu"             # 10GB fits on the card — fast
    # recommend the largest that's GPU/light, NOT the biggest-that-fits
    assert d["recommended"] != "Q8_0"
    assert d["recommended"] in ("Q4_K_M", "IQ3_M")


def test_mtp_preserved_models_are_not_filtered_as_aux():
    """"mtp" as a bare substring rejected entire repos. MTP-PRESERVED models
    carry it in their filename and are exactly what a user wants; only a
    standalone draft head is auxiliary. This made
    SC117/...-MTP-Preserved-APEX-GGUF report "no single-file gguf in that repo"
    when it has three."""
    from rigma.hf_browse import _is_aux_gguf as aux, _is_aux_mtp
    # a TINY file beside a big one — the size a real draft head is
    def head(name, size=240_000_000, biggest=26_000_000_000):
        return _is_aux_mtp(name, size, biggest)

    # real filenames from repos that must WORK: "mtp" mid-name is a descriptor
    assert not aux("Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-"
                   "APEX-I-Compact.gguf")
    assert not aux("Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf")      # unsloth official
    assert not aux("Huihui-Qwen3.6-35B-A3B-abliterated-MTP-Q4_K.gguf")
    assert not head("Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf")
    # genuine auxiliary heads must still be skipped — at draft-head SIZE
    assert head("mtp.gguf")
    assert head("Qwen3-30B-mtp.gguf")
    assert head("model-mtp-head.gguf")
    assert head("model_mtp_head.gguf")
    # a head can also be named like an mmproj, with mtp as a PREFIX
    assert head("mtp-gemma-4-26B-A4B-it.gguf")
    # ...but the SAME name at model size is a model (0bserverx, 2026-08-19)
    assert not head("Qwen3-30B-mtp.gguf", size=13_000_000_000,
                    biggest=50_500_000_000)
    # unrelated markers unchanged, and still name-only
    assert aux("something-imatrix.gguf") and aux("foo-draft.gguf")
    assert not aux("Rocinante-X-12B-v1b-Q6_K.gguf")


def test_quant_labels_are_always_distinguishable():
    """Repos that don't use Q4_K_M-style tags collapsed to a single label.
    SC117's APEX ships I-Compact / I-Quality / I-Balanced: the picker showed
    three identical "GGUF" rows and marked EVERY one recommended, because the
    badge compares on this label."""
    from rigma.hf_browse import _distinct_quants as dq
    apex = ["m-Native-MTP-Preserved-APEX-I-Balanced.gguf",
            "m-Native-MTP-Preserved-APEX-I-Quality.gguf",
            "m-Native-MTP-Preserved-APEX-I-Compact.gguf"]
    got = dq(apex)
    assert len(set(got)) == 3, got
    assert got == ["BALANCED", "QUALITY", "COMPACT"]
    # ordinary quant tags must be left exactly as they are
    assert dq(["m-Q4_K_M.gguf", "m-Q5_K_M.gguf", "m-IQ3_M.gguf"]) == \
        ["Q4_K_M", "Q5_K_M", "IQ3_M"]
    assert dq(["Rocinante-X-12B-v1b-Q6_K.gguf"]) == ["Q6_K"]
    # same stem in different folders still resolves to distinct labels
    assert len(set(dq(["a/model.gguf", "b/model.gguf"]))) == 2


def test_already_added_names_the_repo_you_typed(fake_hf, home):
    """Re-adding the SAME repo must say so plainly."""
    hf_browse.add_model("cool/WebTune-GGUF")
    with pytest.raises(HangarError) as e:
        hf_browse.add_model("cool/WebTune-GGUF")
    msg = str(e.value)
    assert "already added that repo" in msg
    assert "web-tune-7b" in msg              # names where it landed


def test_a_mirror_is_refused_with_the_reason_it_collided(fake_hf, home,
                                                         monkeypatch):
    """A DIFFERENT repo carrying the same model refused under a slug the user
    never typed. Live 2026-07-30: adding 0bserverx/Qwen3.8-27B-Heretic-... was
    rejected as "qwen38-ara-v5 is already in your library" — a name found
    nowhere in the input, because the slug comes from the gguf's own
    general.name."""
    hf_browse.add_model("cool/WebTune-GGUF")
    with pytest.raises(HangarError) as e:
        hf_browse.add_model("someone-else/WebTune-mirror-GGUF")
    msg = str(e.value)
    assert "general.name" in msg              # explains WHERE the slug is from
    assert "cool/WebTune-GGUF" in msg         # and who already holds it
    assert "same model" in msg
    assert "Remove" in msg                    # and what to do about it


# --- an MTP file can be a whole model, not a draft head ----------------------
# Live 2026-08-19: 0bserverx republished the owner's model with MTP variants
# named RVN-IQ3_M-mtp.gguf, RVN-Q8_0-mtp.gguf and so on — full models with the
# draft head baked in, 0.4 GB larger than their plain siblings. The name filter
# treated anything ending in "-mtp.gguf" as a standalone draft head and hid 26
# of the repo's 53 files, including a 50.5 GB BF16. Exactly the files that make
# speculative decoding possible were the ones made invisible.
#
# A real standalone head is TINY next to the models it drafts for — the Gemma
# case that motivated the filter was a 240 MB head beside a 26 GB model. Size is
# the discriminator the name cannot be.

def _tree(monkeypatch, files):
    from rigma import hf_browse
    monkeypatch.setattr(hf_browse, "_get_json",
                        lambda path, params=None: [
                            {"path": p, "size": s} for p, s in files])


def test_a_gigabyte_scale_mtp_file_is_a_model_not_a_head(monkeypatch):
    from rigma import hf_browse
    _tree(monkeypatch, [
        ("RVN-BF16.gguf", 53_800_000_000),
        ("RVN-IQ3_M.gguf", 12_600_000_000),
        ("RVN-IQ3_M-mtp.gguf", 13_000_000_000),
        ("RVN-Q8_0-mtp.gguf", 27_100_000_000),
    ])
    names = [g["file"] for g in hf_browse.repo_files("r")["ggufs"]]
    assert "RVN-IQ3_M-mtp.gguf" in names
    assert "RVN-Q8_0-mtp.gguf" in names


def test_a_tiny_standalone_draft_head_is_still_filtered(monkeypatch):
    """The case the filter was written for: a 240 MB head beside a 26 GB model,
    which if listed becomes a phantom 0.2 GB 'quant' AND, being smallest, gets
    picked as the header-probe source."""
    from rigma import hf_browse
    _tree(monkeypatch, [
        ("gemma-4-26B-A4B-Q4_K_M.gguf", 26_000_000_000),
        ("mtp-gemma-4-26B-A4B-it.gguf", 240_000_000),
        ("gemma-4-26B-A4B-mtp.gguf", 240_000_000),
    ])
    names = [g["file"] for g in hf_browse.repo_files("r")["ggufs"]]
    assert names == ["gemma-4-26B-A4B-Q4_K_M.gguf"]


def test_imatrix_and_other_aux_files_are_unaffected(monkeypatch):
    from rigma import hf_browse
    _tree(monkeypatch, [
        ("model-Q4_K_M.gguf", 16_000_000_000),
        ("model-imatrix.gguf", 5_000_000),
        ("model-draft.gguf", 500_000_000),
        ("model-eagle.gguf", 400_000_000),
    ])
    names = [g["file"] for g in hf_browse.repo_files("r")["ggufs"]]
    assert names == ["model-Q4_K_M.gguf"]
