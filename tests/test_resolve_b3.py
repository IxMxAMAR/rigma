"""B3 fit math: dense partial offload, grow guard, multi-GPU, empty ggufs."""
from rigma.models import CachePolicy, CpuInfo, GgufFile, GpuInfo, \
    HardwareProfile, ModelSpec
from rigma.resolve import _budgets, _grow_ctx, fit_gguf


def _prof(vram=16368, ram_free=20000, gpus=None):
    if gpus is None:
        gpus = [GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=vram,
                        arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                        backends=["vulkan"])]
    return HardwareProfile(gpus=gpus, ram_mb=32768, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _dense(bytes_, n_layers=40, native=32768):
    return ModelSpec(slug="d", family="f", kind="dense", n_layers=n_layers,
                     full_attn_layers=n_layers, kv_heads=8, head_dim=128,
                     native_ctx=native,
                     ggufs=[GgufFile(repo="r", file="d.gguf", bytes=bytes_,
                                     quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy())


def test_dense_partial_offload_when_too_big():
    # 24GB dense file on a 16GB card: must partial-offload, not return None
    spec = _dense(int(22 * 2**30))
    flags = fit_gguf(spec, spec.ggufs[0], _prof(), 4096, [])
    assert flags is not None
    assert 0 < flags.ngl < spec.n_layers        # some layers on GPU, some CPU


def test_dense_fully_fits_keeps_ngl_default():
    spec = _dense(int(4 * 2**30))
    flags = fit_gguf(spec, spec.ggufs[0], _prof(), 4096, [])
    assert flags is not None and flags.ngl == 99   # full offload


def test_dense_too_big_even_for_ram_returns_none():
    spec = _dense(int(40 * 2**30))
    flags = fit_gguf(spec, spec.ggufs[0], _prof(ram_free=1000), 4096, [])
    assert flags is None


def test_grow_ctx_stops_before_forcing_more_offload():
    # a model that fits fully at small ctx but would need offload at huge ctx
    spec = _dense(int(13 * 2**30), native=262144)
    base = fit_gguf(spec, spec.ggufs[0], _prof(), 4096, [])
    assert base is not None and base.ngl == 99
    grown = _grow_ctx(spec, spec.ggufs[0], _prof(), base, [])
    # grow-to-fit raised ctx but never dropped below full GPU offload
    assert grown.ngl == 99 and grown.ctx >= base.ctx


def test_multi_gpu_sums_vram():
    two = [GpuInfo(vendor="nvidia", name="A", vram_mb=24576, arch="ada",
                   slug="a-24g", backends=["cuda"]),
           GpuInfo(vendor="nvidia", name="B", vram_mb=24576, arch="ada",
                   slug="b-24g", backends=["cuda"])]
    v1, _ = _budgets(_prof(gpus=two[:1]))
    v2, _ = _budgets(_prof(gpus=two))
    assert v2 > v1 * 1.8            # roughly double, minus per-card reserve


def test_unknown_nvidia_prefers_cuda():
    from rigma.probe import classify_gpu
    raw = {"vendor_id": 0x10DE, "name": "RTX 6090", "vram_mb": 32768}
    g = classify_gpu(raw, [], "windows")
    assert g.backends[0] == "cuda"
