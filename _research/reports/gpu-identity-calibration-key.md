# GPU identity in Rigma's calibration key

## Verdict

**Yes — add it, but it is not sufficient on its own.** High confidence that throughput varies enough to matter (measured, same model + quant, ~55 GPUs), and equally high that GPU identity does *not* pin throughput. Add the key **and** strengthen runtime validation.

## 1. Throughput really does differ

llama.cpp's CUDA scoreboard, Llama 2 7B Q4_0, single GPU, pp512/tg128, no FA ([discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)):

- Nine distinct **24 GB** cards span tg128 **54.74 → 189.33** and pp512 **1,007 → 11,993**. VRAM size is not an identity proxy.
- RTX 3090→4090: pp512 5,174.69→11,992.70 (**+132%**); 4090→5090: tg128 186.21→290.02 (**+56%**); A100→H100: pp512 4,849.53→9,918.34 (**+105%**).
- Same-family is small: RTX 4080→4080 SUPER pp512 +1.2%, tg128 +4.1%.

Prefill is far more GPU-sensitive than decode. Cross-vendor on Vulkan ([#10879](https://github.com/ggml-org/llama.cpp/discussions/10879)): pp512 95.52 (Arc A770) → 3,726.99 (RX 7900 XTX) = **39x**.

**The same model is not stable either.** RX 7900 XTX in that thread: pp512 1,545 / 2,023 / 3,693 / 3,727 and tg128 88.12 / 136.24 / 170.69 / 182.63 — **2.4x / 2.1x** across driver, OS, overclock and multi-GPU visibility. Both maintainers state: *"Performance may vary depending on driver, operating system, board manufacturer, etc. even if the chip is the same."*

## 2. What to key on

- **NVIDIA — GPU UUID.** `nvmlDeviceGetUUID` returns *"the globally unique immutable UUID associated with this device… It does NOT correspond to any identifier printed on the board"* ([nvml.h](https://github.com/NVIDIA/gpu-monitoring-tools/blob/master/bindings/go/nvml/nvml.h)). It distinguishes two identical cards; PCI bus ID does not (slot address).
- **AMD-ROCm — gfx target + device ID.** gfx is the *effective kernel target* — `HSA_OVERRIDE_GFX_VERSION` overrides it at runtime ([`discover/amd.go`](https://github.com/ollama/ollama/blob/main/discover/amd.go)) — and governs speed. `rocm-smi --showuniqueid` exists but can be missing ([ROCm#5480](https://github.com/ROCm/ROCm/issues/5480)).
- **Vulkan — `vendorID` + `deviceID` + `deviceUUID`.** `deviceID` identifies the **model**, not the card: *"the same device ID should be used for all physical implementations of that device version"* — all four 7900 XTXes report identically ([VkPhysicalDeviceProperties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceProperties.html)). `deviceUUID` *"must be immutable… across instances, processes, driver APIs, driver versions, and system reboots"*, but Khronos warns it is *"not intended to be usable as a serializable persistent identifier"* and may change if a card moves slots ([VkPhysicalDeviceIDProperties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceIDProperties.html)) — use it, treat a change as soft. **Never use `pipelineCacheUUID`** (identifies *"a compatible device and driver combination"*, i.e. driver-sensitive) **or `deviceName`** (it embeds the driver — "…(RADV NAVI31)" vs "…(AMD proprietary driver)" — and Ollama must match it heuristically).
- **CPU** — keep the existing backend key.

## 3. Driver version

**Store it; do not key on it.** It matters — across a driver update on the same card, *"Non-FA PP is 5% faster and FA is 15% faster"* — but `driverVersion` is *"implementation-defined"*, so its encoding is not comparable across vendors. Keying it fragments the cache for an effect validation catches better.

## 4. Failure behaviour

Invalidate and re-measure; never error. Hard-invalidate on a confidently different identity (vendorID/deviceID/UUID/gfx). Soft-invalidate — mark stale, re-measure next run — on driver change and Vulkan `deviceUUID` change. Fall back to un-calibrated defaults on a miss. Also discard an entry when the first measured tokens deviate past a threshold; llama-bench ships `stddev_ts`/`samples_ts` for this.

## 5. Cache growth

Key = model × quant × backend × **normalised identity hash** (truncated sha256 of vendorID|deviceID|UUID|backend); driver stays a field, not a key. Single-machine means the multiplier is 1–4 (iGPU + dGPU). Prune to newest-per-identity; evict identities unseen for N runs.

## 6. Is the risk real?

Yes, and the multi-GPU case is strongest. An Optimus laptop measured Vulkan prefill *"around 3500 instead of 6000 t/s"* without correct device-routing env vars — **42% from device selection alone**. Merely having 4 GPUs visible cut a 7900 XTX's pp512 from 2,023 to 1,545 (**−24%**). eGPU hot-swap is real and Khronos names it. Cross-machine copying is worst because it is **silent**: a 3090 inherits a 4090's number.

**Shaping caveat:** power limits alone moved a 3090's pp512 from 4,175 to 5,406 (**+29%**) with no identity change — identity in the key is necessary but not sufficient.

## Prior art

- **llama-bench** records engine version (`build_commit`/`build_number`) and `gpu_info` = marketing name **only** (CUDA `prop.name`, Vulkan `props.deviceName`), plus raw `samples_ns`/`stddev_ts`. **No driver version, no hostname** ([llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)). ggml already exposes PCI bus ID as `device_id`; llama-bench just does not record it ([cuda](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu), [vulkan](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp)).
- **Ollama** `ml.DeviceInfo` is the model to follow: `Library` (backend), `PCIID` ("bus, device and domain ID… for deduplication when discovered by multiple backends"), `DriverMajor/Minor`, `NVIDIADriverMajor`, `GFXTarget` ([ml/device.go](https://github.com/ollama/ollama/blob/main/ml/device.go)).
- **MLPerf** counts *"drivers that significantly influences the running time"* as part of the SUT, and defines system equivalence by accelerator **model** and count — not per-card serial ([inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)). **vLLM** keys its compile cache on software hashes only ([backends.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/backends.py)).

## Could NOT verify

- NVIDIA's NVML HTML docs 404; NVML verified from NVIDIA's `nvml.h` (mirrored in `NVIDIA/gpu-monitoring-tools`), not a live docs page.
- AMD gfx-target stability across driver updates — no official AMD statement found.
- Geekbench's schema; whether Ollama persists a hardware-keyed cache; vLLM's non-compile caches.
- Any tool that keys a *tuning* cache on GPU identity — found none.
- External corroboration of Rigma's own 2,828 MB / 9.95→37.59 tok/s figure.
- `web_search` was unavailable (no API key); DuckDuckGo HTML intermittently captcha'd, and Bing and Brave were unusable.
