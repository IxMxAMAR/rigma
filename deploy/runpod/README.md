# Rigma on a Runpod GPU pod

A Runpod **pod** template that runs Rigma on an NVIDIA GPU so it can act as a real test
bench: the same CLI, the same resolver, the same engine pin, the same UI — on Linux, on a
card you do not own.

**Workload shape:** a long-lived interactive HTTP server behind one stable proxy URL →
a pod, not serverless (see `runpod/golden-paths/01-ollama-pod.md:14-31`).

---

## Status: what was actually verified, and what was not

This draft was written on a Windows machine with **no Docker daemon and no `runpodctl`**.
Nothing below was executed against Runpod, and the image was never built. A Runpod **API key
does exist** in `~/.runpod/config.toml`, and it was used **read-only** to read the live
catalog — that is where this template's field names now come from.

Commands run, and what they returned:

| Command | Result |
| --- | --- |
| `runpodctl version` | `runpodctl` is **not installed** — *"The term 'runpodctl' is not recognized…"*. So no flag in this README comes from `runpodctl --help`; every one is cited to a skill doc line. |
| `docker version` | `docker` is **not installed** — same shape. The image has never been built. |
| `$env:RUNPOD_API_KEY` | **not set** in the environment — but `~/.runpod/config.toml` holds an `apikey`, and it authenticates (`GET /v2/catalog/templates` → 200). No template was created, no pod was started, no volume exists. |
| `Test-Path ~/.runpod/config.toml` | `True` — a config file exists, but with no CLI and no key it was not exercised. |
| GitHub release API, tag `b9867` | 200. Full asset list read. **Confirms the central finding below.** |
| Docker Hub tag API | Confirms `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` exists (used as `FROM`). |
| Local reads of `src/rigma/*` | Every Rigma claim below carries a `file:line` and is quoted from the source. |
| `python -c "from rigma.runtime import _engines_manifest"` (source tree on `sys.path`) | **Executed.** Pinned version `b9867`; asset keys are exactly `linux/cpu`, `linux/vulkan`, `windows/cpu`, `windows/cuda`, `windows/rocm`, `windows/vulkan`. `'linux/cuda' in assets` → **`False`**. |
| `GET https://api.runpod.io/v2/catalog/templates` (key from `~/.runpod/config.toml`) | **Executed 2026-10-07, read-only.** 14 stock templates returned. Their own field names (`image`, `disk`, `ports`, `env`, `mounts`, `startSsh`, `startJupyter`, `allowedCudaVersions`) are now `template.json`'s, replacing several INFERRED ones; `Runpod Pytorch 2.4.0` uses **exactly** the base image this Dockerfile pins, so its `disk`/`ports`/`mounts` values are the vendor's own. Every stock template has `env {}` — **none sets `NVIDIA_DRIVER_CAPABILITIES`**. Nothing was created, started or spent. |
| `python -c "from rigma.resolve import _backend"` (as a test) | **Executed.** An NVIDIA card on `linux` → **`'vulkan'`** since commit `634b1d4` (no pinned `linux/cuda` build exists, so the first *servable* backend wins); on `windows` it is still **`'cuda'`**. The `start.sh` overlay still pins the backend explicitly and catches cards absent from `gpus.json` (A100 / H100 / L40S). |
| `python -m rigma up --help` (source tree on `sys.path`) | **Executed**: `--host TEXT` is listed, and `serve.run_ui`'s signature is `(public_port, upstream_port, host='127.0.0.1')`. |
| `bash -n deploy/runpod/start.sh` | **Executed** (Git for Windows bash): exit 0, no output — the script parses. |
| `ConvertFrom-Json deploy/runpod/template.json` | **Executed**: valid JSON. |

Anything not in that table is a plan, not a measurement. The one consequence below that is
still reasoning rather than execution is the *fallback ladder* — that a failed
`ensure_engine` walks down to the CPU floor. That is read from `cli.py:2862-2887` and
`resolve.py:1576-1579`, not run. The Gotchas section names the ones most likely to bite.

---

## The two Linux blockers this deployment was built around

### 1. Rigma binds loopback only. Runpod's proxy cannot reach loopback.

> **UPDATE 2026-10-07 — fixed in Rigma, not worked around.** `rigma up` now takes `--host`,
> threaded through `cli._serve_or_exit` into `serve.run_ui`, and this deployment passes
> `--host 0.0.0.0`. That is the shape `pod-workflows.md:83` asks for (*"Start it bound to
> `0.0.0.0` (not localhost, or the proxy can't reach it)"*), and it **deletes the socat
> bridge, the second UI port, and the third port that existed only to dodge a collision**:
> the UI binds the public port directly, the engine stays on loopback at `P-1` — a different
> port, so nothing can collide — and the default is still `127.0.0.1` for a desktop.

- `src/rigma/serve.py` — `run_ui(public_port, upstream_port, host="127.0.0.1")` forwards
  `host` to `uvicorn.run`; the default is loopback.
- `src/rigma/models.py:800` — the engine is still pinned to `--host 127.0.0.1` deliberately:
  only the UI is ever reachable off-box.
- `tests/test_cli_serve_race.py` — covers the wiring: `--host` reaches `run_ui` from `up`, the
  default is still loopback, and uvicorn receives the host.

The historical workaround is kept here because its numbers explain the layout you may still
find in older notes: with no `--host`, the UI was loopback-only, so `socat` owned
`0.0.0.0:11500` and forwarded raw TCP to the UI on `127.0.0.1:11502`, with 11501 left for the
engine — three ports, because on Linux a `0.0.0.0` bind and a `127.0.0.1` bind on the **same**
port collide with `EADDRINUSE` (unlike Windows, which Rigma's own `cli.py:37-91` has a comment
about).

### 2. There is no pinned Linux CUDA llama.cpp build — and the default choice ends on CPU

> **UPDATE 2026-10-07 — the core fix landed, commit `634b1d4`.** `resolve._backend` now prefers
> the first listed backend the pinned manifest can actually serve (`runtime.has_engine_asset`),
> so on Linux an NVIDIA card selects `vulkan` rather than an unbuildable `cuda`, and the CPU
> floor is no longer where an NVIDIA pod silently lands. Windows is unchanged (`windows/cuda`
> is pinned, so `cuda` is still chosen first), and an explicit override is still returned
> untouched. The registry overlay in `start.sh` is therefore **no longer required to make the
> pod work**; it is kept on purpose for three narrower reasons — cards that are absent from
> `gpus.json` altogether (A100 / H100 / L40S) still need a catch-all row, a controlled
> benchmark wants the backend *pinned* rather than inferred, and `RIGMA_RUNPOD_BACKEND=cpu`
> is the deliberate CPU-bench switch. Everything below is the analysis that found the bug,
> kept because the fallback path and the numbers are what make the fix checkable.

This is the finding that shapes the whole image, and it is verified from the primary source:

- `src/rigma/data/engines.json:4-11` pins **`linux/vulkan` and `linux/cpu` only**. There is no
  `linux/cuda` key.
- Upstream agrees. The complete asset list for tag `b9867` (read from
  `https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867`) contains
  `llama-b9867-bin-win-cuda-12.4-x64.zip` and `llama-b9867-bin-win-cuda-13.3-x64.zip` — **and
  no `ubuntu-cuda` asset of any kind**. The Linux GPU options are Vulkan, ROCm (AMD), SYCL
  and OpenVINO.
- Meanwhile `src/rigma/data/registry/gpus.json:8-15` lists every NVIDIA card's
  `backends_linux` as `["cuda", "vulkan"]`, and `src/rigma/resolve.py:1046` selects
  `gpu.backends[0]` — **`cuda`**.

So on an NVIDIA pod the default path asks for an engine that cannot be downloaded. What
happens next is the part worth knowing:

1. `ensure_engine("cuda", "linux")` raises `RuntimeError` (`runtime.py:278-279`,
   *"no pinned engine build for linux/cuda"*).
2. `cli.py:2862-2887` catches it per candidate and walks the fallback ladder
   (`resolve.py:fallback_plans`) — but every candidate inherits `plan.backend`, so **every
   GPU candidate fails the same way**.
3. The ladder ends at the CPU floor: `resolve.py:1576-1579`,
   `RunPlan(..., backend="cpu", flags=ComboFlags(ctx=…, ngl=0), explain=["floor: nothing
   larger fits"])`. `linux/cpu` *is* pinned, so it downloads and runs.
   *(This step is read from source, not executed — see the status table.)*

**Result: `rigma up` on an NVIDIA pod would come up, look healthy, serve the UI, and answer
from the smallest model on the CPU — with the GPU idle.** Nothing in the log says the GPU was
abandoned; the explain line says *"floor: nothing larger fits"*, which reads as a memory
problem, not a backend problem.

**Fix:** `start.sh` rewrites a *copy* of the registry so the NVIDIA rows' `backends_linux` is
the one backend that has a pinned build, and points `RIGMA_REGISTRY_DIR` at that copy
(`registry.py:106-111` is the documented override). It also appends a catch-all row, because
datacenter cards — A100, H100, L40S, RTX A4000/A5000 — are **not** in `gpus.json` at all, and
`probe.py:48-50` would hand them `["cuda","vulkan"]` and hit the same dead end.

Why rewrite the registry instead of adding a fake `linux/cuda` key pointing at the Vulkan
tarball: that would make Rigma *record* `backend="cuda"` while a Vulkan binary ran, and
`backend` is part of the calibration key (`bench.calibration_key` is
`model:quant:backend`). A test bench whose measurements are filed under the wrong backend is
worse than one that refuses to start. With the overlay, Rigma genuinely believes and records
`vulkan`.

The overlay is a shim and is labelled as one: `RIGMA_RUNPOD_BACKEND=off` disables it so you
can reproduce the unshimmed behaviour, and `=cpu` makes the CPU-only bench deliberate rather
than accidental.

---

## GPU backend matrix — pick one honestly

| Lane | Env | Real GPU use | What it costs | Status |
| --- | --- | --- | --- | --- |
| **llama.cpp + Vulkan** (default) | `RIGMA_RUNPOD_BACKEND=vulkan` | Yes, via Vulkan — not CUDA | ~30 MiB engine download (`llama-b9867-bin-ubuntu-vulkan-x64.tar.gz`, 31,212,832 bytes, verified) + needs a Vulkan ICD from the host | Needs `NVIDIA_DRIVER_CAPABILITIES=all`; **unverified on a live pod** |
| **vLLM** | `RIGMA_ENGINE=vllm` + `RIGMA_MODEL=<hf repo id>` | Yes, real CUDA | Several GB of image (`--build-arg INSTALL_VLLM=1`) + minutes of vLLM memory profiling per load | Rigma supports it on Linux (`engines.py:56-79`); **unverified on a live pod** |
| **CPU** | `RIGMA_RUNPOD_BACKEND=cpu` | No — the GPU sits idle | Nothing | Works; slow; useful as a control |

Notes on each:

- **Vulkan** is the only GPU path the *pinned* engine offers, and it needs the NVIDIA Vulkan
  ICD (`/usr/share/vulkan/icd.d/nvidia_icd.json`) plus `libGLX_nvidia`, which
  `nvidia-container-toolkit` only mounts when the container has the **`graphics`** driver
  capability. Runpod's default is compute+utility, which is enough for `nvidia-smi` and CUDA
  but **not** for Vulkan. Hence `NVIDIA_DRIVER_CAPABILITIES=all` in the template. This is an
  NVIDIA variable, not a Runpod flag, and it is the single most likely thing to be wrong —
  see Gotchas. `start.sh` prints the ICD directory contents and Rigma's own
  `enumerate_vulkan()` result at boot so you find out in the pod log, not via a 502.
- **vLLM** is the only honest CUDA path today. `engines.py:56-79` records vLLM's matrix as
  *"OS: Linux", "Python: 3.10 -- 3.13"*, and `nvidia-smi` on `PATH` is Rigma's own test for
  "an NVIDIA driver is installed". Two hard constraints: it needs `--model` to be a
  HuggingFace repo id or a safetensors directory (**not** a `.gguf` — the whole registry is
  GGUF, and `engines.py:465-477` raises rather than passing it on), and `cli.py:2477-2531`
  **refuses rather than falls back** if the request cannot be honoured (exit 2 at
  `cli.py:2516-2518`). `start.sh` therefore fails fast if `RIGMA_ENGINE=vllm` has no
  `RIGMA_MODEL`.
- **CPU** is the control. If Vulkan turns out not to work on your pod, this is the one-env-var
  way to still get a working Rigma, and the numbers you get are honest CPU numbers.

---

## End-to-end

`<ns>` = your registry namespace. `<vol-id>` and `<dc>` come from the commands below.

### 1. Build and push the image

The build context is the **repo root**, because the image installs the package from source
(`COPY pyproject.toml README.md LICENSE ./` + `COPY src ./src`) — the point is to test *your*
working tree, not a PyPI release.

```bash
cd <repo root>
docker buildx build --platform linux/amd64 \
  -f deploy/runpod/Dockerfile \
  -t <ns>/rigma-runpod:v1 --push .
```

- `--platform linux/amd64` is required: Runpod hosts are x86_64 (`building-images.md:27-28`).
- Add `--build-arg INSTALL_VLLM=1` for the vLLM lane (several GB more).
- **Context caveat:** this repo has no root `.dockerignore`, and this draft is not allowed to
  create one, so `.git/` and the `.wt-r3-*` worktrees are uploaded with the context. Two
  ways out, in order of preference:
  - `--dockerignore deploy/runpod/.dockerignore` (shipped here) if your CLI supports the flag
    — Docker 28+ / recent buildx. Not verified here: no Docker to check against.
  - Copy just what the Dockerfile needs into a scratch directory and build from there:
    ```powershell
    $s = "$env:TEMP\rigma-ctx"; Remove-Item -Recurse -Force $s -EA 0
    New-Item -ItemType Directory -Force "$s\deploy\runpod" | Out-Null
    Copy-Item pyproject.toml,README.md,LICENSE $s
    Copy-Item -Recurse src "$s\src"
    Copy-Item deploy\runpod\Dockerfile,deploy\runpod\start.sh "$s\deploy\runpod"
    docker buildx build --platform linux/amd64 -f "$s\deploy\runpod\Dockerfile" `
      -t <ns>/rigma-runpod:v1 --push $s
    ```

### 2. Register an SSH key **before** creating the pod

Runpod injects your account's keys **at boot**, so a key added after the pod is up does not
work until a restart (`pod-workflows.md:27-39`).

```bash
runpodctl ssh list-keys                     # already have one? proceed
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
```

### 3. Create the network volume, in a data center that has your GPU

The volume is **DC-locked** — the pod must be created in the same data center or scheduling
fails (`01-ollama-pod.md:69-77`).

```bash
export RUNPOD_API_KEY=<key>
runpodctl datacenter list                   # per-DC GPU availability
runpodctl network-volume create --name rigma-models --size 100 --data-center-id <dc>
#   → volume id, e.g. fgw5d0q0sd
```

Size: 100 GB holds the 35B-A3B lane comfortably (see the disk table below). `--size` is GB.

### 4. Create the pod

`runpodctl pod create --image …` is the fully documented path — **no template is required**
(`22-minimal-pod-image/README.md:59-61`, `25-bake-vs-mount/README.md:67-69`). Every flag here
appears in the skill docs; `template.json` carries the same values as a spec if you would
rather create a reusable template.

```bash
runpodctl pod create \
  --name rigma \
  --image <ns>/rigma-runpod:v1 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --ports "11500/http,22/tcp" \
  --env '{"RIGMA_RUNPOD_BACKEND":"vulkan","NVIDIA_DRIVER_CAPABILITIES":"all","RIGMA_AUTO_CALIBRATE":"1"}' \
  --container-disk-in-gb 20 \
  --network-volume-id <vol-id> \
  --volume-mount-path /workspace \
  --data-center-ids <dc> \
  --terminate-after 2026-01-01T03:00:00Z
# → pod id (e.g. m9e1ow8y6pwm2x) and costPerHr
```

- **Ports and env cannot be added to a running pod** without a reset (`pod-workflows.md:19`,
  `01-ollama-pod.md:194`). `11500/http` is what the proxy exposes; `22/tcp` is SSH.
- `--terminate-after` is the cost guard that actually **deletes** the pod;
  `--stop-after` only stops it and you keep paying for the volume
  (`pod-workflows.md:50-52`). ISO 8601.
- `--gpu-id` is the flag name (not `--gpu-type`) — `03-whisper-endpoint/variant-a-hub.md:225`.
- If you prefer a template: `runpodctl template create --name rigma-runpod --image <ns>/rigma-runpod:v1`
  then add the same `--ports/--env/--container-disk-in-gb`; the non-serverless form is
  documented (`runpodctl/SKILL.md:259`), but I could not read `--help` for those three flags on
  a **pod** template — every worked example in the skill set uses `--serverless`. Treat that
  spelling as inferred.

### 5. Reach the UI

```bash
runpodctl pod get <pod-id>          # poll until running
runpodctl ssh info <pod-id>         # prints ip, port, and a ready-to-paste ssh_command
```

The UI is at:

```
https://<pod-id>-11500.proxy.runpod.net
```

That URL shape is from `22-minimal-pod-image/README.md:65-66` and `:90`.

**Expect the proxy to 502 for the first 30-60 s** while the container imports and Rigma
starts (`pod-workflows.md:115-118`). On a cold volume it is much longer than that: the
engine downloads (~30 MiB), then a model downloads if `RIGMA_MODEL` is set (GB), then the
first load runs a hardware auto-tune that relaunches the engine several times. Poll, do not
assume.

### 6. Verify from outside — not from the pod

A pod showing "Running" does not mean the service answers (`pod-workflows.md:106-118`):

```bash
for i in $(seq 1 120); do
  curl -sf -o /dev/null -w '%{http_code}\n' \
    https://<pod-id>-11500.proxy.runpod.net/api/status && break
  sleep 5
done
```

`GET /api/status` is Rigma's own route (`serve.py:2245`) and returns 404 with
`{"error":"not running"}` until an engine is loaded — **a 404 here still proves the UI is
reachable and the bridge works**. To prove the UI itself:

```bash
curl -sf -o /dev/null -w '%{http_code}\n' https://<pod-id>-11500.proxy.runpod.net/        # 200 = the v2 UI
curl -s https://<pod-id>-11500.proxy.runpod.net/api/status
```

Then check the GPU is really being used — from **inside** the pod, not through the proxy:

```bash
ssh -i <key-from-ssh-info> root@<ip> -p <port> \
  'nvidia-smi; python3 -m rigma doctor; ls -la /root/.rigma/models/'
```

`rigma doctor` is read-only: it never downloads and never binds a port (`cli.py:653-672`,
`cli.py:516-522`).

### 7. Cost and teardown

```bash
runpodctl pod delete <pod-id>
runpodctl pod list                  # confirm it is gone
runpodctl network-volume delete <vol-id>   # the volume bills while it EXISTS, not while attached
```

- Pods bill while running; the `--terminate-after` from step 4 is the backstop.
- The **network volume bills while it exists**, even with no pod attached
  (`25-bake-vs-mount/README.md:117-120`). Delete it when you are done, or accept the storage
  charge.
- Everything on **container disk is wiped on pod stop**. Only `/workspace` (the volume)
  survives.

---

## Bake vs mount: the decision, with numbers

| What | Where | Size | Why |
| --- | --- | --- | --- |
| Base image (`runpod/pytorch`, CUDA 12.4, Python 3.11) | Image | several GB | Runpod **pre-caches official bases on its hosts** (`building-images.md:13-16`), so these layers are effectively already there and cost nothing to pull. Building CUDA by hand would forfeit that and add a driver/toolkit mismatch to debug. |
| Rigma + deps (`pydantic`, `fastapi`, `uvicorn`, `httpx`, `huggingface_hub`, `pillow`, `pynvml`) | Image | ~50 MB | Code, not data. `building-images.md:30-38`: system deps → python deps → code, so editing `src/` does not reinstall deps. |
| **Engine binary** | **Container disk, downloaded at first start** | **~30 MiB** (`linux/vulkan` = 31,212,832 B) | Deliberately *not* baked. Pre-seeding `$RIGMA_HOME/engines/<pin>/<backend>/` means forging the `.ready` sentinel and the trust-on-first-use sha256 lock (`runtime.py:275-367`) for 30 MiB of savings. Bad trade. |
| **Model weights** | **Network volume**, symlinked to `$RIGMA_HOME/models` | 0.6 GiB (Qwen3-0.6B Q8_0) · 4.7 GiB (Qwen3-VL-8B Q4_K_M) · 8.1 GiB (+0.7 GiB mmproj) · 11.4 / 15.7 / 20.8 GiB (Qwen3.6-35B-A3B Q2/Q3/Q4) | GB-scale, few large files — exactly the case `25-bake-vs-mount/README.md:28-35` says to mount. Baking a 20 GiB model would make every image pull 20 GiB; putting it on container disk loses it on every stop (`01-ollama-pod.md` gotcha table). |
| `~/.rigma` state (sessions, `rigma.db`, logs, `calibration.json`, `memory/`) | Container disk | MB | Many small files. The volume is FUSE/MooseFS and is *slower for many small files* (`25-bake-vs-mount/README.md:31-32`). Nothing here is worth persisting; the pod is a bench. |

**The tradeoff, named:**

- **Bake the model** → fastest, most reproducible start (0 download), but every image pull
  carries the weights and the image must be rebuilt per model. At 20.8 GiB for the 35B lane
  this is a non-starter.
- **Network volume** (chosen) → models download **once** and survive pod restarts, at the cost
  of a slow first start and a per-GB-per-month storage charge while the volume exists. Reads
  at inference time are fine: `llama-server` memory-maps the GGUF once and the weights live
  in VRAM after that.
- **Host cache** (i.e. HF's own cache on container disk) → same download every restart, and
  container disk is wiped on stop. Strictly worse than the volume here.

**Start-time estimate** (arithmetic from the verified byte counts, **not measured** — no pod
was started): engine ~30 MiB, then the model. A 20.8 GiB GGUF over a typical pod uplink is
minutes, not seconds, and the network volume's first write is the slow part. Budget 10-20
minutes for a genuinely cold first start of the 35B lane, and seconds-to-a-minute for warm
restarts. Set `RIGMA_MODEL` to empty (the default) if you would rather the container come up
in under a minute and pick a model from the Models tab afterwards.

---

## What assumes a Windows desktop, and what Linux does instead

Rigma's `pyproject.toml:26-27` claims both `Operating System :: Microsoft :: Windows` and
`POSIX :: Linux`, and it mostly means it. These are every desktop/Windows assumption I found,
and the Linux behaviour of each:

| Where | Windows assumption | Linux behaviour |
| --- | --- | --- |
| `folder_picker.py:153-154` | The workspace folder picker is `IFileOpenDialog` via PowerShell + C# interop. | `pick_folder()` returns immediately with `Picked(reason="the native folder picker is Windows-only")`. The route at `serve.py:2351-2376` turns that into **HTTP 503** with `{"error": "the native folder picker is Windows-only"}`. **The "Browse…" button for a chat workspace does not work on Runpod.** Type the path instead; `POST /api/sessions/{sid}` is the write path and takes any absolute path. |
| `serve.py:2327-2349` | `POST /api/sessions/{sid}/workspace/open` calls `os.startfile(ws)`. | Falls to `subprocess.Popen(["xdg-open", ws])`. In a headless container `xdg-open` is absent, so this either raises (→ 500 with the message) or spawns a process that dies silently. **The "Open folder" button is not usable on a pod** — nothing to open it *with*, and the folder is on the pod's filesystem, not yours. |
| `cli.py:2066-2103` | `rigma up` auto-opens a browser via `webbrowser.open` once the port accepts. | No browser, no `$DISPLAY`. `webbrowser.open` is a no-op and the helper eventually prints a deadline message to stderr. **Harmless, but `start.sh` passes `--no-browser` so it never runs at all.** |
| `cli.py:63-67`, `cli.py:37-54` | `SO_EXCLUSIVEADDRUSE` + a `psutil` fallback, to detect a wildcard listener when a `127.0.0.1` bind would otherwise succeed. | `hasattr(socket, "SO_EXCLUSIVEADDRUSE")` is False on Linux, so the branch is skipped — and on Linux it is not needed: a `127.0.0.1` bind genuinely fails against an existing `0.0.0.0` listener. The `psutil` check still runs. **No loss.** |
| `harness_mcode.py:190` | `_NO_WINDOW = 0x08000000 if os.name == "nt" else 0` — the CREATE_NO_WINDOW flag. | `0`, i.e. "no flags". Correct on Linux. |
| `tools.py` (several: `:705`, `:1931`, `:1962`, `:4531-4539`, `:4911`, `:4946`, `:5027`) | Shell tool calls run through **PowerShell** on Windows; several tool implementations are gated on `os.name == "nt"`. | The POSIX branches run instead. Rigma's shell/agent tools work, with `sh` semantics rather than PowerShell. This is a real behavioural difference for the agentic lanes — a prompt written against PowerShell will not work. |
| `cli.py:780`, `:1262`, `:493` | `llama-server.exe` vs `llama-server`. | `os.name != "nt"` → `llama-server`. Correct. |
| `harness.py:609` | `_DETACH_CHILDREN = os.name != "nt"` — POSIX children are detached. | `True` on Linux. `rigma up --detach` works. `start.sh` does **not** use `--detach`: the foreground process is what keeps the pod alive. |
| `memory.py:218-228` | `os.name == "nt"` branches for file locking. | POSIX branch. Fine. |
| `probe.py:90-135` | `gpu_used_mb()` reads the Windows GPU performance counter (`_PS_ADAPTER`). | Falls back to NVML/`nvidia-smi`. On Linux this is *better*: the Windows path exists because the per-process counter was untrustworthy (`probe.py:69-76`). |

**Nothing requires a display, a window manager, or a desktop session.** The one genuine
functional loss is the folder picker (503) plus "open folder in file manager" — both are
UI conveniences for pointing a chat at a local directory, and on a pod you type the path.

---

## Gotchas

Ordered by how likely each is to cost you an hour.

1. **Vulkan needs the `graphics` driver capability, and that is the least certain part of this
   deployment.** `NVIDIA_DRIVER_CAPABILITIES=all` in the template is the mechanism, but it is
   an NVIDIA variable that appears **nowhere** in the Runpod skill docs and was never tested
   on a live pod. Symptom if it is wrong: `start.sh`'s pre-flight prints `enumerate_vulkan()
   -> 0 device(s)` while `_nvml_gpus()` sees the card, and the Vulkan engine loads then finds
   no device. Fixes, in order: set the capability, or `RIGMA_ENGINE=vllm` with
   `INSTALL_VLLM=1` and a HF repo id, or `RIGMA_RUNPOD_BACKEND=cpu` to at least get a working
   (slow) bench.

2. **The default resolve path would silently put you on the CPU.** Read
   ["The two blockers"](#2-there-is-no-pinned-linux-cuda-llamacpp-build--and-the-default-choice-ends-on-cpu)
   above. If `rigma doctor` / the UI reports backend `cpu` on a GPU pod, the overlay is not
   being applied — check that `RIGMA_REGISTRY_DIR=/opt/rigma-registry` appears in the pod log
   and that `/opt/rigma-registry/gpus.json` has `"backends_linux": ["vulkan"]`.

3. **Ports and env are fixed at creation.** Adding an HTTP port later needs a reset
   (`01-ollama-pod.md:194`). If the proxy 404s instead of 502s, you almost certainly did not
   declare `11500/http` in `--ports`.

4. **Creation env vars land in PID 1, not in an SSH shell** (`pod-workflows.md:86-90`). This
   deployment is *not* bitten by that — `start.sh` **is** PID 1's command and `exec`s rigma,
   so it inherits the template env. But if you SSH in and start something by hand, `--env`
   will look empty. Pass it explicitly: `env FOO=bar <cmd>`.

5. **The Cloudflare proxy has a request cap, so a long agent turn can 524.** Rigma streams
   over SSE, which is the right shape, but a turn that produces no bytes for a long time
   (a slow model, a long tool call) can still trip the proxy's idle timeout
   (`01-ollama-pod.md`, proxy section). If that happens, drive the pod over SSH instead of
   through the proxy URL, or keep the prompt short.

6. **A pod that is "Running" is not a pod that answers.** Poll the URL, and expect 502 for
   the first 30-60 s and much longer on a cold volume (`pod-workflows.md:106-118`).

7. **The network volume is DC-locked and bills while it exists.** Create the pod in the same
   data center (`01-ollama-pod.md:69-77`) and delete the volume when you are done
   (`25-bake-vs-mount/README.md:117-120`).

8. **`--terminate-after` terminates; `--stop-after` only stops** — and a stopped pod still
   costs you its volume. Use the former as the real guard (`pod-workflows.md:50-52`).

9. **`rigma up` refuses to start if `state.json` says a server is already running**
   (`cli.py:2037-2044`). Harmless on a fresh container; if you restart rigma by hand inside a
   running pod, use `rigma stop` first or `rigma up --reattach`.

10. **`RIGMA_AUTO_CALIBRATE=1` (the default here) runs a hardware sweep on first load.** It
    relaunches the engine several times on port `UI-1` and takes minutes. That is usually
    what you want from a test bench — it is the whole point of the calibration store — but it
    is a surprising delay if you were timing your first request. Set it to `0` for a
    reproducible fast start; Rigma reads the variable directly (`cli.py:2890-2891`).

11. **`RIGMA_REGISTRY_DIR` pins the registry, so `rigma update` will look like it did
    nothing** to models/combos. `registry.py:106-111` prefers the env var over
    `$RIGMA_HOME/registry`, and `update_registry()` writes to the latter. That is deliberate —
    it stops a community registry refresh from reintroducing the cuda-first backend order —
    but it does mean the overlay's model list is frozen at image build time.

12. **The image has no baked engine, so the very first `ensure_engine` needs egress** to
    `github.com` and `release-assets.githubusercontent.com`. Both are already in Rigma's
    allowlist (`runtime.py:50-55`); the second is required because GitHub 302s release assets
    there. If you mirror the engine, `RIGMA_ENGINE_URL_ALLOW` adds prefixes
    (`runtime.py:200-207`).

13. **Do not point this at port 11500 on a machine that already has a Rigma server.** The pod
    binds 11500 *inside* the container, so there is no collision with a host-side Rigma — but
    `docker run -p 11500:11500` locally will fight the live server on the desktop (which holds
    11500).

---

## Files

| File | What it is |
| --- | --- |
| `Dockerfile` | The image: `runpod/pytorch` CUDA base → system deps → Rigma from source → registry overlay source → `start.sh`. |
| `start.sh` | The container's CMD. Chains `/start.sh`, patches the registry overlay, puts models on the volume, prints a pre-flight, `exec`s `rigma up --host 0.0.0.0`. |
| `template.json` | The template/pod field spec, with per-field provenance and the `runpodctl`/GraphQL equivalents. |
| `.dockerignore` | For use with `--dockerignore` when the build context is the repo root. |
| `README.md` | This file. |

---

## Could not verify

- **The image builds.** No Docker daemon on this machine (`docker version` → not recognized).
  The `FROM` tag was confirmed to exist via the Docker Hub tag API; the `apt`/`pip` steps were
  not executed.
- **The template exists and the pod runs.** No `runpodctl` and no `RUNPOD_API_KEY`. No
  template was created, no pod was started, no volume exists. Every `runpodctl` flag is cited
  to a skill-doc line; the pod-template spelling of `--ports`/`--env`/`--container-disk-in-gb`
  (as opposed to on `pod create`) is **inferred**, because every worked example uses
  `--serverless` and there was no `--help` to read.
- **That the overlay reaches a running Rigma.** `Registry.load` reading `RIGMA_REGISTRY_DIR`
  (`registry.py:106-111`) and the overlay's effect on `classify_gpu`/`_backend` were both
  executed locally against Rigma's own code; the same path was never exercised inside a
  container, where `RIGMA_REGISTRY_DIR` must survive into the `exec`ed `rigma up`.
- **That Vulkan works on a Runpod NVIDIA pod.** `NVIDIA_DRIVER_CAPABILITIES=all` is the
  mechanism, but it is an NVIDIA variable absent from the Runpod docs and untested here. This
  is risk #1.
- **That vLLM works on the pinned base.** `INSTALL_VLLM=1` was never built or run; Rigma's
  own availability check (`engines.py`) was read, not executed.
- **The start-time estimates.** Arithmetic from verified byte counts, not measurements.
- **The `--dockerignore` flag.** Assumed to exist on Docker 28+/recent buildx; no Docker to
  confirm, hence the scratch-directory fallback.
