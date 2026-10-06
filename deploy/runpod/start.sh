#!/usr/bin/env bash
# Rigma on a Runpod GPU pod — container entrypoint.
#
# What it does, in order:
#   0. chains the base image's /start.sh (sshd + web terminal) in the background
#   1. validates the port plan
#   2. points Rigma's registry at an overlay whose NVIDIA backends_linux is the
#      one backend that actually has a pinned engine build (see the Dockerfile)
#   3. puts $RIGMA_HOME/models on the network volume and checks it really is one
#   4. prints a loud pre-flight (GPU, Vulkan, Rigma's own doctor) so a broken
#      driver is visible in the pod log instead of surfacing as a 502
#   5. execs `rigma up --host 0.0.0.0` in the foreground (the pod's long-lived
#      process)
#
# ENV VARS THIS SCRIPT READS (all optional)
#   RIGMA_UI_PORT        default 11500  the port the UI binds AND the proxy reaches
#   RIGMA_HOST           default 0.0.0.0  the interface the UI binds
#   RIGMA_RUNPOD_BACKEND default vulkan vulkan | cpu | off
#   RIGMA_MODEL          default ""     a registry slug, or an HF repo id for vllm
#   RIGMA_ENGINE         default llamacpp  llamacpp | vllm
#   RIGMA_MODELS_DIR     default /workspace/rigma/models
#   RIGMA_HOME           Rigma's own; default $HOME/.rigma
#   RIGMA_AUTO_CALIBRATE read directly by cli.py:2891; "0" disables the sweep
#
# Port arithmetic:
#   rigma up --port P --host H  =>  UI on H:P, engine on 127.0.0.1:(P-1)
#   (cli.py `_serve_or_exit` -> serve.run_ui(port, port - 1, host);
#    models.py:800 pins llama-server to `--host 127.0.0.1` regardless)
#
#   ONE port, not three. Until `--host` existed the UI was loopback-only, so the
#   only way to reach it was a raw-TCP socat bridge on a second port, with a
#   third left free because on Linux a 0.0.0.0 bind and a 127.0.0.1 bind on the
#   SAME port collide. `rigma up --host 0.0.0.0` removes both: the UI binds the
#   public port directly, and the engine keeps to loopback on P-1, which is a
#   DIFFERENT port and therefore never collides. This is also the shape Runpod's
#   own pod workflow asks for — "bind 0.0.0.0 and declare the port"
#   (runpod-usage/reference/pod-workflows.md).
#
# NOTE on env-var visibility: pod-workflows.md:86-90 warns that creation env
# vars land in PID 1 and are ABSENT from an SSH login shell. That is why a
# service launched over SSH binds loopback and the proxy 502s. This script is
# PID 1's own command, so it DOES see the template env — and so does the
# `exec`ed rigma below. Nothing here needs to be re-passed by hand.

set -euo pipefail

log() { printf '[rigma-runpod] %s\n' "$*"; }
die() { printf '[rigma-runpod] FATAL: %s\n' "$*" >&2; exit 2; }

# --- 0. base image startup ---------------------------------------------------
if [ -x /start.sh ]; then
  log "chaining the base image's /start.sh (sshd, web terminal) in the background"
  /start.sh &
  sleep 2
else
  log "WARNING: /start.sh not found — no SSH, no web terminal. You can still"
  log "         reach the UI through the proxy port, but you cannot exec in."
fi

# --- 1. ports ----------------------------------------------------------------
UI_PORT="${RIGMA_UI_PORT:-11500}"
HOST="${RIGMA_HOST:-0.0.0.0}"

is_port() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }

is_port "$UI_PORT" || die "RIGMA_UI_PORT='$UI_PORT' is not a port number (1-65535)"

ENGINE_PORT=$((UI_PORT - 1))
[ "$ENGINE_PORT" -ge 1 ] || die "RIGMA_UI_PORT=$UI_PORT leaves no room for the engine port (UI-1)"

log "ports: UI $HOST:$UI_PORT, engine 127.0.0.1:$ENGINE_PORT"
if [ "$HOST" = "0.0.0.0" ]; then
  log "the UI is bound to EVERY interface so Runpod's proxy can reach it. The"
  log "  proxy URL (https://<pod-id>-$UI_PORT.proxy.runpod.net) is public to"
  log "  anyone who knows the pod id, and Rigma's UI has no login of its own."
else
  log "WARNING: RIGMA_HOST=$HOST is not 0.0.0.0, so Runpod's proxy cannot reach"
  log "         the UI and the port will 502. This is only useful for debugging"
  log "         from inside the pod."
fi

# --- 2. registry overlay: make the pinned Linux engine reachable -------------
# Since commit 634b1d4 `resolve._backend` prefers the first listed backend the
# pinned manifest can actually serve, so an NVIDIA card on Linux picks vulkan by
# itself and this overlay is no longer required to make the pod work. It is kept
# deliberately, for three narrower reasons:
#   * probe.py:48 hands cards ABSENT from gpus.json (A100, H100, L40S, RTX
#     A4000/A5000 ...) the generic ["cuda","vulkan"] list; the appended catch-all
#     row classifies them explicitly instead.
#   * a benchmark wants the backend PINNED, not inferred, so the run's
#     calibration key (model:quant:backend) is what the operator chose.
#   * RIGMA_RUNPOD_BACKEND=cpu is the deliberate CPU-only bench, and =off
#     reproduces the unshimmed behaviour.
RIGMA_RUNPOD_BACKEND="${RIGMA_RUNPOD_BACKEND:-vulkan}"
case "$RIGMA_RUNPOD_BACKEND" in
  off)
    log "RIGMA_RUNPOD_BACKEND=off — packaged registry left alone."
    log "  Expect the backend chosen by resolve._backend itself (vulkan on an"
    log "  NVIDIA Linux card, since there is no pinned linux/cuda build). Use"
    log "  this only to reproduce the unshimmed behaviour."
    ;;
  vulkan|cpu)
    python3 - "$RIGMA_RUNPOD_BACKEND" <<'PY'
import json, sys
backend = sys.argv[1]
path = "/opt/rigma-registry/gpus.json"
with open(path, encoding="utf-8") as fh:
    rows = json.load(fh)
touched = []
for row in rows:
    if row.get("vendor") == "nvidia":
        row["backends_linux"] = [backend]
        touched.append(row.get("match"))
rows.append({"match": "", "vendor": "nvidia", "arch": "unknown",
             "backends_linux": [backend]})
with open(path, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=1)
print(f"[rigma-runpod] registry overlay: nvidia rows {touched} -> ['{backend}'], "
      f"plus a catch-all row for cards not in the table")
PY
    export RIGMA_REGISTRY_DIR=/opt/rigma-registry
    log "RIGMA_REGISTRY_DIR=/opt/rigma-registry (so 'rigma update' cannot"
    log "  silently reintroduce the cuda-first backend order)"
    ;;
  *)
    die "RIGMA_RUNPOD_BACKEND must be vulkan|cpu|off, got '$RIGMA_RUNPOD_BACKEND'"
    ;;
esac

# --- 3. models on the volume -------------------------------------------------
RIGMA_HOME_DIR="${RIGMA_HOME:-${HOME:-/root}/.rigma}"
MODELS_DIR="${RIGMA_MODELS_DIR:-/workspace/rigma/models}"
export RIGMA_HOME="$RIGMA_HOME_DIR"

mkdir -p "$RIGMA_HOME_DIR" "$MODELS_DIR"

if [ -e "$RIGMA_HOME_DIR/models" ] && [ ! -L "$RIGMA_HOME_DIR/models" ]; then
  # A real directory (an earlier boot without the volume). Move it onto the
  # volume rather than deleting it — and if the copy fails, keep it and do NOT
  # symlink, so nothing is ever lost to a failed move.
  if cp -an "$RIGMA_HOME_DIR/models/." "$MODELS_DIR/" 2>/dev/null; then
    rm -rf "$RIGMA_HOME_DIR/models"
  else
    log "WARNING: could not copy $RIGMA_HOME_DIR/models to $MODELS_DIR."
    log "         Leaving it in place and NOT symlinking: models stay on"
    log "         container disk and will be wiped when this pod stops."
    MODELS_DIR="$RIGMA_HOME_DIR/models"
  fi
fi

if [ ! -e "$RIGMA_HOME_DIR/models" ]; then
  ln -s "$MODELS_DIR" "$RIGMA_HOME_DIR/models"
fi
log "models: $RIGMA_HOME_DIR/models -> $(readlink -f "$RIGMA_HOME_DIR/models" 2>/dev/null || echo "$MODELS_DIR")"

# Is that actually a network mount? Measured method from golden path 25:
# baked/image storage reports `overlayfs`, a network volume reports `fuse`.
# Without a volume, /workspace is container disk and is wiped on pod stop
# (golden path 01, "Models vanish after stop").
if [ -d /workspace ]; then
  WS_FS="$(stat -f -c %T /workspace 2>/dev/null || echo unknown)"
  log "/workspace filesystem: $WS_FS"
  case "$WS_FS" in
    fuse|nfs|nfs4|cifs|smb2|9p|virtiofs) : ;;
    *) log "WARNING: /workspace is not a network mount ('$WS_FS'). Anything Rigma"
       log "         downloads will be WIPED when this pod stops. Attach a volume:"
       log "         --network-volume-id <id> --volume-mount-path /workspace" ;;
  esac
else
  log "WARNING: /workspace does not exist; models are on container disk and will"
  log "         be wiped on pod stop"
fi

# --- 4. loud pre-flight ------------------------------------------------------
log "python: $(python3 -V 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi present (Rigma's engines.py:193 uses exactly this as its"
  log "  'an NVIDIA driver is installed' test for the vLLM lane)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 \
    | sed 's/^/[rigma-runpod]   gpu: /' || true
else
  log "nvidia-smi MISSING. On a GPU pod this means the NVIDIA container runtime"
  log "  did not inject the driver; Rigma will see no CUDA path."
fi

if [ "$RIGMA_RUNPOD_BACKEND" = "vulkan" ]; then
  LOADER_HITS="$(ldconfig -p 2>/dev/null | grep -c 'libvulkan\.so\.1' || true)"
  log "vulkan loader entries: ${LOADER_HITS:-0}"
  ICDS="$(ls -A /usr/share/vulkan/icd.d 2>/dev/null || true)"
  if [ -n "$ICDS" ]; then
    log "vulkan ICDs present:"
    printf '%s\n' "$ICDS" | sed 's/^/[rigma-runpod]   /'
  else
    log "vulkan ICDs: NONE (no /usr/share/vulkan/icd.d, or it is empty)."
    log "  The NVIDIA ICD ships with the host driver and is only mounted when the"
    log "  container has the 'graphics' driver capability. NO stock Runpod"
    log "  template sets NVIDIA_DRIVER_CAPABILITIES (checked against the live"
    log "  catalog: every GPU template has env {}), so this is not something the"
    log "  platform does for you — see the README's note. Without it the Vulkan"
    log "  engine will load and then find no device."
  fi

  log "asking Rigma's own probe what it sees:"
  python3 - <<'PY' || log "  (the probe raised — see the traceback above)"
try:
    from rigma.probe import enumerate_vulkan, _nvml_gpus
except Exception as exc:                      # pragma: no cover - diagnostic
    print(f"[rigma-runpod] could not import rigma.probe: {exc}")
    raise SystemExit(0)

vulkan = enumerate_vulkan()
nvml = _nvml_gpus()
print(f"[rigma-runpod]   enumerate_vulkan() -> {len(vulkan)} device(s)")
for dev in vulkan:
    print(f"[rigma-runpod]     vulkan: {dev['name']} {dev['vram_mb']} MiB "
          f"type={dev.get('device_type')}")
print(f"[rigma-runpod]   _nvml_gpus()       -> {len(nvml)} device(s)")
for dev in nvml:
    print(f"[rigma-runpod]     nvml:   {dev['name']} {dev['vram_mb']} MiB")

if not vulkan and not nvml:
    print("[rigma-runpod] RIGMA SEES NO GPU. Options, in order of preference:")
    print("[rigma-runpod]   1. give the container the graphics driver capability "
          "(NVIDIA_DRIVER_CAPABILITIES=all) so the NVIDIA Vulkan ICD is mounted;")
    print("[rigma-runpod]   2. RIGMA_ENGINE=vllm with a CUDA image and "
          "RIGMA_MODEL=<hf repo id>;")
    print("[rigma-runpod]   3. RIGMA_RUNPOD_BACKEND=cpu for a deliberate CPU-only "
          "bench (slow, but honest).")
elif not vulkan:
    print("[rigma-runpod] NOTE: NVML sees the card but Vulkan enumerates nothing. "
          "With RIGMA_RUNPOD_BACKEND=vulkan the engine WILL fail to find a device "
          "at launch. Switch to RIGMA_ENGINE=vllm or RIGMA_RUNPOD_BACKEND=cpu.")
PY
fi

log "rigma doctor (read-only: it never downloads, never binds a port):"
python3 -m rigma doctor 2>&1 | sed 's/^/[rigma-runpod]   /' \
  || log "  (doctor exited non-zero — at least one hard failure, see rows above)"

# --- 5. rigma ------------------------------------------------------------------
ENGINE="${RIGMA_ENGINE:-llamacpp}"
ARGS=(--no-browser --host "$HOST" --port "$UI_PORT")

case "$ENGINE" in
  llamacpp)
    if [ -n "${RIGMA_MODEL:-}" ]; then
      log "RIGMA_MODEL=$RIGMA_MODEL — downloading and loading it on start"
      log "  (a cold 35B-A3B UD-Q4_K_XL is ~20.8 GiB of weights; this step takes"
      log "   minutes on a fresh volume)"
      ARGS+=(--model "$RIGMA_MODEL" --yes)
    else
      log "no RIGMA_MODEL — the UI starts with no model loaded; pick one in the"
      log "  Models tab and it downloads, auto-tunes and loads on demand"
    fi
    ;;
  vllm)
    # engines.py:56-79: vLLM is Linux-only, Python 3.10-3.13. cli.py:2477-2531
    # refuses (exit 2 at cli.py:2516-2518) rather than falling back if the
    # request cannot be honoured, so this needs a real CUDA image with vLLM
    # installed (build --build-arg INSTALL_VLLM=1) and a non-GGUF model.
    [ -n "${RIGMA_MODEL:-}" ] \
      || die "RIGMA_ENGINE=vllm needs RIGMA_MODEL=<a HuggingFace repo id or a local safetensors dir>; Rigma's registry is all GGUF and engines.py:465-477 refuses to hand vLLM a .gguf"
    log "RIGMA_ENGINE=vllm — vLLM serves $RIGMA_MODEL on 127.0.0.1:$ENGINE_PORT."
    log "  The llama.cpp fit/bench machinery does not apply on this path."
    ARGS+=(--engine vllm --model "$RIGMA_MODEL" --yes)
    ;;
  *)
    die "RIGMA_ENGINE must be llamacpp|vllm, got '$ENGINE'"
    ;;
esac

log "exec: python3 -m rigma up ${ARGS[*]}"
log "the UI is served on http://$HOST:$UI_PORT — through the proxy that is"
log "  https://<pod-id>-$UI_PORT.proxy.runpod.net"
log "Rigma's own startup can take minutes on a cold volume: engine download"
log "  (~30 MiB), model download (GB), then a first-load hardware auto-tune."

# `python3 -m rigma`, not the `rigma` console script: __main__.py calls the same
# typer app, and this cannot land on a different interpreter than the one pip
# installed into. exec so this is the pod's long-lived process and a pod stop
# (SIGTERM) reaches uvicorn, whose finally block stops the engine (cli.py:2997).
exec python3 -m rigma up "${ARGS[@]}"
