# syntax=docker/dockerfile:1
#
# Rigma on a Runpod GPU pod — image definition.
#
# WHY A PREBUILT BASE INSTEAD OF BUILDING CUDA
# --------------------------------------------
# FROM is an official Runpod PyTorch image, pinned to an exact tag that was
# confirmed to exist on Docker Hub while writing this file (see README
# "Could not verify"). Two reasons, both from the Runpod docs
# (runpod-usage/reference/building-images.md:9-28):
#   * torch/CUDA already match Runpod hosts, so there is no driver/toolkit
#     mismatch to debug, and nothing compiles CUDA here;
#   * Runpod pre-caches official base images on its hosts, so the base layers
#     are effectively already on the machine — only the layers added below are
#     pulled. A hand-built CUDA base throws both away.
# The tag is pinned (not `latest`) so a rebuild is reproducible.
#
# The base also ships /start.sh, which is what makes a pod usable (sshd +
# web terminal). start.sh below chains it — see
# building-images.md:56-93, "Don't clobber the base image's startup".
#
# WHY THIS IMAGE DOES *NOT* BAKE IN AN ENGINE OR A MODEL
# ------------------------------------------------------
# Rigma downloads its pinned llama.cpp build into $RIGMA_HOME/engines/<pin>/
# <backend>/ with a `.ready` sentinel and a trust-on-first-use sha256 lock
# (src/rigma/runtime.py:275-367). Pre-seeding that layout in the image means
# forging the lock file, and the pinned Linux GPU asset is only ~30 MiB
# (llama-b9867-bin-ubuntu-vulkan-x64.tar.gz = 31,212,832 bytes, read from the
# GitHub release API). Not worth the fragility. Models are GB-scale and are the
# case a network volume is for — start.sh symlinks $RIGMA_HOME/models onto it.

ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Set to 1 to add vLLM (the only honest CUDA path for Rigma on Linux today —
# see the README's "GPU backend matrix"). Costs several GB of image.
ARG INSTALL_VLLM=0

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_XET=1

# --- system deps -------------------------------------------------------------
# libvulkan1  : the Vulkan LOADER. Required by src/rigma/probe.py:293-298, which
#               dlopen()s "libvulkan.so.1" to enumerate GPUs. Deliberately NOT
#               mesa-vulkan-drivers: that is lavapipe (a CPU software
#               implementation) and probe.py:320 filters VK_DEVICE_TYPE_CPU, so
#               installing it would only add a decoy ICD. The NVIDIA ICD must
#               come from the host driver via nvidia-container-toolkit.
# ca-certificates/curl : TLS for the engine + model downloads and for the
#               readiness poll.
# No socat: it used to bridge 0.0.0.0:11500 -> 127.0.0.1:11502 because the UI
# was loopback-only. `rigma up --host 0.0.0.0` binds the public port directly
# (cli.py `_serve_or_exit` -> serve.run_ui(port, port - 1, host)), which is also
# the shape Runpod's pod workflow asks for.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl libvulkan1 \
 && rm -rf /var/lib/apt/lists/*

# --- python ------------------------------------------------------------------
# Layer order follows building-images.md:30-38 (system deps, then python deps,
# then code) so editing src/ does not reinstall dependencies.
WORKDIR /opt/rigma-src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3 -c "import sys; assert sys.version_info[:2] >= (3, 11), sys.version" \
 && python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir ".[nvidia]"

# pynvml (the [nvidia] extra) is NOT optional here: probe.py:411 falls back to
# _nvml_gpus() when Vulkan enumeration returns nothing, and probe.py:370 imports
# pynvml lazily and returns [] on ImportError. Without it, a pod with no working
# Vulkan ICD reports ZERO GPUs and Rigma plans for CPU.

RUN if [ "${INSTALL_VLLM}" = "1" ]; then \
      python3 -m pip install --no-cache-dir vllm; \
    fi

# --- the registry overlay source --------------------------------------------
# As of commit 634b1d4 `resolve._backend` prefers the first backend the pinned
# manifest can actually serve, so an NVIDIA card on Linux picks vulkan by itself
# and this overlay is no longer required to make the pod work. It is kept for
# the catch-all row (cards absent from gpus.json — A100, H100, L40S — get
# probe.py:48's generic ["cuda","vulkan"]) and so a benchmark can PIN the
# backend rather than infer it.
#
# Background, since it explains the row below: the packaged registry
# (src/rigma/data/registry/) lists every NVIDIA card's backends_linux as
# ["cuda", "vulkan"], and the pinned engine manifest has NO linux/cuda asset
# (src/rigma/data/engines.json pins linux/vulkan and linux/cpu only, and
# upstream publishes no Linux CUDA build for b9867 — verified against the GitHub
# release API). Taking element [0] therefore asked for an engine that cannot be
# downloaded, ensure_engine raised (runtime.py:278-279), and cli.py's fallback
# ladder walked down to the CPU floor.
#
# start.sh rewrites this COPY per $RIGMA_RUNPOD_BACKEND and points
# RIGMA_REGISTRY_DIR at it. Copying here (rather than generating at boot) means
# the only thing that changes at runtime is the one field being overridden.
RUN python3 -c "\
import os, shutil, rigma; \
src = os.path.join(os.path.dirname(rigma.__file__), 'data', 'registry'); \
shutil.rmtree('/opt/rigma-registry', ignore_errors=True); \
shutil.copytree(src, '/opt/rigma-registry'); \
print('registry overlay seeded from', src)"

# --- entrypoint --------------------------------------------------------------
# Pattern 2 from building-images.md:81-93: a custom CMD on a runpod/pytorch base
# MUST invoke /start.sh, or the pod comes up with no SSH and no web terminal.
COPY deploy/runpod/start.sh /opt/rigma-runpod/start.sh
RUN chmod +x /opt/rigma-runpod/start.sh

# Documentation only — Runpod takes the exposed ports from the template/pod
# creation call ("ports are set at creation": pod-workflows.md:19,23-25).
EXPOSE 11500 22

CMD ["/opt/rigma-runpod/start.sh"]
