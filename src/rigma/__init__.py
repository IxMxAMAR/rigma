import os

# Disable the HF "xet" transfer backend BEFORE huggingface_hub is imported
# anywhere. huggingface_hub reads HF_HUB_DISABLE_XET into a module constant at
# IMPORT time, so setting it later (e.g. in ensure_model) is too late — xet
# stays on. On this Windows box xet hangs downloads at 0 bytes (reproduced
# 2026-07-06, -14, -18); the classic downloader runs at line rate. This module
# runs first for any `rigma` import, so the flag is set in time.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "0.8.1"
