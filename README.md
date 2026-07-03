# Rigma

Hardware-aware local LLM deployment for consumer machines: `rigma up` probes your GPU/RAM,
picks the community-verified best model + quant + flag combo for your exact hardware,
downloads a pinned llama.cpp build and the model, and serves an OpenAI-compatible endpoint —
no knob-mashing required.

**Status:** pre-alpha, M1 in progress. See `docs/superpowers/specs/2026-07-03-rigma-design.md`.
