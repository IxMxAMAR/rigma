# Rigma

Hardware-aware local LLM deployment for consumer machines: `rigma up` probes your GPU/RAM,
picks the community-verified best model + quant + flag combo for your exact hardware,
downloads a pinned llama.cpp build and the model, and serves an OpenAI-compatible endpoint —
no knob-mashing required.

Unlike generic runners, Rigma applies the tuning that actually matters per machine:
MoE expert offload (`--n-cpu-moe`) sized to your RAM, architecture-aware KV-cache policies
(e.g. `q8_0` K-cache floor on DeltaNet-family models), backend selection per GPU generation
(e.g. Vulkan over ROCm on RDNA4), flash attention, and session persistence
(`--slot-save-path`) on by default. Every decision is auditable: `rigma plan --explain`
shows the arithmetic and sources.

## Quickstart (pre-alpha)

```powershell
git clone https://github.com/IxMxAMAR/rigma && cd rigma
pip install -e .
rigma up            # probe -> resolve -> download -> serve at http://127.0.0.1:11500/v1
```

Commands: `rigma doctor` (what Rigma sees), `rigma plan --explain` (what it would run and why),
`rigma models` (what fits your machine), `rigma up --use-case coding` (serve).

## Status

Pre-alpha (M1). Verified combos:

| Hardware | Model | Backend | Result |
|---|---|---|---|
| RX 9070 XT 16GB + 16GB RAM (Windows) | Qwen3.6-35B-A3B UD-Q3_K_XL, ctx 32K, n_cpu_moe 10 | Vulkan (llama.cpp b9867) | pending first full run |

Design: `docs/superpowers/specs/2026-07-03-rigma-design.md`. License: Apache-2.0.
RAG integration (via [Raggity](https://github.com/IxMxAMAR/raggity), AGPL-3.0, separate process) lands in M4.
