# Using Rigma as the backend for coding agents

Rigma serves an OpenAI-compatible API at `http://127.0.0.1:11500/v1` (shown in
the UI under ⚙ → Server → agents). Tool calling and structured outputs pass
straight through to llama.cpp — use a **tools-capable model**
(`rigma up --use-case coding` picks one automatically; the Server tab shows the
running model's capabilities).

## aider

    aider --openai-api-base http://127.0.0.1:11500/v1 \
          --openai-api-key local --model openai/qwen3.6-35b-a3b

## Cline / Roo (VS Code)

Settings → API Provider: "OpenAI Compatible" →
Base URL `http://127.0.0.1:11500/v1`, API key `local`, model id = anything
(the running model answers regardless of the id you type).

## Continue (VS Code / JetBrains)

    models:
      - name: rigma
        provider: openai
        apiBase: http://127.0.0.1:11500/v1
        apiKey: local
        model: qwen3.6-35b-a3b

## Notes

- **Two slots, not one** (`--parallel 2 --kv-unified`). This note used to say
  "one request at a time (`--parallel 1`)" and that is no longer true — an agent
  author who designed around forced serialisation would be working around a
  limitation that was removed. One slot carries the conversation and one absorbs
  Rigma's own aux calls (auto-title, compaction, delegate), because with a single
  slot every aux call evicted the conversation's prompt cache and forced a full
  re-prefill of 20–60K tokens before the next real turn. `--kv-unified` keeps it
  one shared pool of size `ctx`, so this costs no extra memory. More than two
  concurrent requests still queue.
- Structured outputs: `response_format`/`json_schema` are honored by the engine.
- Reasoning models stream `reasoning_content`; most agents ignore it safely.
- Keep an eye on the context meter — agent transcripts grow fast. The
  resolver's grow-to-fit already targets the largest context that FITS;
  `rigma up --ctx N` overrides it (clamped to the model's native window, but
  NOT re-fit-checked — oversize values can fail at engine launch).
