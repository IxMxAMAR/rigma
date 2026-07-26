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
pip install rigma
rigma up            # probes your machine, downloads the best model, opens the chat UI
```

That's it — a browser tab opens with a chat connected to your tuned local model, and any
OpenAI-compatible tool can use `http://127.0.0.1:11500/v1`.

Chats persist server-side across restarts — the browser UI lists past sessions in a rail,
renders markdown (fenced code, copy button), and supports regenerate / edit-last. Each
session carries its own system prompt (registry ships sensible defaults per use case —
general, creative, coding — so creative-writing models stay in character from the first
message) and a per-session "use my documents" RAG toggle with inline citations.

## Commands

| Command | What it does |
|---|---|
| `rigma up` | Start everything; opens the chat UI in your browser |
| `rigma chat` | Chat with the running model in the terminal; `--session <id>` resumes a session started in the browser UI |
| `rigma status` | What's running, where |
| `rigma stop` | Stop the model server and UI |
| `rigma models` | What fits your machine |
| `rigma plan --explain` | What `up` would run, with the math |
| `rigma doctor` | What Rigma detects on this machine |
| `rigma update` | Pull the latest [community combo registry](https://github.com/IxMxAMAR/rigma-registry) |
| `rigma bench` | Measure real speed; `--evidence FILE` exports registry-format proof |
| `rigma rag add PATH` | Index a folder into your local knowledge base ([Raggity](https://github.com/IxMxAMAR/raggity) sidecar) |
| `rigma rag ask "..."` | Answer grounded in your documents, with citations, via your tuned model |

`rigma up` flags: `--use-case coding` · `--model SLUG` · `--port 11500` · `--no-browser` ·
`--turbo` (fast download, may hog your bandwidth) · `--yes` · `--dry-run`

## Status

Pre-alpha (M5). `rigma bench` records machine-local calibration that outranks registry
combos on your machine, and a failed launch automatically falls back (smaller quant →
CPU floor) with each step explained. Combos come in two grades: **verified** (benchmarked on real hardware,
evidence attached) and **provisional** (research-seeded fit math — run one and PR your
numbers to [rigma-registry](https://github.com/IxMxAMAR/rigma-registry)). Verified so far:

| Hardware | Model | Backend | Result |
|---|---|---|---|
| RX 9070 XT 16GB + 16GB RAM (Windows) | Qwen3.6-35B-A3B UD-Q3_K_XL, ctx 32K, n_cpu_moe 10 | Vulkan (llama.cpp b9867) | **verified 2026-07-06**: 57.1 t/s gen, 689 t/s prefill @ 4K prompt |

## Methods (set a chat up for the work, then automate it)

A **Method** configures a whole activity in one click — system prompt, sampler
profile, thinking effort, tool posture and a Notes template — and then goes
further: it carries **rules**, **macros** and **workflows** you can author
yourself.

Six ship built in: Coding, Writing a book, Roleplay, Research, Learning,
Organizing files.

- **Macro** — a button above the message box. *Finish chapter* summarises the
  chapter into your story bible and opens the next one.
- **Rule** — either standing guidance folded into the system prompt, or a
  trigger that fires on its own (*after a chapter file changes, remind me to
  update the bible*).
- **Workflow** — a named multi-step sequence.

All three are the same thing underneath: a **step list**. A step is one of
`tool`, `prompt`, `settings`, `note` or `new_chat`, and steps can carry
placeholders — `{{var}}`, `{{last_reply}}`, `{{step:0}}`, `{{ask:Label}}`,
`{{transcript}}`, `{{title_next}}`, `{{selection}}`.

**Make your own:** hit **+ Create method** in the Methods panel. That opens a
chat where the model has only the method-building tools, so it can do nothing
but help you build one — it asks a question, calls a tool, and the method takes
shape beside the chat as it goes. Hit **Save method** when you like it.

**Safety.** A macro that only reads runs straight away. Anything that writes,
moves, runs a command, or lets the model loose with its tools asks first, with
a one-line preview naming the real files — *Finish chapter: will run write_file
on story_bible.md*. Choose **Always allow** and it stops asking for that macro.
That choice is stored on your machine and is never part of a shared method.

**Sharing.** A method is a JSON file in `~/.rigma/methods/`. Export it, send
it, import theirs. Importing never overwrites one of yours and never carries
someone else's "always allow".

## Skills (instructions you pull in mid-chat)

A **Skill** is a block of instructions you write once and drop into any chat by
typing `/name` — no method to apply, no settings to change. Where a Method
configures a whole activity, a Skill is a single reusable briefing: house style,
a checklist, the shape of a file format you keep explaining.

```
/wildcard write me three variants
```

The skill's text goes in front of your ask for that turn only. Manage them in
the **Skills** panel, or write them by hand: each one is a `.md` file in
`~/.rigma/skills/`, so they live in your editor and your git repo like anything
else. `/skill:name` works too, and a message that merely starts with a slash and
names no skill — a path, a lone `/` — is sent exactly as you typed it.

## In the chat

- **Type ahead.** Send a follow-up while a reply is still streaming and it
  queues; the model starts on it the moment the current turn ends. The queue is
  in memory, so it never survives a restart — a prompt waiting behind a
  generation stops meaning anything once the server is gone.
- **Stop keeps what you got.** Stopping a reply aborts the request *and* keeps
  the text already on screen, marked as a partial, instead of discarding it.

## RAG (chat with your documents)

Rigma pairs with [Raggity](https://github.com/IxMxAMAR/raggity) (AGPL-3.0, runs as a separate
low-RAM process — ~300 MB) for grounded, cited answers from your own files through your tuned
local model: `pip install raggity[server]`, then `rigma rag add <folder>` and
`rigma rag ask "..."`. If raggity isn't on PATH, point `RIGMA_RAGGITY_CMD` at it.

License: Apache-2.0.
