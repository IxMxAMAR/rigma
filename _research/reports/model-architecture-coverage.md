# Model architecture coverage for Rigma (llama.cpp + vLLM)

Research date: 2026-09-24. Sources are pinned source files, fetched and read directly; every factual
claim carries a URL. Statements derived rather than read are marked **[inference]**.

Primary sources used throughout:

- llama.cpp `master` architecture table — <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp>
- llama.cpp `b9867` architecture table — <https://github.com/ggml-org/llama.cpp/blob/b9867/src/llama-arch.cpp>
- `master` ggml type enum — <https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h>
- `b9867` ggml type enum — <https://github.com/ggml-org/llama.cpp/blob/b9867/ggml/include/ggml.h>
- multimodal projector table — <https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h>
- multimodal docs — <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
- TTS tool — <https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md>
- server docs — <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- server routes — <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp>
- CLI flags — <https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp>

---

## Verdict

**The architecture list is not Rigma's gap. The gap is role classification and per-family requirements —
and, separately, the engine pin is version-skewed in a way that is already producing a false negative.**

1. **The motivating defect is already fixed in Rigma, and the fix is correct.** `src/rigma/engine_compat.py`
   names `GGML_TYPE_PQ2_0 = 142` as private to a third-party fork (`PrismML-Eng/llama.cpp`, branch `prism`).
   I verified that fork's own header: it defines `GGML_TYPE_PQ2_0 = 142`,
   `GGML_TYPE_PTQ1_0 = 143`, `GGML_TYPE_COUNT = 144`
   (<https://github.com/PrismML-Eng/llama.cpp/blob/prism/ggml/include/ggml.h#L435-L437>). Mainline stops at
   `GGML_TYPE_Q2_0 = 42` / `GGML_TYPE_COUNT = 43`
   (<https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h#L430-L431>). So "needs a newer
   engine" was the wrong diagnosis and Rigma's replacement wording ("NO mainline build can load this file")
   is right.

2. **But Rigma's type table is calibrated to `master`, while the pinned engine is `b9867` — and those
   differ.** `b9867`'s enum ends at `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_COUNT = 42`
   (<https://github.com/ggml-org/llama.cpp/blob/b9867/ggml/include/ggml.h#L431-L432>). `GGML_TYPE_Q2_0 = 42`
   exists **only** on `master`. Rigma's `KNOWN_TYPES` includes `42: "Q2_0"` and sets
   `MAINLINE_MAX_TYPE = 42`, so a Q2_0 model passes Rigma's check and then dies on the pinned build with
   `invalid ggml type 42. should be in [0, 42)`. This is a live false negative, not a hypothetical:
   the model in the original report is named `…-PQ2_0`, and `b9867` is dated **2026-07-03**, not mid-2025
   (`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867`). **The type bound must become a
   property of the engine build, not a module constant.**

3. **Rigma's chat-shaped UI covers roughly half of what llama.cpp actually runs.** Of 153 architecture
   strings on `master`, the great majority are chat/completion LLMs and are in scope. The families that a
   chat UI does *not* fit, and that Rigma currently has no vocabulary for, are: **ASR / audio-input LLMs**
   (`/v1/audio/transcriptions`, needs an audio projector), **TTS** (`qwen3tts`, `pockettts` — new on
   `master`, CLI-only, needs *two* projector files), **embeddings and rerankers** (`--embedding`,
   `--reranking`, `/v1/embeddings`, `/v1/rerank`), and **diffusion LMs** (`dream`, `llada`, `llada-moe`,
   `rnd1`). None of these is a chat model, and all four are reachable from a pinned llama.cpp build.

4. **Two namespaces, not one.** Vision and audio are *not* LLM architectures in llama.cpp. The projector
   architecture is a separate table with 62 entries in `tools/mtmd/clip-impl.h`
   (<https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h#L512-L573>), and it is
   independent of the LLM arch: SmolVLM, Pixtral, Moondream, Idefics3 and Granite-Vision run on `llama`,
   `mistral`, `phi2` or `granite` base LLMs with a vision projector attached
   (<https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md#L54-L103>). **A vision model is
   therefore not identifiable from `general.architecture` alone** — Rigma must key off the presence of an
   `mmproj` / `clip.*` metadata, which `gguf_meta.inspect_gguf` already does.

5. **Out of scope, definitively:** image generation (no diffusion *image* architecture exists in llama.cpp;
   stable-diffusion.cpp is a separate project), Bark, MusicGen, a bare `vocoder` arch, `whisper` as an LLM
   arch, and every mainstream TTS system except Qwen3-TTS, Pocket TTS and the legacy OuteTTS demo
   (Kokoro, Piper, XTTS, F5-TTS, Orpheus, Chatterbox, CSM, Dia — none run on llama.cpp).

---

## 0. Correction to the brief

| Claim in the brief | Reality | Source |
|---|---|---|
| `b9867` = commit `152d337fa`, ~mid-2025 | Published **2026-07-03T14:56:03Z**; commit `152d337fa` is 2026-07-03T13:40:06Z ("spec: support spec-draft-p-min in DFlash (#25246)") | GitHub API `releases/tags/b9867`, `commits/152d337fa` |
| `whisper`, `bark`, `musicgen`, `vocoder`, `stable-diffusion`, `llava`, `clip`, `qwen2audio` are LLM archs to enumerate | **None of these are LLM architecture strings.** `clip` exists only as a dummy for `llama-quantize`; the rest are absent from both builds | `src/llama-arch.cpp` (master and b9867) |
| type `142` is "a ggml tensor type newer than the build's type table" | `142` is not a mainline type on **any** build; it is a fork-private id. `b9867` `COUNT = 42`, `master` `COUNT = 43`, PrismML fork `COUNT = 144` | `ggml.h` (three revisions) |

The latest llama.cpp release at research time is `v0.5.0`, published 2026-09-23
(`https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`).

---

## 1. llama.cpp architecture coverage — exhaustive

### 1.1 The two tables

llama.cpp resolves a model's identity from one GGUF key, `general.architecture`
(<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp> — `LLM_KV_GENERAL_ARCHITECTURE`).
That string indexes `LLM_ARCH_NAMES`, a `std::map<llm_arch, const char*>`; unknown values resolve to
`LLM_ARCH_UNKNOWN` / `"(unknown)"` via `llm_arch_from_string`.

Multimodal **projectors** are a completely separate namespace. `tools/mtmd/clip-impl.h` holds
`PROJECTOR_TYPE_NAMES`, a `std::map<projector_type, std::string>`; the projector's own type string lives in
the mmproj file's metadata, not in `general.architecture`
(<https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h#L511-L582>).

### 1.2 All 153 LLM architecture strings (`master`)

Flat, in source order. `clip` and `(unknown)` are sentinels, not models.

```
clip, llama, llama4, deci, falcon, grok, gpt2, gptj, gptneox, mpt, baichuan, starcoder, refact, bert,
modern-bert, nomic-bert, nomic-bert-moe, neo-bert, jina-bert-v2, jina-bert-v3, eurobert, bloom, stablelm,
qwen, qwen2, qwen2moe, qwen2vl, qwen3, qwen3moe, qwen3next, qwen3vl, qwen3vlmoe, qwen35, qwen35moe,
qwen4exp, phi2, phi3, phimoe, plamo, plamo2, plamo3, codeshell, orion, internlm2, minicpm, minicpm3,
gemma, gemma2, gemma3, gemma3n, gemma4, gemma4-assistant, gemma-embedding, starcoder2, mamba, mamba2,
maple, jamba, falcon-h1, xverse, command-r, cohere2, cohere2moe, dbrx, olmo, olmo2, olmoe, muse-glimmer,
openelm, arctic, deepseek, deepseek2, deepseek2-ocr, deepseek32, deepseek4, chatglm, glm4, glm4moe,
glm-dsa, bitnet, t5, t5encoder, jais, jais2, nemotron, nemotron_h, nemotron_h_moe, exaone, exaone4,
exaone-moe, rwkv6, rwkv6qwen2, rwkv7, arwkv7, granite, granitemoe, granitehybrid, graniteswitch,
granite_swa, chameleon, wavtokenizer-dec, plm, bailingmoe, bailingmoe2, bailingmoe3, dots1, dots3note,
arcee, afmoe, laguna, ernie4_5, ernie4_5-moe, hunyuan-moe, hunyuan-dense, hunyuan_vl, hy_v3, hy_v4,
smollm3, gpt-oss, lfm2, lfm2moe, dream, smallthinker, llada, llada-moe, seed_oss, grovemoe, apertus,
minimax-01, hrm_text, minimax-m2, minimax-m3, cogvlm, rnd1, pangu-embedded, mistral3, eagle3, dflash,
mistral4, paddleocr, mimo2, step35, spark2_5, llama-embed, maincoder, kimi-linear, kimi-k3, talkie,
mellum, nanbeige, qwen3tts, pockettts, (unknown)
```

`b9867` carries **135** of these; `master` carries **153**. Nothing was removed between them.

### 1.3 Grouped by role

Groups below are **[inference]** from the arch name plus the evidence cited per group, *except* where a
group is explicitly marked as sourced from a llama.cpp predicate.

**Hybrid / SSM / recurrent — sourced.** llama.cpp states these directly:
`llm_arch_is_recurrent()` = `mamba`, `mamba2`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7`;
`llm_arch_is_hybrid()` = `jamba`, `falcon-h1`, `plamo2`, `granitehybrid`, `lfm2`, `lfm2moe`, `nemotron_h`,
`nemotron_h_moe`, `qwen3next`, `kimi-linear`, `bailingmoe3`, `kimi-k3`, `qwen35`, `qwen35moe`, `qwen4exp`,
`deepseek4`, `minimax-01`
(<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp> — `llm_arch_is_recurrent`,
`llm_arch_is_hybrid`).

**Diffusion LMs (text, non-autoregressive) — sourced.** `llm_arch_is_diffusion()` = `dream`, `llada`,
`llada-moe`, `rnd1` (same file). These are text generators with a non-causal decode loop, demonstrated by
`examples/diffusion` (<https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion>). **Not image
generation.**

**MoE — partially sourced.** 21 archs have a model class literally named `*moe*`/`*mix*` in
`src/models/models.h` (<https://github.com/ggml-org/llama.cpp/blob/master/src/models/models.h>):
`nomic-bert-moe`, `llada-moe`, `qwen2moe`, `qwen3moe`, `qwen3vlmoe`, `phimoe`, `cohere2moe`, `olmoe`,
`glm4moe`, `nemotron_h_moe`, `exaone-moe`, `granitemoe`, `bailingmoe`, `bailingmoe2`, `bailingmoe3`,
`afmoe`, `ernie4_5-moe`, `hunyuan-moe`, `gpt-oss`, `lfm2moe`, `grovemoe`, `qwen35moe`. Other archs are
MoE in practice but not named so — at minimum `llama4`, `dbrx`, `deepseek2`, `deepseek32`, `deepseek4`,
`minimax-m2`, `minimax-m3`, `qwen3next`, `kimi-k3`, `mimo2`, `step35`, `dots3note` — **[inference]**.

**Vision / multimodal base LLMs — partially sourced.** Documented with pre-quantized GGUF in
`docs/multimodal.md`: `gemma3`, `gemma4`, `qwen2vl` (Qwen2-VL / Qwen2.5-VL), `llama4`, `mistral3`
(Mistral Small 3.1), `internlm2` (InternVL), `qwen3vl`/`qwen3vlmoe`, `hunyuan_vl`, `cogvlm`,
`deepseek2-ocr`, `paddleocr`
(<https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md#L52-L141>). Per-arch docs also exist
for MobileVLM, GLM-Edge, Granite Vision, LLaVA, MiniCPM-o 2.6/4.0 and MiniCPM-V 2.5/2.6/4.0/4.5/4.6
(<https://github.com/ggml-org/llama.cpp/tree/master/docs/multimodal>). Crucially, **many vision models do
not have a vision-specific LLM arch at all** — SmolVLM, Pixtral 12B, Moondream2 and Idefics3 run on
`llama`/`mistral`/`phi2` base archs (`docs/multimodal.md#L60-L96`).

**Audio-input LLMs / ASR — sourced.** `docs/multimodal.md#L105-L141` lists pre-quantized audio GGUF for
Ultravox 0.5 (`llama` base + `ultravox` projector), Voxtral (`mistral3`-family + `voxtral`), Qwen3-ASR
(`qwen3a` projector) and, as mixed-modality, Qwen2.5-Omni / Qwen3-Omni and Gemma 4. There is **no `whisper`
LLM arch**; Whisper appears only as an *encoder implementation* inside mtmd
(<https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/models/whisper-enc.cpp>).

**TTS — sourced, new.** `qwen3tts` and `pockettts` are LLM archs on `master` with model implementations
`src/models/qwen3tts.cpp` and `src/models/pockettts.cpp`, conversion scripts `conversion/qwen3tts.py` and
`conversion/pockettts.py`, and generation/speaker projectors. `wavtokenizer-dec` is a legacy audio-decoder
arch; `talkie` is a 13B text LLM, **not** TTS. See §2.2.

**Embedding / reranker / encoder-only — partially sourced.** Encoder-style archs present:
`bert`, `modern-bert`, `nomic-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`,
`eurobert`, `t5encoder`, `gemma-embedding`, `pangu-embedded`, `llama-embed`. `t5` is the full
encoder-decoder. `granite`, `qwen3` and `gemma` also serve as embedding bases. See §2.4.

**Draft / speculative heads — [inference].** `eagle3` and `dflash` are speculative-decoding draft heads,
not standalone chat models; `gemma4-assistant` is Gemma 4's assistant/draft component
(<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp> — `LLM_ARCH_EAGLE3`,
`LLM_ARCH_DFLASH`, `LLM_ARCH_GEMMA4_ASSISTANT`).

**Everything else** is a dense or MoE decoder-only text LLM suitable for a chat UI: `llama`, `deci`,
`falcon`, `grok`, `gpt2`, `gptj`, `gptneox`, `mpt`, `baichuan`, `starcoder`, `refact`, `bloom`, `stablelm`,
`qwen`, `qwen2`, `qwen3`, `qwen35`, `phi2`, `phi3`, `plamo`, `plamo3`, `codeshell`, `orion`, `minicpm`,
`minicpm3`, `gemma`, `gemma2`, `gemma3n`, `starcoder2`, `xverse`, `command-r`, `cohere2`, `olmo`, `olmo2`,
`openelm`, `arctic`, `deepseek`, `chatglm`, `glm4`, `glm-dsa`, `bitnet`, `jais`, `jais2`, `nemotron`,
`exaone`, `exaone4`, `granite`, `graniteswitch`, `granite_swa`, `chameleon`, `plm`, `dots1`, `arcee`,
`laguna`, `ernie4_5`, `hunyuan-dense`, `hy_v3`, `hy_v4`, `smollm3`, `smallthinker`, `seed_oss`, `apertus`,
`mistral4`, `maincoder`, `mellum`, `nanbeige`, `muse-glimmer`, `maple`, `spark2_5`, `talkie`, `hrm_text`,
`llama4`.

### 1.4 All 62 projector (mmproj) type strings (`master`)

From `PROJECTOR_TYPE_NAMES` (<https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h#L511-L573>).
Vision, audio and TTS projectors share this one namespace.

```
mlp, ldp, ldpv2, resampler, adapter, qwen2vl_merger, qwen2.5vl_merger, qwen3vl_merger, ling3vl, step3vl,
gemma3, gemma3nv, gemma3na, gemma4v, gemma4a, gemma4uv, gemma4ua, phi4, idefics3, pixtral, ultravox,
internvl, llama4, qwen2a, qwen3a, glma, qwen2.5o, voxtral, meralion, musicflamingo, lfm2, kimivl,
paddleocr, lightonocr, cogvlm, janus_pro, dots_ocr, dots3note_v, dots3note_a, deepseekocr, deepseekocr2,
deepseek4v, lfm2a, glm4v, youtuvl, yasa2, kimik25, nemotron_v2_vl, exaone4_5, hunyuanvl, minicpmv4_6,
granite_speech, mimovl, minimax_m3, granite4_vision, mimo_audio, parakeet, qwen3tts_spkenc, qwen3tts_gen,
pockettts_spkenc, pockettts_gen, muse-glimmer
```

**Audio projectors:** `ultravox`, `qwen2a`, `qwen3a`, `glma`, `qwen2.5o`, `voxtral`, `meralion`,
`musicflamingo`, `lfm2a`, `granite_speech`, `mimo_audio`, `parakeet`, `dots3note_a`, and the four TTS ones.
**`b9867` has 50 of these 62** — it lacks `parakeet`, `mimo_audio`, `qwen3tts_spkenc`, `qwen3tts_gen`,
`pockettts_spkenc`, `pockettts_gen`, `minimax_m3`, `muse-glimmer`, `ling3vl`, `dots3note_v`, `dots3note_a`,
`deepseek4v`
(<https://github.com/ggml-org/llama.cpp/blob/b9867/tools/mtmd/clip-impl.h>).

### 1.5 What is NOT in llama.cpp (asked about explicitly)

| Asked about | Status | Evidence |
|---|---|---|
| `whisper` as an LLM arch | **Absent.** Whisper survives only as an *audio encoder* used by other projectors (`clip_graph_whisper_enc::build()` dispatches on `ULTRAVOX`, `QWEN2A`, `VOXTRAL`, `MUSIC_FLAMINGO`, `MERALION`, `GLMA`) | <https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/models/whisper-enc.cpp#L3-L120> |
| `bark`, `musicgen` | **Absent** from both arch tables and from the whole file tree (no `examples/bark`, no `examples/musicgen`) | `src/llama-arch.cpp`; repo tree via GitHub trees API |
| `vocoder` | **Absent** as an arch string | `src/llama-arch.cpp` |
| `stable-diffusion` (image) | **Absent.** `stable-diffusion.cpp` is a **separate project** sharing only ggml | <https://github.com/leejet/stable-diffusion.cpp> |
| `llava`, `qwen2audio`, `clip` as LLM archs | `llava`/`qwen2audio` absent as arch strings; `clip` present only as a dummy for `llama-quantize`; LLaVA is an mmproj (`mlp`) + base-LLM combination | <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp>; <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/llava.md> |
| `nomic-bert`, `jina-bert`, `bge` | `nomic-bert` and `jina-bert-v2`/`jina-bert-v3` are archs. **`bge` is not an arch string**; BGE models convert as `bert` — the repo even ships `models/ggml-vocab-bert-bge.gguf` | `src/llama-arch.cpp`; repo tree (`models/ggml-vocab-bert-bge.gguf`) |

### 1.6 Added after `b9867` (18 archs)

`bailingmoe3`, `dots3note`, `granite_swa`, `graniteswitch`, `hrm_text`, `hy_v3`, `hy_v4`, `kimi-k3`,
`laguna`, `maple`, `minimax-01`, `minimax-m3`, `muse-glimmer`, `nanbeige`, `pockettts`, `qwen3tts`,
`qwen4exp`, `spark2_5`.

Corresponding new model files on `master` include `src/models/{pockettts,qwen3tts,kimi-k3,maple,laguna,
muse-glimmer,minimax-01,minimax-m3,spark2-5,nanbeige,hrm-text,hy-v3,hy-v4,granite-swa,granite-switch,
bailingmoe3,dots3note,qwen4exp}.cpp` and the new mtmd models `{qwen3tts-gen,qwen3tts-spkenc,
pockettts-gen,pockettts-spkenc,pockettts-seanet,mimo-audio,parakeet,ling3vl,deepseek4v,dots3note,
muse-glimmer,minimax-m3}.cpp` (repo tree via GitHub trees API, `master` vs `b9867`).

---

## 2. The non-LLM side

### 2.1 ASR / speech-to-text

- **llama.cpp has no Whisper inference.** There is no `whisper` arch, no `examples/whisper`, and no
  whisper model file. Speech-to-text in llama.cpp today means **running an audio-LLM with an audio
  projector**, not running Whisper.
- **What the loader needs:** an audio-capable **mmproj**. `docs/multimodal.md` states multimodal input is
  enabled with `-m model.gguf --mmproj file.gguf`, or automatically by `-hf` (with `--no-mmproj` to opt
  out) (<https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md#L10-L17>). The server README
  confirms `-hf` "mmproj is also downloaded automatically if available"
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L102>).
- **CLI/API surface:**
  - `--image` / `--audio` / `--video FILE` — one flag with three aliases, for `llama-mtmd-cli` and `llama-cli`
    only (not server) (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L2625-L2633>).
  - **`POST /v1/audio/transcriptions` and `POST /audio/transcriptions`** exist on `llama-server`
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L266-L267>). They are
    OpenAI-shaped and are **converted into a chat completion** internally
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp#L5018-L5028>).
  - Gate: the handler refuses unless `meta->has_mtmd && chat_params.allow_audio`, returning
    `"The current model does not support audio input."` with `ERROR_TYPE_NOT_SUPPORTED`
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp#L5009-L5015>).
  - Chat-completions audio input uses an `input_audio` content part; `data` or `url`, formats mp3/wav/flac
    via `miniaudio` (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L1332-L1339>).
- **Models that actually ship audio GGUF** (from `docs/multimodal.md#L105-L141`):
  `ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF`, `ggml-org/ultravox-v0_5-llama-3_1-8b-GGUF`,
  `ggml-org/Voxtral-Mini-3B-2507-GGUF`, `ggml-org/Qwen3-ASR-0.6B-GGUF`,
  `ggml-org/Qwen3-ASR-1.7B-GGUF`, `ggml-org/Qwen2.5-Omni-3B-GGUF`, `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`.
- **`whisper.cpp` is a separate project** with its own `whisper-server` exposing a non-OpenAI `/inference`
  endpoint (multipart WAV in), configurable via `--inference-path`
  (<https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md#L3-L51>).

### 2.2 TTS / voice cloning

**Supported by llama.cpp (`master`), and only these:**

| Model | How | Evidence |
|---|---|---|
| **Qwen3-TTS** | `llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "…" --output out.wav`; `--tts-lang` (zh/en/de/it/pt/es/ja/ko/fr/ru), `--tts-speaker-file` for reference audio (voice cloning) | <https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md#L20-L34> |
| **Pocket TTS** (Kyutai) | `llama-tts -m pocket-tts.gguf -mm mmproj-pocket-tts.gguf …`; `--tts-speaker-file` is **required**; convert one `languages/<name>` dir, not the root, with `--mmproj` for the projector | <https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md#L36-L59> |
| **OuteTTS** | The `tools/tts` tool *was* the OuteTTS demo and was "converted to a more model-agnostic tool"; `b9867` still ships `tools/tts/tts-outetts.py` and an OuteTTS-0.2-500M conversion recipe | <https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md#L5>; <https://github.com/ggml-org/llama.cpp/blob/b9867/tools/tts/README.md> |

TTS flags `--tts-lang` and `--tts-speaker-file` are registered with `set_examples({LLAMA_EXAMPLE_TTS})`
only (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L4422-L4436>).

**Critical for Rigma: there is no OpenAI-compatible TTS endpoint in llama-server.** The full route table
contains no `/v1/audio/speech` or any speech-synthesis route
(<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L244-L373>). TTS is a
one-shot CLI binary that writes a WAV file. A chat-shaped UI cannot serve it without a wrapper.

**NOT supported by llama.cpp** — each of these is a different engine:

| System | Runs on | Evidence |
|---|---|---|
| **Kokoro** | PyTorch inference library (`pip install kokoro`); served via Kokoro-FastAPI or Speaches | <https://github.com/hexgrad/kokoro>; <https://github.com/remsky/Kokoro-FastAPI#L18> |
| **Piper** | ONNX/VITS C++ — repo moved to `OHF-Voice/piper1-gpl`; served via openedai-speech, Speaches, LocalAI | <https://github.com/OHF-Voice/piper1-gpl>; <https://github.com/matatonic/openedai-speech#L28-L30> |
| **Coqui XTTS v2** | PyTorch (`coqui-tts`); voice cloning, ~4 GB VRAM | <https://github.com/idiap/coqui-ai-TTS>; <https://github.com/matatonic/openedai-speech#L30-L33> |
| **F5-TTS** | PyTorch | <https://github.com/SWivid/F5-TTS> |
| **Orpheus TTS** | **vLLM** (`pip install orpheus-speech`, "uses vllm under the hood") | <https://github.com/canopyai/Orpheus-TTS#L56-L58> |
| **Chatterbox** | PyTorch (`pip install chatterbox-tts`) | <https://github.com/resemble-ai/chatterbox#L52> |
| **Sesame CSM** | Python reference impl (`python run_csm.py`) | <https://github.com/SesameAILabs/csm> |
| **Dia** | Python / HF Transformers | <https://github.com/nari-labs/dia> |
| **Zonos** | PyTorch | <https://github.com/Zyphra/Zonos> |
| **Bark / MusicGen** | Not in llama.cpp; no engine found in this pass | — |

LocalAI's README additionally documents a family of dedicated C++/GGML TTS ports that are **not** part of
llama.cpp: `moss-tts.cpp`, `magpie-tts.cpp`, `voxtral-tts.c`, `vibevoice.cpp`, plus ASR engines
`parakeet.cpp` and `moss-transcribe.cpp`
(<https://github.com/mudler/LocalAI#L232-L239>). Note `parakeet` and `mimo_audio` *are* also projector types
inside llama.cpp's mtmd on `master` — see §1.4.

### 2.3 Vision / multimodal

- **How mmproj works.** The projector is a **separate GGUF artifact** attached with `-mm/--mmproj`
  (`--mmproj-url` for a URL, `--mmproj-auto`/`--no-mmproj`, `--mmproj-offload`, `--mmproj-device`)
  (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L2574-L2624>). It is downloaded
  automatically only in `-hf` mode. In a local directory layout the file name **must start with `mmproj`**
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L1680-L1692>).
- **Which architectures support it.** All of them, in the sense that any LLM arch can in principle be paired
  with a projector; what matters is that the *pair* is a supported combination. `docs/multimodal.md`
  enumerates the working pairs, and its list is dominated by `llama`, `mistral`, `phi2`, `qwen2vl`,
  `qwen3vl`, `gemma3`, `gemma4`, `llama4`, `internlm2` base archs
  (<https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md#L52-L141>).
- **What breaks on a mismatch.** llama.cpp does **not** publish a clean "wrong projector" error for a
  merely-incompatible pair. The checks it does perform are:
  - a hard failure on an **unknown projector type** — `clip.cpp` resolves the type string and returns
    `PROJECTOR_TYPE_UNKNOWN`, aborting with an unsupported-projector message
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h#L575-L582>);
  - a hard failure when the model cannot accept the modality at all — for audio,
    `"The current model does not support audio input."`
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp#L5012-L5014>);
  - clients are told to **ask first**: "A client *must not* specify this field unless the server has the
    multimodal capability. Clients should check `/models` or `/v1/models` for the `multimodal` capability
    before a multimodal request"
    (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L509>).
  A **shape-mismatched but known** projector (right type string, wrong hidden size) is **[inference]** a
  tensor-shape assertion or garbage output rather than a friendly error; I did not find a dedicated
  compatibility check for it and did not run an engine to observe one.
- **Practical rule for Rigma:** the only reliable pre-flight signal is *which projector types exist* plus
  the projector's own declared dimensions. Treat mmproj as a **first-class file with its own type**, exactly
  as `gguf_meta.inspect_gguf` already does by flagging `arch == "clip"` or any `clip.*` key
  (<https://github.com/ComfyUI/RD/rigma-review/blob/main/src/rigma/gguf_meta.py#L253-L254>).

### 2.4 Embeddings / rerankers

- **Archs:** `bert`, `modern-bert`, `nomic-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`,
  `jina-bert-v3`, `eurobert`, `t5encoder` (and full `t5`), `gemma-embedding`, `pangu-embedded`,
  `llama-embed`; plus `granite`, `qwen3` and `gemma` used as embedding bases
  (<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp>).
- **Pooling types** are `{none, mean, cls, last, rank}` via `--pooling`, defaulting to the model's own
  choice (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L2310>;
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L175>). The enum in
  `gguf-py` is `NONE=0, MEAN=1, CLS=2, LAST=3, RANK=4`
  (<https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py#L5681-L5686>).
- **Flags:**
  - `--embedding` / `--embeddings` — "restrict to only support embedding use case; use only with dedicated
    embedding models" (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L3472-L3478>).
  - `--rerank` / `--reranking` — sets **both** `embedding = true` and
    `pooling_type = LLAMA_POOLING_TYPE_RANK`
    (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L3479-L3486>).
- **Endpoints** (all POST):
  `/embedding` (legacy), `/embeddings`, **`/v1/embeddings`** (OpenAI-compatible);
  `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking`
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L270-L276>).
  `/v1/embeddings` is documented as the OAI-compatible one; `/embedding` is explicitly "**not**
  OAI-compatible" (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L744>).
- **Reranking requires a reranker model plus `--embedding --pooling rank`**, e.g. `bge-reranker-v2-m3`
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L763-L766>).
- With `--pooling none` the endpoint returns unnormalized per-token embeddings; all other pooling types
  return pooled, Euclidean-normalized embeddings
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L936>).
- Reranking also accepts multimodal prompts (`/reranking` "also supports multimodal embeddings")
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L748>).
- **Sparse/lexical, ColBERT multi-vector, Matryoshka truncation:** no support found in the arch table or
  the server docs — **[inference]** from absence.

### 2.5 Diffusion / image generation

- **No image generation in llama.cpp.** The word "diffusion" in this repo means *diffusion language
  models*: `dream`, `llada`, `llada-moe`, `rnd1`, exposed through `examples/diffusion`
  (<https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp> — `llm_arch_is_diffusion`;
  <https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion>).
- **stable-diffusion.cpp is a separate project** — "Plain C/C++ implementation based on ggml, working in the
  same way as llama.cpp" (<https://github.com/leejet/stable-diffusion.cpp#L37>). It is not linked into
  llama.cpp and shares no arch table.

---

## 3. Family → requirements

`Rigma chat UI?` = does a chat-shaped UI make sense for this family as-is.

| Family | Arch strings (llama.cpp) | Extra files | Required flags | API endpoint | Rigma chat UI? |
|---|---|---|---|---|---|
| Dense text LLM | `llama`, `qwen2`, `qwen3`, `gemma*`, `phi*`, `mistral*`, `glm4`, `granite`, `exaone*`, … | none | `-m`, optional `--chat-template`, `-ngl`, `-c` | `/v1/chat/completions`, `/v1/completions` | **Yes** |
| MoE text LLM | `*moe*`, `gpt-oss`, `deepseek2/32/4`, `llama4`, `dbrx`, `minimax-m*`, … | none | as above + MoE offload flags | same | **Yes** |
| Hybrid / SSM / recurrent | `mamba`, `mamba2`, `jamba`, `falcon-h1`, `granitehybrid`, `lfm2*`, `nemotron_h*`, `qwen3next`, `qwen35*`, `kimi-linear`, `kimi-k3`, `plamo2` | none | as above; recurrent state instead of KV | same | **Yes** |
| Diffusion LM | `dream`, `llada`, `llada-moe`, `rnd1` | none | diffusion decode loop (`examples/diffusion`), not standard sampling | none in server — **[inference]** | **No** |
| Draft / speculative head | `eagle3`, `dflash`, `gemma4-assistant` | pairs with a base model | speculative flags | n/a (not standalone) | **No** |
| Vision VLM | base `llama`/`mistral`/`phi2`/`qwen2vl`/`qwen3vl`/`qwen3vlmoe`/`gemma3`/`gemma4`/`llama4`/`internlm2`/`minicpm*`/`hunyuan_vl`/`cogvlm`/`paddleocr`/`deepseek2-ocr` | **`mmproj` GGUF** (vision projector: `mlp`, `pixtral`, `idefics3`, `qwen2vl_merger`, `gemma3`, …) | `-m` + `-mm`, optional `--no-mmproj-offload`, `--image-min/max-tokens` | `/v1/chat/completions` with image parts; check `/v1/models` `multimodal` capability | **Yes** (needs attachment UI) |
| Audio-in / ASR | base `llama` (Ultravox), `mistral3`/Voxtral, `qwen3` (Qwen3-ASR), `qwen2`/`qwen3`-Omni, `gemma4` | **audio `mmproj`** (`ultravox`, `qwen2a`, `qwen3a`, `voxtral`, `glma`, `qwen2.5o`, `granite_speech`, `lfm2a`, `meralion`, …) | `-m` + `-mm`; CLI `--audio`; server needs `allow_audio` | **`/v1/audio/transcriptions`**, `/audio/transcriptions`; chat with `input_audio` | **Partly** — transcription is not chat |
| TTS | `qwen3tts`, `pockettts` (**`master` only**) | **two** projectors: `*_gen` + `*_spkenc` (Pocket TTS: one `mmproj-*.gguf` via `-mm`) | `llama-tts` binary, `--tts-lang`, `--tts-speaker-file`, `--output` | **none** — no `/v1/audio/speech` in llama-server | **No** |
| TTS (legacy) | `wavtokenizer-dec` (decoder component); OuteTTS via `tools/tts` on `b9867` | OuteTTS LLM + audio decoder GGUF | `llama-tts`, `--tts-oute-default` | none | **No** |
| Embedding | `bert`, `modern-bert`, `nomic-bert*`, `neo-bert`, `jina-bert-v2/v3`, `eurobert`, `t5encoder`, `gemma-embedding`, `pangu-embedded`, `llama-embed`, `granite`, `qwen3` | none | **`--embedding`**, optional `--pooling {none,mean,cls,last,rank}`, batch flags | `/v1/embeddings` (OAI), `/embedding` (legacy) | **No** |
| Reranker | cross-encoder `bert`-family (e.g. `bge-reranker-v2-m3`) | none | **`--embedding --pooling rank`** (or `--reranking`, which sets both) | `/v1/rerank`, `/rerank`, `/reranking`, `/v1/reranking` | **No** |
| Encoder-decoder | `t5` | none | standard | server text endpoints | **Partly** — seq2seq, not chat |
| Projector-only file | `general.architecture = clip` or any `clip.*` key | is itself the extra file | never launched directly | n/a | **No** |
| **Not in llama.cpp at all** | Kokoro, Piper, XTTS, F5-TTS, Orpheus, Chatterbox, CSM, Dia, Zonos, Bark, MusicGen, stable-diffusion (image) | — | — | — | separate engine required |

Notes that apply across rows:

- `--no-conversation` / `-no-cnv` is a **CLI-only** flag ("default: auto enabled if chat template is
  available") registered for `LLAMA_EXAMPLE_COMPLETION`, i.e. `llama-cli`, **not** the server
  (<https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L1898-L1908>). It does not exist as a
  server option; the server decides from the model's own template.
- A chat template is **optional at the file level** but required for conversation mode. Rigma already
  tracks this as `has_template` because a quantiser can strip `tokenizer.chat_template`
  (<https://github.com/ComfyUI/RD/rigma-review/blob/main/src/rigma/gguf_meta.py#L418-L450>).
- `-hf` auto-downloads mmproj when available
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L102>); for local paths the
  file name must start with `mmproj`
  (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L1680-L1692>).

---

## 4. Engine version requirements

### 4.1 Type table — the actual version boundary

| | `b9867` (2026-07-03) | `master` | PrismML `prism` |
|---|---|---|---|
| Highest type id | `GGML_TYPE_Q1_0 = 41` | `GGML_TYPE_Q2_0 = 42` | `GGML_TYPE_PTQ1_0 = 143` |
| `GGML_TYPE_COUNT` | **42** | **43** | **144** |
| Loader check | `gguf.cpp`: `if (info.t.type < 0 \|\| info.t.type >= GGML_TYPE_COUNT)` → `"tensor '%s' has invalid ggml type %d. should be in [0, %d)"` | same, bound 43 | n/a (fork) |

Sources: <https://github.com/ggml-org/llama.cpp/blob/b9867/ggml/include/ggml.h#L431-L432>,
<https://github.com/ggml-org/llama.cpp/blob/b9867/ggml/src/gguf.cpp#L701-L703>,
<https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h#L430-L431>,
<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp#L722-L724>,
<https://github.com/PrismML-Eng/llama.cpp/blob/prism/ggml/include/ggml.h#L435-L437>.

**Consequences for Rigma:**

- `GGML_TYPE_Q2_0 = 42` is **newer than the pinned build**. A Q2_0 file passes Rigma's current check
  (`KNOWN_TYPES[42]`, `MAINLINE_MAX_TYPE = 42`) and fails on `b9867`. This is the concrete false negative.
- Type `142` is **not** a "newer engine" case at all: no mainline build, however new, accepts it. Rigma's
  existing wording is correct; only the *bound* is version-skewed.
- Correct model: `(engine_build) -> (max_type_id, type_count, name_table)`, and the verdict becomes
  "needs engine ≥ build N" only when the type is mainline-but-newer; "fork-private" when it exceeds the
  newest known mainline id.

### 4.2 Architectures that need newer than `b9867`

The 18 archs listed in §1.6 — all require `master`. Practically relevant ones:

- **`qwen3tts`, `pockettts`** — TTS does not exist on `b9867` at all (neither arch, nor `src/models/*.cpp`,
  nor the four TTS projectors). On `b9867`, `tools/tts` is the OuteTTS demo only
  (<https://github.com/ggml-org/llama.cpp/blob/b9867/tools/tts/README.md#L1-L10>).
- **`kimi-k3`, `qwen4exp`, `minimax-01`, `minimax-m3`, `granite_swa`, `graniteswitch`, `bailingmoe3`,
  `hy_v3`, `hy_v4`, `maple`, `laguna`, `muse-glimmer`, `nanbeige`, `spark2_5`, `dots3note`, `hrm_text`** —
  newer text LLMs; `b9867` fails on them at the arch lookup (`LLM_ARCH_UNKNOWN`).
- **`parakeet` and `mimo_audio` projectors** are also post-`b9867`.

### 4.3 What needs vLLM instead of llama.cpp

- **Orpheus TTS** — vLLM-backed (<https://github.com/canopyai/Orpheus-TTS#L56-L58>).
- **Whisper as a dedicated ASR model** — `WhisperForConditionalGeneration` is listed in vLLM's
  Transcription table (<https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md>).
  llama.cpp has no Whisper arch; it only has Whisper-*style encoders inside projectors.
- **Broader audio-language coverage.** vLLM's audio-language table includes `Qwen2AudioForConditionalGeneration`,
  `UltravoxModel`, `GraniteSpeechForConditionalGeneration`, `KimiAudioForConditionalGeneration`,
  `MiDashengLMModel`, `MossAudioModel`, `AudioFlamingo3ForConditionalGeneration`,
  `Step1ForCausalLM` (Step-Audio), `Qwen2_5OmniThinkerForConditionalGeneration`,
  `Qwen3OmniMoeThinkerForConditionalGeneration`, `Qwen3ASRForConditionalGeneration`
  (<https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md>). Several of these
  (Kimi-Audio, MiDashengLM, MOSS-Audio) are **vLLM-only** — no llama.cpp arch or projector exists for them.
- **Realtime/streaming transcription** — `VoxtralRealtimeGeneration`, `Qwen3ASRRealtimeGeneration`
  (<https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md>).

### 4.4 What needs neither

Kokoro, Piper, XTTS, F5-TTS, Chatterbox, CSM, Dia, Zonos and image generation do not use llama.cpp or vLLM
as their engine; they are PyTorch/ONNX projects served by their own servers (see §5).

---

## 5. Voice-model deployment reality

Working stacks, engine + model + serving:

| Want | Stack | Served how | OpenAI-compatible? |
|---|---|---|---|
| **ASR (fastest path)** | `faster-whisper` (CTranslate2) + `speaches` | Docker, `/v1/audio/transcriptions`, streaming SSE | **Yes** (<https://github.com/speaches-ai/speaches#L3>, <https://github.com/SYSTRAN/faster-whisper#L240>) |
| **ASR (ggml/C++)** | `whisper.cpp` + `whisper-server` | `whisper-server --host 0.0.0.0 -m ggml-base.bin`; endpoint `/inference` (multipart WAV), path configurable via `--inference-path` | **No** — not the OpenAI shape (<https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md#L45-L98>) |
| **ASR (in llama.cpp)** | audio-LLM + audio mmproj, e.g. `ggml-org/Qwen3-ASR-0.6B-GGUF` | `llama-server -hf …`; `/v1/audio/transcriptions` | **Yes** (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L266-L267>) |
| **TTS (llama.cpp)** | `qwen3tts` / `pockettts` + `llama-tts` | CLI only, writes `out.wav` | **No endpoint** |
| **TTS (OpenAI-compatible, Piper)** | Piper + `openedai-speech` | `POST /v1/audio/speech`, `tts-1` = Piper, `tts-1-hd` = XTTS v2 cloning | **Yes** (<https://github.com/matatonic/openedai-speech#L16-L31>) |
| **TTS (OpenAI-compatible, Kokoro)** | Kokoro-82M + `Kokoro-FastAPI` | `POST /v1/audio/speech`, `GET /v1/audio/voices`; `/v1/*` declared the stable API | **Yes** (<https://github.com/remsky/Kokoro-FastAPI#L18>, #L796) |
| **ASR + TTS in one server** | `speaches` — faster-whisper + Piper/Kokoro | "aims to be Ollama, but for TTS/STT models"; OpenAI API compatible; dynamic model load/offload; Realtime API | **Yes** (<https://github.com/speaches-ai/speaches#L3-L21>) |
| **Everything (multi-backend)** | LocalAI — wraps llama.cpp, vLLM, whisper.cpp, stable-diffusion, kokoro, parakeet.cpp, plus GGML TTS ports | OpenAI-compatible incl. Realtime API | **Yes** (<https://github.com/mudler/LocalAI#L204-L239>) |
| **Orpheus TTS** | vLLM + `orpheus-speech` | vLLM's OpenAI-compatible server | **Yes**, via vLLM (<https://github.com/canopyai/Orpheus-TTS#L56-L58>) |
| **Chatterbox / F5-TTS / XTTS / CSM / Dia / Zonos** | PyTorch libraries; Gradio or bespoke servers | third-party OpenAI-compatible wrappers exist; not from upstream | **[inference]** — upstream READMEs show no OpenAI server |

**Answer to "can one client talk to both text and voice?"** Yes, but not through llama.cpp alone.
The two coherent answers are:

1. **`speaches`** (or LocalAI) as a sidecar: `llama-server` for text, `speaches` for
   `/v1/audio/transcriptions` + `/v1/audio/speech`. One OpenAI client, two base URLs.
2. **llama.cpp alone** covers text **and** ASR (`/v1/audio/transcriptions` on `master`), but **not TTS** —
   there is no speech-synthesis route. TTS from llama.cpp is a CLI invocation.

---

## 6. What this means for Rigma

**In scope, worth handling explicitly (a chat UI can front all of these):**
text chat (dense/MoE/hybrid), vision (needs `mmproj` attachment + capability advertisement), audio-in/ASR
(needs audio `mmproj`; `/v1/audio/transcriptions` is a genuinely different request shape from chat).

**In scope but not chat-shaped — Rigma should model them as *roles*, not launch them in the chat pane:**
embeddings (`--embedding`), rerankers (`--embedding --pooling rank`), TTS (`llama-tts`, CLI, no endpoint),
diffusion LMs (special decode loop).

**Out of scope, and Rigma should say so rather than plan a launch:**
Kokoro, Piper, XTTS, F5-TTS, Orpheus, Chatterbox, CSM, Dia, Zonos, Bark, MusicGen, image generation, and
`whisper` as a standalone model. Each needs a different engine; LocalAI/speaches/whisper.cpp are the
practical bridges.

**Two concrete defects to file from this research:**

1. **Version-skewed type bound (false negative).** `engine_compat.MAINLINE_MAX_TYPE = 42` describes
   `master`, not the pinned `b9867` (`COUNT = 42`, max 41). A Q2_0 model is currently accepted by the
   checker and rejected by the engine. Make the bound a function of the pinned build.
2. **No role vocabulary.** `ModelSpec` has `family`, `kind: dense|moe`, `capabilities: tools|vision|thinking`
   and `mmproj`
   (<https://github.com/ComfyUI/RD/rigma-review/blob/main/src/rigma/models.py#L247-L262>). There is no
   `embedding` / `reranker` / `asr` / `tts` / `diffusion` role, and `gguf_meta` detects mmproj but not
   *which modality* the projector is. The projector type string (62 values, §1.4) is the missing key: it
   distinguishes a vision projector from an audio one from a TTS one before launch.

---

## Could NOT verify

- **Web search was unavailable.** `web_search` returned
  `Error: DeepSeek search has no API key for "DEEPSEEK_API_KEY"`, and the documented
  `https://html.duckduckgo.com/html/?q=…` endpoint now serves the anti-bot challenge itself (HTTP 202,
  `anomaly.js`, `cc=botnet`, "Select all squares containing a duck"). All external claims here therefore
  rest on **direct source/README fetches** (raw.githubusercontent.com), which is the stronger source anyway,
  but there was **no search-based discovery**: a project that exists but whose exact repo path I did not
  guess is invisible to this report.
- **`Ternary-Bonsai-2-27B` itself** — I could not confirm the model exists on Hugging Face, nor inspect its
  GGUF metadata. The fork, its type ids and its file-type ids **are** verified from source. The claim that
  this specific model uses type 142 comes from Rigma's own docstring
  (`src/rigma/engine_compat.py`), not from the file.
- **No engine was built or run.** Every behavioural claim is source-reading. I did not observe the
  `invalid ggml type` error, the `"does not support audio input"` error, or any projector mismatch at
  runtime.
- **Mismatched-projector behaviour.** I found no dedicated "wrong projector for this base model" check with
  a friendly error, but proving a negative from source is weak; the runtime symptom for a
  same-type/wrong-shape projector is unverified.
- **Hugging Face repo contents** for Whisper GGUF (e.g. `ggerganov/whisper.cpp`), mmproj naming patterns
  beyond `docs/multimodal.md`, and GGUF availability for Kokoro/Piper/XTTS were not fetched. Repo names in
  §2.1/§2.3 are cited only where `docs/multimodal.md` or a README states them.
- **TTS-only repo existence:** `OuteAI/OuteTTS`, `senstella/csm.cpp` and
  `R-D-BioTech-Alaska/Orpheus-TTS.cpp` all failed to fetch (404/private/renamed) at the paths I tried, so
  "a llama.cpp port of CSM/Orpheus exists" is **unconfirmed**. `senstella/csm.cpp` and `OuteTTS` should be
  re-checked by search once search works.
- **vLLM TTS.** vLLM's `supported_models.md` has no `/v1/audio/speech` mention and no TTS section I could
  find; that is **absence of evidence**, not verified absence of a TTS endpoint.
- **MoE membership** for archs whose model class is not named `*moe*` (see §1.3) is inferred from arch
  names and is not sourced.
- **`docs/multimodal/llava.md` and the other per-model multimodal docs** were listed but not read
  individually; their arch mappings are reported from `docs/multimodal.md` and the file listing only.
- **Projector↔base-model pairing** is enumerated from `docs/multimodal.md`'s pre-quantized list, which is a
  curated subset; the projector table has 62 types and I did not map every one to its base LLM.
