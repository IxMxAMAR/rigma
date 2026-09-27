# 06-r3 fit / resolve / probe / downloads — run 3 adversarial re-audit

fork: `r3/fit` from `5b5790b` (= `review/deep-audit-2026-09-22` HEAD, all run-2 fixes present)
scope: resolve.py, models.py, quant_quality.py, probe.py, gguf_meta.py, registry.py,
hf_browse.py, hangar.py, bench.py, kvcache.py, prefixcache.py, server_ops.py, engine_log.py

Prior work read first, so nothing below re-reports a closed finding:
`docs/review/findings/06-resolve-fit-probe.md` (06-1..06-8),
`docs/review/findings/07-hangar-downloads.md` (07-1..07-7), and the
`### 06`/`### 07` sections of `docs/review/REPORT.md`.

## Summary

10 findings: 1 HIGH, 5 MEDIUM, 4 LOW. Five fixed on `r3/fit` (one finding, one
commit, test-first, ruff clean):

| id | severity | fix commit | one line |
|----|----------|-----------|----------|
| 06R3-1 | HIGH | `c9d11c2` | `_measured_placement` types the placement it returns |
| 06R3-2 | MEDIUM | `7413183` | `_install_download` refuses a body nothing can verify |
| 06R3-3 | MEDIUM | `0924b29` | `KV_CACHE_TYPES` derived from `CACHE_BYTES`, not restated |
| 06R3-4 | MEDIUM | `7ff3785` | a negative `Content-Length` reads as "not declared" |
| 06R3-5 | MEDIUM | `032db43` | flat registry zip raises a named error, not `StopIteration` |

Not fixed, with reasons in "Could not execute / deliberately not fixed":
06R3-6 (calibration key is not GPU-specific), 06R3-7 (`list_models` swallows
every fit exception), 06R3-8 (GGUF version floor-only), 06R3-9 (unbounded
`params` from tensor dims), 06R3-10 (log line attribution).

---

### 06R3-1 [HIGH] FIXED `c9d11c2` — `server_ops._measured_placement` merges raw `calibration.json` values into `ComboFlags` with `model_copy(update=...)`, the exact bypass 06-4 fixed in the sibling path
- **Where:** `src/rigma/server_ops.py:223-224` (read), applied unvalidated at `:506` and `:525`
- **Trigger:** `~/.rigma/calibration.json` contains a row keyed `<slug>:<quant>:<backend>` whose `flags` object carries a non-integer for a placement field, e.g.
  `{"ctx": 32768, "engine": "b9867", "flags": {"ngl": "not-a-number", "n_cpu_moe": "eight"}}`.
  Reached on the ordinary switch path: `perform_switch` → `_measured_placement(rp, want)` → `rp.flags.model_copy(update=update)`.
- **Consequence:** the two fields the fit math and the launch argv both depend on become strings. `RunPlan.server_args` dies with a `TypeError` before the engine is ever spawned:
  ```
  _measured_placement -> {'ngl': 'not-a-number', 'n_cpu_moe': 'eight'}
  merged.flags.ngl = 'not-a-number'  n_cpu_moe = 'eight'
  server_args RAISED TypeError: '>' not supported between instances of 'str' and 'int'
    (models.py:406, `if self.flags.n_cpu_moe > 0:`)
  ```
  `server_ops.perform_switch` wraps nothing here, so the Models page gets a 500 and `rigma up` a traceback; on the way there `_spilled`/`quant_verdicts` do arithmetic on the same value. A *numeric* but wrong value is worse: `{"ngl": 9999}` is passed literally as `-ngl 9999`.
- **Cause:** 06-4 fixed `resolve._apply_calibration` by routing the merge through `ComboFlags.model_validate` (`resolve.py:113-125`, all-or-nothing with an explain note). `_measured_placement` performs the same read-merge-into-ComboFlags operation but returns the raw dict and lets the caller `model_copy` it, which by design does not validate. `server_ops.py:225-226` guards only against a *corrupt file* (the `except Exception: return {}`), not against a well-formed file with a bad value — the module's own comment says "a missing or corrupt calibration is not an error".
- **Fix:** validate the two fields in `_measured_placement` before returning them (**implemented, `c9d11c2`**). Only `ngl` and `n_cpu_moe` are read out, so only they need checking; a whole-number float is coerced (JSON has no integer type), and anything else drops the entry rather than half-applying it.
- **Verified:** `python .scratch/00-recon/r3d-cal.py` → the transcript above (real `TypeError` at `models.py:406`). Regression tests in `tests/test_server_ops.py`: `test_a_calibration_row_with_a_non_integer_placement_is_ignored`, `test_a_numeric_calibration_placement_still_wins`, `test_a_float_placement_is_coerced_to_an_integer`, `test_a_fractional_placement_is_not_replayed`, `test_a_negative_placement_is_not_replayed` — the first four fail before the fix, all 22 pass after.

### 06R3-2 [MEDIUM] FIXED `7413183` — `_install_download` installs whatever bytes arrived when neither the server nor the registry declares a length
- **Where:** `src/rigma/hangar.py:1256-1293` (`_install_download`), reached from `:1397` and `:1367`
- **Trigger:** a response with no `content-length` (chunked transfer) on a file the registry also prices at `bytes: 0` — or any caller passing `size=0`. `_object_size` returns 0 for `(200, {})`, and `_download_file:1359` computes `size = _object_size(...) or expect_bytes`, which is 0 when both are unknown.
- **Consequence:** the size gate is skipped (`if size and have != size`), and with no `sha256` in the registry the identity gate is skipped too. A 9-byte body is renamed in as a complete multi-GB quant:
  ```
  === A. _install_download with size=0 (server declared no length) ===
    INSTALLED a 9-byte file as complete: True 9
  ```
  `list_models` then marks the row `on_disk`, `_calculate` plans against `gguf.bytes == 0`, and the engine fails to load a file that the UI says is downloaded. `GgufFile.bytes` can legitimately be 0 — `hangar.merge_repo_files:650-651` builds a new entry as `sizes.get(fname, 0)`.
- **Cause:** the size/hash gates were added by 07-3 in front of the `dest.exists()` short-circuit, but both are conditional on a *declared* size and a *stored* hash. With neither, the function's only remaining evidence is "the transport ended cleanly", which is exactly what `_ShortBody` exists to disbelieve.
- **Fix:** refuse to install when there is nothing to verify against — `if not size and not sha256: raise HangarError(...)` before the `os.replace` (**implemented, `7413183`**). The bytes stay in the `.part`, so the message can tell the user to retry rather than discard their progress.
- **Verified:** `python .scratch/00-recon/r3d-hangar.py` section A (transcript above). Regression tests `tests/test_audit_hangar.py::test_a_body_with_no_declared_length_is_not_installed_unverified` (fails before, passes after) plus two controls proving a chunked origin still installs when the registry knows the size, or when a sha256 is known. `test_hangar.py::test_download_file_streams_with_progress_and_resume` was relying on the old silence — its fake origin declared no length — and now declares one, as a real origin does.

### 06R3-3 [MEDIUM] FIXED `0924b29` — `server_ops.KV_CACHE_TYPES` is still a third, narrower vocabulary, so the three types 06-3 added to the fit math cannot be selected
- **Where:** `src/rigma/server_ops.py:307`, consumed by `serve.py:3492`, `:3688`, `:3835` and `server_ops.py:408`
- **Trigger:** ask the Models-page explorer or `perform_switch` for any of the cache types `CACHE_BYTES` now prices.
- **Consequence:**
  ```
  CACHE_BYTES       : ['bf16', 'f16', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0']
  KV_CACHE_TYPES    : ['f16', 'q4_0', 'q5_1', 'q8_0']
  in CACHE_BYTES not in KV_CACHE_TYPES: ['bf16', 'q4_1', 'q5_0']
  ```
  06-3's fix made `quant_verdicts(spec, prof, kv="q4_1")` work (verified: `kv='q4_1': ok, kv reported = q4_1`), but the HTTP boundary still answers `400 kv must be one of f16, q8_0, q5_1, q4_0` for `q4_1`, `q5_0` and `bf16`. The `_KV_PPL` table publishes a reference loss figure for all three, so the UI offers a quality comparison it refuses to compute. `test_phase0_contracts.py::test_cache_bytes_covers_all_offered_kv_types` only asserts the subset direction (`KV_CACHE_TYPES ⊆ CACHE_BYTES`), so it passes while the gap is real.
- **Cause:** the run-2 note in `PROGRESS.md` recorded this as an observation ("server_ops.KV_CACHE_TYPES is a third, narrower vocabulary") but no fix was assigned. The test that exists checks the direction that cannot fail.
- **Fix:** derive the tuple from the one table — `KV_CACHE_TYPES = tuple(CACHE_BYTES)` (**implemented, `0924b29`**) — and tighten the contract test to equality, so the third vocabulary cannot reappear.
- **Verified:** `python .scratch/00-recon/r3d-fit.py` section 1 and `r3d-r4.py` section B (transcripts above). Regression tests `tests/test_phase0_contracts.py::test_cache_bytes_and_offered_kv_types_are_one_table` (fails before with `priced-but-unofferable: ['bf16', 'q4_1', 'q5_0']`) and `::test_every_offered_kv_type_gets_a_fit_verdict`. No frontend change: the new types are additive to the whitelist — nothing previously accepted is now refused — so the committed bundle keeps offering the original four and needs no rebuild.

### 06R3-4 [MEDIUM] FIXED `7ff3785` — `_object_size` returns a negative total from a lying `Content-Length`, and every resume then discards a good `.part`
- **Where:** `src/rigma/hangar.py:1236-1239`
- **Trigger:** a server or proxy that answers `200` with `Content-Length: -5` (the hypothesis in the brief: "a server that lies about Content-Length").
- **Consequence:** `_object_size(200, {"content-length": "-5"}) == -5`. `-5` is truthy, so the size gate fires on every attempt and the message is nonsense:
  ```
  _object_size(200, {'content-length': '-5'}) = -5
  RAISED HangarError neg.gguf came back 100 bytes where -5 were expected —
         those are not this file's bytes, so they were discarded rather than installed.
  ```
  `have > size` (100 > -5) takes the "these are not this file's bytes" branch, which calls `_discard_partial` — so each retry throws away the accumulated multi-GB `.part` and the pull can never converge. The terminal error blames the connection ("download kept dropping"), not the header that caused it.
- **Cause:** `int(headers.get("content-length") or 0)` accepts any integer, including a negative one. `content-length` is an RFC 7230 `1*DIGIT` field; a negative value is a malformed header, and the honest reading of "malformed" is "not declared" (0), which is how every other unparseable case is already handled two lines up.
- **Fix:** clamp to non-negative — `return max(0, int(...))` (**implemented, `7ff3785`**). 0 then means "not declared" consistently, which is what `_download_file`'s `or expect_bytes` fallback already assumes.
- **Verified:** `python .scratch/00-recon/r3d-r5.py` section E (transcript above). Regression test `tests/test_audit_hangar.py::test_a_negative_content_length_is_read_as_undeclared` (fails before with `assert -5 == 0`).

### 06R3-5 [MEDIUM] FIXED `032db43` — `registry.update_registry` raises a bare `StopIteration` out of `rigma update`
- **Where:** `src/rigma/registry.py:57` — `inner = next(p for p in tmp.iterdir() if p.is_dir())`
- **Trigger:** a registry zip whose entries are all at the top level (no `rigma-registry-master/` wrapper), i.e. any re-packaging or a GitHub archive of a repo whose default branch layout changed.
- **Consequence:**
  ```
  RAISED StopIteration:
  cli catches only ResolveError? -> StopIteration is not a ResolveError
  ```
  `rigma update` dies with a bare traceback and — because the exception escapes before the `shutil.rmtree(tmp)` on line 59 — leaves `~/.rigma/registry.tmp` behind, which the next `rigma update` then `rmtree`s at line 54. The neighbouring malformed case (`missing gpus.json`) already raises a named `RuntimeError`.
- **Cause:** run 2's own improvement list in `06-resolve-fit-probe.md` flagged this exact line ("a registry zip with no top-level directory raises StopIteration out of `rigma update`"), but no `06-*` finding id was assigned to it, so no commit was made. Confirmed still present at HEAD.
- **Fix:** raise the same `RuntimeError` shape as the sibling check (**implemented, `032db43`**), cleaning up the half-extracted tree on the way out.
- **Verified:** `python .scratch/00-recon/r3d-r4.py` section D (transcript above: `RAISED StopIteration`). Regression tests `tests/test_registry_update.py::test_update_refuses_a_zip_with_no_top_level_directory` (fails before) and `::test_update_refuses_a_wrapper_that_is_missing_gpus_json`.

### 06R3-6 [MEDIUM] the calibration key is `model:quant:backend` — a placement measured on a different GPU is replayed if the desktop footprint happens to match
- **Where:** `src/rigma/resolve.py:84` (`_apply_calibration`) and `src/rigma/server_ops.py:219-220` (`_measured_placement`) — the same key, built in two places
- **Trigger:** two machines (or one machine after a GPU swap) with the same `model_slug`, `quant`, `backend`, `ctx` and engine build, and a desktop VRAM footprint within `VRAM_DRIFT_TOLERANCE_MB` (700 MiB) of the stored `vram_used_mb`. The `flags` payload that gets replayed is exactly the placement that decides whether the weights fit (`ngl`, `n_cpu_moe`, `cache_type_k/v`, `spec_type`).
- **Consequence:** a placement measured on a 24GB card is applied on a 16GB card. `calibration_stale` cannot see it: it compares the *desktop's* VRAM reading, never the card's capacity or identity. `_desktop_vram_mb` even clamps that reading against the *current* card's total (`resolve.py:66-70`), so the check is defined in terms of the machine it is supposed to be validating. On Linux the check is skipped entirely — `probe.gpu_used_mb` returns `None` off Windows, `save_calibration` therefore writes no `vram_used_mb`, and `calibration_stale:161-162` returns `None` ("read leniently") for every entry.
- **Cause:** the key was chosen before the drift check existed and was never widened. `HardwareProfile.fingerprint` (`models.py:86-89`, `vendor-slug/ram-<tier>/<os>`) is the identity the registry combos are already keyed by and is not used here.
- **Fix:** not implemented — see "Could not execute / deliberately not fixed" below for why. The smallest correct change is to add `profile.fingerprint` (or at least the primary GPU slug) to the key in both places, with a read-fallback to the legacy two-part key so existing rows are not silently retired.
- **Verified:** `python .scratch/00-recon/r3d-cal.py` prints the stored entry (`{'measured', 'schema', 'engine', 'vram_used_mb', 'ctx', 'date'}` — no GPU field) and `r3d-r4.py` section E shows the same calibration being applied to two different profiles whose only difference is `vram_used_mb` vs a live probe reading.

### 06R3-7 [LOW] `list_models` swallows every fit exception, so a broken fit is indistinguishable from "no profile"
- **Where:** `src/rigma/hangar.py:885-890`
- **Trigger:** any exception inside `quant_verdicts` — a `KeyError` from an unexpected cache type, a `ZeroDivisionError`, an `AttributeError` from a spec written by an older version.
- **Consequence:** `fits` stays `[{} for _ in spec.ggufs]` and every row renders with an empty fit dict: no size verdict, no ctx, no reason. This is the same silent-blank the 06-1 finding described for the Models page, and it is what makes 06-3-style `KeyError`s invisible in the first place:
  ```
  fit verdict on the row: {}
  (a KeyError from the fit math is indistinguishable from 'no profile')
  ```
- **Cause:** `except Exception: pass` with the comment "a fit we can't compute must not blank the page" — but blanking the *fit column* is exactly what it does. The exception is not logged, so nothing anywhere records that the arithmetic failed.
- **Fix:** not implemented (a behaviour change to a user-facing surface, and the log call would need the project's logger convention). The smallest change is to `_log.exception(...)` in the handler and put `{"ok": False, "speed": "no", "error": "fit failed"}` in the rows, matching the shape `quant_verdicts` already emits at `resolve.py:560-562`.
- **Verified:** `python .scratch/00-recon/r3d-r4.py` section A (transcript above).

### 06R3-8 [LOW] `gguf_meta` accepts every GGUF version `>= 2`, including formats it cannot read
- **Where:** `src/rigma/gguf_meta.py:99-100`
- **Trigger:** a file whose header declares `version = 4`, `5`, `100` or `4294967295`.
- **Consequence:** the header is parsed as if it were v2/v3 and every field is silently wrong:
  ```
  v2: ACCEPTED   v3: ACCEPTED   v4: ACCEPTED   v5: ACCEPTED
  v100: ACCEPTED v4294967295: ACCEPTED
  ```
  The module's whole premise is that the header is the file's self-description; accepting a version whose layout it does not know turns a parse failure into a wrong `n_layers`/`kv_heads`/`context_length`, which is the "wrong number means a model that will not load" case. `_read_meta` already rejects the other end (`version < 2`), so the asymmetry is not deliberate.
- **Cause:** the bound was written as a floor only. Note the direction is deliberately generous: refusing a genuinely-newer v4 would break a valid file, so this is reported rather than fixed.
- **Fix:** not implemented. The safe form is an allow-list of the versions whose layout is actually implemented (`{2, 3}`) with a `GgufParseError` naming the version — but that trades a wrong number for a refusal, and the right answer depends on whether GGUF v4 exists, which cannot be checked from this clone (no network, no engine).
- **Verified:** `python .scratch/00-recon/r3d-r5.py` section C (transcript above).

### 06R3-9 [LOW] a tensor dimension that overflows produces an unbounded parameter count, and `moe_from_probe` keeps the ratio
- **Where:** `src/rigma/gguf_meta.py:167-175` (`n *= _read(f, "<Q", 8)`, `idx.params += n`), consumed at `hangar.py:317-324`
- **Trigger:** a header with `rank = 8` and every dim `2**64 - 1` (8 dims is inside the `n_dims > 8` allowance).
- **Consequence:** Python's arbitrary-precision int means there is no overflow — there is a silently enormous number:
  ```
  params overflow: 13407807929942597093759315203840991004188031530987402520718628407015669769757842313630909715223819254400837606388228716074377856895316039510175975812890625
  ```
  `moe_from_probe` then divides two of them, so `expert_weight_fraction` survives as a plausible-looking `0.5` (not the `0.0` I expected), `total_b` becomes `2.04e81`, and `_spilled` reports `0.5` for a fully-offloaded MoE. Nothing downstream bounds `params`; `quant_quality.measured_bpw` rounds it to `0.0` and `label_overstates` then reports a −100% drift as a real finding.
- **Cause:** the tensor table is treated as the file's own inventory (correctly — that is why it is read at all), but its *dims* are multiplied without any plausibility bound, while `n_tensors`, `n_kv`, `n_dims` and string lengths all have one.
- **Fix:** not implemented — a bound needs a defensible constant (the largest real model is ~1e12 params; `params > 1e15` is certainly corrupt), and I could not verify a threshold against real files here.
- **Verified:** `python .scratch/00-recon/r3d-gguf.py` section E and `r3d-r5.py` section D (transcripts above).

### 06R3-10 [LOW] `engine_log.findings` attributes the last matching line to the current run, whatever run wrote it
- **Where:** `src/rigma/engine_log.py:95-97` (`"example": hits[-1][:300]`)
- **Trigger:** a log file that spans several launches and whose LAST occurrence of a pattern is not from the live engine — a rotated/appended log, or a previous run whose warning line sorts after the current run's clean start.
- **Consequence:** the Engine page shows a diagnostic (`cache_reuse_disabled`, `swa_disabled`, `kv_cache_full`, …) with `confirmed_here: True` for a flag state the running engine may not be in. `confirmed_here` is a static per-pattern constant, not a statement about this run, so nothing in the payload lets a caller tell.
  ```
  findings: [('cache_reuse_disabled', 2, True)]
  ```
- **Cause:** the docstring asserts "the last occurrence: on a log covering several launches it is the one describing the engine currently running". That holds only if the current run is the last writer, which the function cannot know from `log_text` alone — it is a pure function over text by design.
- **Fix:** not implemented (would need the current engine's start marker threaded in from `server_ops.log_tail_bounded`, which is a signature change). Reported so the claim in the docstring is not trusted as a guarantee.
- **Verified:** `python .scratch/00-recon/r3d-r4.py` section F (transcript above).

---

## Considered and cleared (verified, not assumed)

- **Run-2 fix 06-1 is complete.** `_fit_with_cache` guards `spec.n_layers <= 0` in BOTH branches (`resolve.py:409`, `:429`) and `hangar.spec_fields_from_probe` floors `n_layers` at 1 (`hangar.py:333`). Verified: `n_layers=0`, `n_layers=-3`, `native_ctx=0`, `native_ctx=-1`, `gguf.bytes=0` all go through `fit_gguf` and `resolve()` without raising (`r3d-fit.py` sections 3/3b).
- **Run-2 fix 06-3 is complete in the fit math.** `CACHE_BYTES` carries all seven GGML types with the right block sizes (`models.py:20-28`), `CachePolicy.k/v` and `ComboFlags.cache_type_k/v` both validate against it, and `quant_verdicts(kv="q4_1"/"bf16"/"q5_0")` now returns a verdict instead of a `KeyError` (`r3d-r4.py` section B). The remaining gap is the HTTP boundary (06R3-3), not the arithmetic.
- **Run-2 fix 06-4 holds for `resolve._apply_calibration`** — `ComboFlags.model_validate` with an all-or-nothing `ValidationError`/`TypeError` handler (`resolve.py:113-125`). 06R3-1 is the *other* merge site, which the fix did not cover.
- **Run-2 fix 06-5 holds.** `_next_ctx_rung` steps TO the floor, not past it, and terminates: `native_ctx ∈ {1, 2048, 8192, 10000, 12288, 32768}` all produce a ladder whose last rung is exactly `_ctx_floor` and whose next step is below it (`r3d-fit.py` section 2). `native_ctx=0` yields an empty ladder, which is correct (a model with no declared window is not fitted at a window it does not have).
- **Run-2 fix 06-6 holds.** `_as_int` names the key and raises `GgufParseError`; the head-count list uses `max(..., default=0)`. Verified across `block_count`/`context_length`/`head_count`/`head_count_kv`/`embedding_length`/`sliding_window`/`expert_count` as empty arrays and as strings (`r3d-gguf.py` sections A/A2/A3).
- **Run-2 fix 06-7 holds.** `draft_cache_mb(spec, ctx, spec_type, n_max)` no longer takes `kv`; the measured constants still reproduce the recorded points exactly (`465.0` MiB @16K, `565.0` MiB @32K for depth 1) and `with_launch_overheads` converts MiB→bytes correctly (`r3d-fit.py` section 11).
- **Run-2 fix 07-3 holds on every path I could reach.** The `dest.exists()` short-circuit verifies size and then sha256 before returning (`hangar.py:1311-1325`), and on a mismatch it falls through to refetch. Verified by construction with a wrong-size dest and a wrong-hash dest.
- **Run-2 fix 07-4 holds.** The cancel flag is set under `_PULL_LOCK`, checked per chunk (`hangar.py:1391`) and re-checked under the same lock immediately before `os.replace` (`:1288-1292`); a cancelled install discards the `.part` instead of renaming it in. The pull key is `f"{slug}::{file}"` — slug is the model's stable identity (renaming writes a new spec and the old key is unreachable), and a leftover `.part` from a cancelled pull is only trusted when its note matches the full `(repo, file)` identity (`_part_owner`/`_part_ident`, `:1201-1215`). Verified: the note round-trips, a different `(repo, file)` is rejected, and a fresh pull resets `cancel` to `False` (`r3d-hangar.py` section F/G).
- **Run-2 fix 07-5 holds for the fresh-pull case.** The gate runs before the thread is spawned and refuses with the GB shortfall named. It does not count an existing `.part` toward free space and does not re-check per retry — noted, not a finding: the check is `want + 2 GiB <= free` on a destination that does not exist yet, and a resumed pull needs at most the remainder.
- **Run-2 fix 07-6 and 07-7 hold** (`delete_model` drops `templates/<slug>.jinja` and every `<slug>:` calibration row at `hangar.py:1027-1049`; `_AUX_MARKERS` is tokenised via `_AUX_TOKEN_SPLIT` at `hf_browse.py:35`).
- **The `NUL` / device-name class is closed at every sink I tested.** `GgufFile.file` (`models.py:173`), `hangar._check_model_file_name` (`hangar.py:133-136`) and `hangar.custom_spec_path` (`:91-93`) all reject `NUL`, `nul.gguf`, `con.gguf`, `CON`, `com1.gguf`, `lpt9`, `aux.gguf`; nested `Q4_K_M/model.gguf` is still accepted, and 14 hostile forms are rejected (`r3d-hangar.py` section H). Windows trailing-space/dot normalisation (`a.gguf ` → `a.gguf`) is benign here: both spellings resolve to one path, so a spec can only carry one of them, and the reserved-device check strips trailing dots and spaces before matching.
- **`GgufFile.file` traversal confinement is real** for all seven forms 06-8 listed plus `sub/../../x.gguf` and `a//b.gguf`.
- **The GGUF parser does not hang or allocate unboundedly on the inputs I could construct.** Declared string length > file → `GgufParseError: truncated gguf header`; array count `2**63` → the item loop is bounded by the file's own length and the read fails on the second element (`elapsed 0.00s`); `n_tensors = 2**40` → `implausible tensor count`; nested array 20 deep → `array nesting too deep`; unknown value type / unknown array element type → `GgufParseError: unknown gguf value type N`; zero-length / magic-only / bad-magic files → `GgufParseError`; `n_kv = 100_000` with an empty body → `truncated`; duplicate and empty metadata keys are accepted (last wins), which matches GGUF's map semantics (`r3d-gguf.py` sections B/C/D/F/G/H/I/J/K/L/M).
- **`head_dim` cannot divide by zero.** `heads == 0` with a present `embedding_length` short-circuits the `// heads` at `gguf_meta.py:317-320` and reports `head_dim: 0`; `install_model` (`hangar.py:808`) and `_spec_from_repo` (`hf_browse.py:229-230`) both refuse a spec with `kv_heads <= 0` or `head_dim <= 0`, so the zero-KV false-positive at `fit_gguf` is unreachable from either install path (`r3d-fit.py` sections 3/3c).
- **`_configured` does not mutate the caller's spec.** `model_copy(deep=True)` before any write; the shared spec's `cache_type_policy` is unchanged after `quant_verdicts(kv="q4_0")` (`r3d-r4.py` section C).
- **`_distinct_quants` preserves list length** for `[]`, single, duplicate and nested-path inputs, so `zip(spec.ggufs, labels, axes, fits)` in `list_models` cannot silently drop rows (`r3d-r5.py` section B).
- **`fallback_plans` survives a registry model with `ggufs: []`** — `have_ggufs` filters it out before the `ggufs[-1]` subscript (`r3d-r5.py` section A).
- **`quant_quality.quality_of` substring matching is sound**: longest-first means `BF16` beats `F16`, `Q2_K_L` beats `Q2_K`, `UD-Q3_K_XL` beats `Q3_K_XL`, and `IQ2_XS` does not match `IQ2_S` (re-checked against the table by hand).
- **`kvcache.fingerprint` is pure** over `FINGERPRINT_FIELDS` with a NUL sentinel that distinguishes "absent" from `""`, and `config_of` is the single definition shared by the launch and restore paths. The `flash_attn` field added by 02-3 is present in both the field tuple and `config_of`, so the two cannot drift.
- **The `_budgets` reserve direction is safe**: `reserve = max(floor, measured) + COMPUTE_BUFFER_MB`, a measurement can only shrink the budget, and a reading at or above the card total is discarded rather than believed.

## Could not execute / deliberately not fixed

- **No network, no engine, no real model files** (per the brief). Every number above comes from synthetic GGUF bytes built with `struct.pack`, monkeypatched helpers, or arithmetic run in-process. Specifically NOT executed: a real Hugging Face `/resolve` response (so 06R3-2's chunked-transfer case is reasoned from `_object_size`'s own contract plus the reproduced 9-byte install, not from a live server), a live `llama-server` launch (so "would the engine refuse `-ngl 99` on a 0-layer model" stays an open question), and Vulkan/NVML enumeration.
- **06R3-6 not fixed, deliberately.** Widening the calibration key touches a user-editable file whose rows are written by `bench.save_calibration` and read by two modules; a naive change retires every existing row and splits new ones across two spellings when `profile` is `None` (`perform_switch` accepts `profile=None`). That is a migration, not a one-line fix, and getting it wrong costs a re-tune of every model on the machine. Reported with the exact smallest-correct-change sketch instead.
- **06R3-6, 06R3-7, 06R3-8, 06R3-9, 06R3-10 not fixed** — each needs either a defensible constant (the `params` bound), a signature change (`engine_log`), a UI-visible behaviour change (`list_models`), or a fact I cannot check without network/engine (whether GGUF v4 exists). Reported rather than guessed at.
- **`--kv-unified` allocation semantics and `-ngl 99`'s meaning** remain unverified against llama.cpp, exactly as the 06 findings recorded. Nothing in this run changed that.
- **The pytest suite was run only for the affected files.** The full suite (~6 minutes) was not run.

<!-- coverage: src/rigma/resolve.py L1-856 (read), src/rigma/models.py L1-437 (read), src/rigma/quant_quality.py L1-244 (read), src/rigma/probe.py L1-301 (read), src/rigma/gguf_meta.py L1-417 (read), src/rigma/registry.py L1-151 (read), src/rigma/hf_browse.py L20-60/L110-300/L359 (read), src/rigma/hangar.py L60-220/L290-350/L592-712/L729-1000/L1096-1430 (read), src/rigma/bench.py L40-169 (read), src/rigma/kvcache.py L1-199 (read), src/rigma/engine_log.py L1-99 (read), src/rigma/server_ops.py L198-257/L296-320/L380-570/L600-672 (read), src/rigma/serve.py L3660-3739 (read), src/rigma/state.py (grep), tests/test_audit_fit.py L1-80 (read), tests/test_resolve_calibration.py (read), tests/test_registry_update.py (read), tests/test_phase0_contracts.py L240-282 (read), tests/test_audit_hangar.py L400-449 (read), tests/test_server_ops.py L1-140 (read), tests/test_gguf_meta.py (grep + L20-62), tests/test_gguf_header_types.py (grep) -->
