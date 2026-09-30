# OWNER-DECISIONS — Rigma improvement program, 2026-09-30

Decisions that are the owner's, not the Head Agent's (brief §0.8). Each has the options, the
evidence, a recommendation, and what happens if nothing is decided. Nothing here has been
implemented; nothing here changes a global default.

---

## OD-1 — What should the default run profile be? (finding 13-2, prompt injection → secret exfiltration)

**State today:** the run profile defaults to `"all"` (`serve.py:2630`, `:6262`), which grants the
full network + delete surface to an autonomous run. The finding's other three mitigations are
already in the tree and both the read-confinement and outbound-post grants default **OFF**
(`serve.py:2642-2643`), and an unknown profile is currently *silently coerced* to `"all"`
(`serve.py:6324`) — that coercion is being fixed as item A7 regardless.

**Options**
1. Keep `"all"` as the default. A run that silently cannot reach the network is a run that fails
   in a confusing way; the grants exist and the UI shows them.
2. Default to `"read"` (no network, no delete). Safer by construction; a run that needs to fetch
   something now stops and asks, which the approval UI supports.
3. Default to a new `"ask"` profile: every network/delete action raises an approval prompt.
   Safest, but it turns a long autonomous run into a click-through.
4. Leave the default alone and make the *first* launch of a new run show the profile chooser with
   `"all"` preselected — informed consent without changing behaviour.

**Evidence:** `docs/review/findings/13-adversarial-security.md` (finding 13-2); the grants are at
`serve.py:2642-2643`; the profile is read at `serve.py:2630,6262`.

**Recommendation: option 4.** It is the only one that does not change a global default, it costs one
dialog, and the finding's real risk is *uninformed* consent rather than the profile value. If you
want a behavioural change, option 2 is the next best.

**If nothing is decided:** nothing changes; the current default stands. This is not a blocker.

---

## OD-2 — `view_image` reads absolute paths outside the workspace; `copy/move_files` can write anywhere (R3-9, R3-10)

**State today:** `read_file` refuses an absolute path outside the workspace without a grant, but
`view_image` does not (`findings-r3/04-sandbox-r3.md:225`), and `copy_files`/`move_files` accept any
absolute destination — including the Startup folder (`:250`). Both are read/write confinement gaps,
not exploits by themselves, but together they are a file-exfiltration and a persistence primitive
for a model that has been talked into it.

**Options**
1. Route `view_image` through the same confinement check as `read_file` (needs a grant for outside
   paths), and restrict `copy/move_files` destinations to the workspace + a configured allowlist.
   Consistent, and the grant mechanism already exists.
2. Add `view_image` to the confinement check only, and leave `copy/move_files` as-is.
3. Add both to a new "write outside the workspace" grant that defaults OFF, and keep the current
   behaviour when the user turns it on.
4. Do nothing; document the gap.

**Recommendation: option 1**, with the destination allowlist seeded from the paths the owner already
works in. It reuses the existing grant machinery (`serve.py:2642-2643`) rather than inventing a
second one, and it is the smallest change that closes both.

**If nothing is decided:** nothing changes. This is a security-policy trade-off and therefore not
the Head Agent's call.

---

## OD-3 — A pre-fix RAG index still holds secrets until it is reindexed (R3-11 residual)

**State today:** the credential-`exclude` denylist fix (`R3-17`) applies to *new* indexing. An index
built before the fix still contains whatever was indexed. Reindexing is the fix, but reindexing
walks and re-reads the owner's documents — which the Head Agent must not trigger.

**Options**
1. Reindex on the next RAG use, automatically, once, and say so in the log.
2. Offer a "rebuild index" button in the RAG panel and leave the decision to the owner.
3. Do nothing.

**Recommendation: option 2.** Auto-reindexing touches the owner's files without being asked, which
§0.8 excludes.

**If nothing is decided:** the stale index keeps its old contents. Recorded here so it is not lost.

---

## OD-4 — LoRA attach (`--lora`/`--lora-scaled` + `/lora-adapters`)

**State today:** Rigma has no LoRA code at all. The engine accepts `--lora`/`--lora-scaled` and the
server exposes `/lora-adapters` for runtime scaling. The owner trains LoRAs, so this is the one
lever in the R4 survey that is a missing *feature* rather than a missing knob.

**Options**
1. Add a per-model LoRA list (path + scale) that is appended to the launch argv, plus a
   `/lora-adapters` passthrough so scale can change without a relaunch. A real feature: a model
   editor row, a store, a UI control, tests.
2. Add the argv fields only (CLI + `LaunchDefaults`), no UI.
3. Defer.

**Recommendation: option 1, but as its own program** — it is larger than any single item in this
run's backlog and deserves its own design note. Recorded so it is not rediscovered.

**If nothing is decided:** no LoRA support; nothing regresses.

---

## OD-5 — Two fit terms that can only be settled with the card (NEEDS-GPU)

The fit currently charges the whole GGUF file to VRAM, charges the recurrent state zero (being fixed
as A2), and uses a flat `COMPUTE_BUFFER_MB = 150` (`resolve.py:28`). A real load log
(`.scratch/prism-v.log`) shows `CPU_Mapped 322.07 MiB` + `ROCm0 6539.67 MiB` for a 6872 MiB file,
`compute buffer 410.28 MiB`, `RS 149.62 MiB`, `graph splits 2`.

- **`token_embd` host residency.** Whether the host-resident 322.07 MiB is a property of this
  fork/model or of hybrid models in general is UNVERIFIED. Subtracting it universally could
  over-commit a model that does put it on the card. Needs a Rigma-launched log at `--parallel 2`.
- **`COMPUTE_BUFFER_MB`.** The 150 was deliberately lowered from 400 because it was obtained by
  differencing measured VRAM and already contains the draft head's buffers (`resolve.py:14-28`).
  Raising it back would double-count. Needs a measurement, not a guess.

**Recommendation:** run the extended plan in `docs/review/findings-r3/37a-262k-verification-plan.md`
(the Head Agent must not; §0.1) and hand the two numbers back. Both are recorded as `NEEDS-GPU` in
`BACKLOG.md` (E1, E2).

**If nothing is decided:** the RS term (A2) is still correct and strictly improves the plan; the
other two stay as they are, with the divergence now *detectable* because A17/S2 parses the engine's
own log.

---

## OD-6 — DSH ACP client, and answering `ask_user`

**State today:** the DSH ACP client is not on the wire (`findings-r3/28-harness-transport-matrix.md`
§320), and mcode `ask_user`/plan are gated on the client declaring `elicitation.form`/`plan`, which
Rigma does not (`28 §6a`). Answering a question needs a live mcode turn, which §0.1 forbids in this
run.

**Recommendation:** adopt the ACP client — it is the only transport that gives Rigma a real
permission round-trip instead of display-only notifications — and declare the elicitation
capability at the same time. Both need one live mcode session to verify.

**If nothing is decided:** the seam keeps working as it does today; questions continue to be
auto-declined, and the drop is now *visible* (item B2 makes dropped events a notice rather than a
silent discard).
