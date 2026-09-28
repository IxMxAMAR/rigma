# R3-UI-1 — controls that render as live and do nothing

**Status:** fixed, tested, bundle rebuilt.
**Reported by:** the owner, testing `Run.bat`: *"I wasn't able to Switch the harness
or Check any buttons like allow reads outside workspace."*
**Commits:** `e9d32d4` (harness, permission, three grants), `8ae190d` (grounded
toggle, Save, method apply).

---

## What was actually wrong

Not a stale build, and not a missing control. `Run.bat` runs the review checkout
(`RIGMA_REPO=C:\ComfyUI\RD\rigma-review`, `PYTHONPATH` set to its `src`), so the
running code WAS the code under review. Both controls exist, render, and look
functional.

The cause is **one guard, copied into six places**:

```ts
if (!currentId) return;
```

`currentId` is `null` until a chat is opened **or the first message is sent** —
because `send` creates one on demand:

```ts
// chatStore.send
let id = active;
if (!id) {
  id = (await api.createSession()).id;
  set({ currentId: id });
}
```

Every *other* per-chat write gave up instead. And the Sidecar renders its controls
regardless of whether a chat exists, because it is opened by the gear button, not
by selecting a chat. So on a fresh page:

| control | what the user saw | what happened |
|---|---|---|
| harness picker | the dropdown changed, then snapped back | store never updated |
| permission picker | same | same |
| three "this chat may" checkboxes | could not be ticked at all | handler returned |
| **Grounded chat** toggle | flipped on screen | write never sent — looked saved |
| **Save** (params + prompt) | button did nothing | handler returned |
| **Apply** a method | button did nothing | handler returned |

The grounded toggle is the worst of the six: it set local state *before* the write
(`setGrounded(next)`), so it appeared to work until the panel was reopened. The
others at least failed visibly.

`MethodCard.apply` is the most telling: it already had an error path, added because
*"a refused apply used to do nothing at all — no state, no message, no way to tell
it from a dead button"* (AUDIT F11-4). The guard one line above that error path was
that same dead button.

## The fix

`ensureSession` on the store — the "which chat does this setting belong to"
question answered once, creating the chat when there is none, exactly as `send`
already did. It returns the id, or `null` after recording `lastError`, so no caller
writes to a chat that was not made.

```ts
ensureSession: async () => {
  const sid = get().currentId;
  if (sid) return sid;
  try {
    const s = await api.createSession();
    set({ currentId: s.id, messages: [],
          harness: s.harness ?? "native",
          permission: s.permission ?? "full", lastError: null });
    await get().loadSessions();
    return s.id;
  } catch (e) {
    set({ lastError: errText(e) });
    return null;
  }
},
```

All six writers now `await get().ensureSession()`.

**Chosen over disabling the controls.** `send` already auto-creates a chat, so a
setting that creates one is consistent with the rest of the app rather than a new
behaviour. Disabling would also need a "start a chat first" affordance on a panel
opened by a gear button — a worse trade for the same fix.

## What is NOT the bug (verified, so the real cause is not mistaken for it)

- **The server accepts every field.** `sessions.MUTABLE_FIELDS` includes
  `harness`, `permission`, `confirm_exec`, `allow_absolute_reads`,
  `allow_outbound_post`, so `update_session` writes them. Had one been missing, the
  write would have been silently dropped *while returning 200* — a worse bug, and
  worth having ruled out.
- **The server validates and refuses rather than falling back.**
  `serve.update_session` resolves the name through `harness.resolve` and 400s an
  unusable one; the chat route checks again at line 4117; `_llm_turn` checks a
  third time for the paths that bypass the route (a run, a macro) and emits an
  `error` event. So a session that asked for DSH can never quietly get the native
  loop.
- **The switch is actually honoured.** `_llm_turn` selects the backend, emits a
  `harness` event **up front** (so the badge appears before the first token), and
  routes a non-native backend to `_external_turn`.
- **All three backends are runnable on this machine.** `harness.list_harnesses()`
  reports `dsh`, `mcode` and `native` as `installed=True runnable=True`, so the
  dropdown options were never `disabled`.

## The sweep, which found four more

The first fix covered three sites. Listing **every** occurrence in the frontend
rather than reading the file found three more handlers (above) plus four
legitimate ones:

- `Sidecar.tsx` lines 399, 623, 982, 1187 — all `useEffect` loaders (fetch this
  chat's grounded flag / params / method / draft). Doing nothing with no chat is
  correct.
- `chatStore.stop` — reachable only from the Escape key while a turn is streaming.
- `WorkspacePanel.tsx:52` — `return null`, a conditional *render*, not a dropped
  write.

Two of the three new fixes needed `ensureSession` added as a per-component
selector; `tsc --noEmit` caught both, which is why the sweep was run before the
commit rather than after.

## Verification

- `tsc --noEmit` clean; **179 vitest tests pass** (up from 174 — five new, covering
  the two cases that matter: the change is not silently lost when the chat cannot
  be created, and an unchanged value does not litter the rail with an empty chat).
- **The bundle was rebuilt and the fix verified present in it.** The pre-commit
  hook exists precisely because committing UI source alone ships the *old*
  interface and every test still passes — `pytest` cannot see the bundle and
  `vitest` tests the source.
- `resources.files("rigma").joinpath("data/ui_v2")` resolves to
  `C:\ComfyUI\RD\rigma-review\src\rigma\data\ui_v2`, so `Run.bat`'s `PYTHONPATH`
  makes the fork's bundle the one served. `index.html` carries `_NO_STORE`, so a
  refresh picks up the new hash; the hashed asset is immutable.

**Rigma was not running when this was found** (the power cut stopped it), so no
restart was needed to clear a stale process — but a Rigma that was already running
would serve the old bundle until restarted.
