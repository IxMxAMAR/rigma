ITEM: B4b-schema required-container fix (75960df)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes — reproduced exactly. With both new guards deleted from `problemFor` (frontend-v2/src/chat/governance.ts:446-448, 462-464) and the committed tests unchanged: `Test Files 2 failed (2) / Tests 7 failed | 78 passed (85)`, the same 7 tests the implementer named (governance shapes 1, 2, 2b, 6, 9, 10 + QuestionForm "will not submit a required array whose only row is blank"). An independent probe at that pre-fix state also showed shape 1 `ready=true, problem="", answer={}` and 198 ready-but-required-absent violations in a 4000-trial fuzz. Restoring the guards via `git checkout --` returned 85 passed and 0 fuzz violations.
TEST COMMAND: npx.cmd vitest run src/chat/governance.test.ts src/chat/QuestionForm.test.tsx   (cwd: C:\ComfyUI\RD\rigma-review\frontend-v2)
TEST OUTPUT (before fix):
  ❯ src/chat/governance.test.ts (73 tests | 6 failed)
  ❯ src/chat/QuestionForm.test.tsx (12 tests | 1 failed)
  Test Files  2 failed (2)
       Tests  7 failed | 78 passed (85)
TEST OUTPUT (after fix):
  ✓ src/chat/governance.test.ts (73 tests)
  ✓ src/chat/QuestionForm.test.tsx (12 tests)
  Test Files  2 passed (2)
       Tests  85 passed (85)
RULE-13 STATE CHECKED: (independent probe, 22 shapes + 4000-trial fuzz; not their test)
  - required array, one good row + one blank row (strings and objects): ready=true; answer `{"tags":["a"]}` / `{"steps":[{"cmd":"make"}]}` — only the blank dropped. Correct.
  - required number of `0`: `{n:"0"}` -> ready=true, answer `{"n":0}`; `0` is a definite answer, NOT absent. `{n:""}` -> blocked. Correct.
  - required boolean with `default:false`: `questionDefaults` -> `{flag:false}`; ready=true; answer `{"flag":false}`. Correct.
  - required `$ref` and required `allOf` property: `schemaField` degrades both to a text field; filled -> ready=true/answer string; blank -> blocked. The required invariant holds (a `$ref` naming an object is still sent as a string — pre-existing form limitation, untouched).
  - required object, all children optional (shape 2) and touched-then-blank (2b): now BLOCKED (ready=false, `“Config” is required`); the only unblock is filling the OPTIONAL child (`{cfg:{a:"x"}}` -> ready=true). Strict JSON Schema accepts `{}` here, so this is an over-block. Judged acceptable-but-noted: it is the fail-safe direction, the implementer documented it, and the alternative (emit `{}`) would change the "empty container is ABSENT" contract. NIT, not a FAIL.
  - nested required LEAF under an OPTIONAL parent (shape 5): still blocks, byte-unchanged by this commit; JSON Schema would allow omitting the optional parent. Pre-existing and pinned by an existing test (governance.test.ts:418-421). Not a regression.
  - bonus: required array of booleans, one blank row -> ready=true, answer `{"bs":[false]}` (required key present; boolean is always definite). Consistent, not a violation.
  - fuzz (deterministic mulberry32, depth<=2 random schemas x random values): pre-fix 198 ready-but-required-absent violations; post-fix 0 / 4000.
  - single-judge proof by mutation: changing `answerFor` so object/array always return the container made `problemFor` follow exactly (shape 1 -> ready, `{"tags":[]}`; shape 2 -> ready, `{"cfg":{}}`; shape 10 -> ready, `{"steps":[{}]}`) — the guard IS `answerFor`, not a second copy of the rule. The array branch still carries the older `rows.length === 0` short-circuit, but it is currently exactly equivalent to `answerFor`'s absence for zero/undefined rows and errs toward blocking, so the dangerous direction is single-judged.
BREAKS: nothing at the gates. Only behavior change beyond the intended block is the shape-2/2b over-block above (no file:line regression).
NITS:
  - Shape 2/2b over-block: a schema-valid `{}` for a required object whose children are all optional is now unsubmittable, and the message `“Config” is required` describes absence where the container is merely empty (governance.ts:446-448).
  - Shape 5 over-block is pre-existing and unchanged (governance.ts:436-439) — out of scope, but the fix does not address it.
  - `$ref`/`allOf` required properties degrade to a text box (governance.ts:239-264); a `$ref` to an object would be POSTed as a string. Pre-existing, untouched by this commit.
  - The array branch keeps a restated emptiness predicate (`rows.length === 0`, governance.ts:454) alongside the new `answerFor` guard; currently equivalent, but it is the one place a future `answerFor` change could over-block rather than drift into ready-but-absent.
REASON: The original defect reproduces to the exact test counts and the fix closes it — all 22 probed shapes plus a 4000-trial fuzz now satisfy "ready => every required key present", and a mutation of `answerFor` proves the guard is single-judged. All gates pass (tsc 0; build twice idempotent and byte-identical to the committed ui_v2; vitest 56 files / 709 tests; ruff clean), the permission path is untouched, and OD-12/OD12-n1 still hold; only the documented shape-2/2b conservative over-block and pre-existing shape-5/$ref limitations remain as nits.
