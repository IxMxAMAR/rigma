import type { HarnessInfo } from "../lib/api";

/** The disclosures that describe the agent backend this chat is actually on.
 *
 *  Extracted verbatim from the Sidecar panel so it can be exercised by a
 *  render test: it is pure props, no store and no effects. `selectedHarness`
 *  is the menu row matching `harness`; the stored name may not be on the menu
 *  (a moved checkout), in which case it is `null`/`undefined` and only the
 *  permission sentence's guard is consulted. */
export interface HarnessFactsProps {
  selectedHarness: HarnessInfo | null | undefined;
  harness: string;
  drifted: HarnessInfo | null | undefined;
}

export default function HarnessFacts({
  selectedHarness,
  harness,
  drifted,
}: HarnessFactsProps) {
  return (
    <>
      {/* R3-HARN-2: the cost of the backend you are ACTUALLY on.
          The same `title`-on-an-`<option>` mistake as the picker above, one
          level worse: the list of what a backend does not get from Rigma was
          rendered only for backends that are unusable, so choosing a WORKING
          external agent hid the entire disclosure — including "the sandbox is
          pinned danger-full-access, so a confined profile does not survive the
          seam", which is the sentence that explains the permission selector.
          `unsupported` is non-empty for every external backend, so this is the
          normal case, not an edge one. */}
      {/* The other direction, and it goes FIRST: what this backend can do is
          the question a user is asking when they open this panel. A backend
          whose capabilities Rigma supplies — DSH's goals, subagents, todos,
          skills, plan mode and filesystem tools all come from Rigma's patch,
          not from the minimal profile — reads as featureless without this. */}
      {selectedHarness && (selectedHarness.capabilities?.length ?? 0) > 0 && (
        <details className="rounded-md bg-surface px-2.5 py-1.5 text-[11.5px]" open>
          <summary className="cursor-pointer text-secondary">
            what Rigma gives {selectedHarness.label}
          </summary>
          <ul className="mt-1 flex flex-col gap-0.5 pl-3 text-muted leading-snug">
            {(selectedHarness.capabilities ?? []).map((c) => (
              <li key={c}>+ {c}</li>
            ))}
          </ul>
        </details>
      )}
      {selectedHarness && selectedHarness.runnable && selectedHarness.installed
        && selectedHarness.unsupported.length > 0 && (
        <details className="rounded-md bg-surface px-2.5 py-1.5 text-[11.5px]">
          <summary className="cursor-pointer text-secondary">
            what {selectedHarness.label} does not get from Rigma
          </summary>
          <ul className="mt-1 flex flex-col gap-0.5 pl-3 text-muted leading-snug">
            {selectedHarness.unsupported.map((u) => (
              <li key={u}>— {u}</li>
            ))}
          </ul>
        </details>
      )}
      {/* Said out loud, and only when it is TRUE. A backend that moved under us
          is the one thing about this seam the owner cannot see any other way:
          the turn keeps working right up until it quietly does not, and the
          adapter was measured against a build that is no longer the one
          installed. `null` says nothing — "I could not tell" is not "it
          agrees", and it is not a warning either. */}
      {drifted && (
        <div className="rounded-md bg-amber/10 px-2 py-1 text-[11.5px] text-amber">
          {drifted.label} is {drifted.version || "a different build"} — its
          adapter was measured against {drifted.verified || "nothing"}. Turns
          may still work; if one misbehaves, that is the first thing to check.
        </div>
      )}
      {/* R3-HARN-1: the setting is that agent's vocabulary, and showing it on
          Rigma's own loop would offer a knob that does nothing. But that is
          exactly what it was ALSO doing for a backend whose adapter ignores the
          field — DSH's `run_turn` accepts `permission` and drops it, because
          DSH's confinement is its own bundle's business. So the selector is now
          gated on the backend actually applying it, and the case where it does
          not gets a sentence instead of a control that lies.

          A backend that does not send the field at all reads as honouring it,
          which keeps the control rather than silently removing it. */}
      {harness !== "native" && selectedHarness?.honours_permission === false && (
        <div className="rounded-md bg-surface px-2 py-1 text-[11.5px] text-muted">
          <span className="text-secondary">{selectedHarness.label}</span> sets
          its own permission policy, so this chat&apos;s permission setting does
          not apply to it. Pick a different agent to control how much it may do
          without asking.
        </div>
      )}
    </>
  );
}
