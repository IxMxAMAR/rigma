// A8c: the sentence a switch or a relaunch answers with — today the KV-cache
// restore it refused, or the cache it stepped down to. The server has always
// returned it (`server_ops.perform_switch` adds `notice`); the body was typed
// `unknown` and every caller discarded it, so the sentence never reached the
// screen.
//
// WHY THIS IS ITS OWN COMPONENT, and not four copies of a conditional. Three
// surfaces show the notice (Engine, Models and the model picker) at three
// different type scales, so the ONE thing they must agree on — a notice is
// drawn NEUTRALLY, never in the error slot, and only when there is one — is
// stated once here. It is pure and props-only, so a render test can pin that
// markup without a store or a DOM event (which is the whole reason the wave-1
// wiring was unasserted: see W5F1a).
export default function SwitchNote({ note, className }: {
  /** `switchNotice()`'s output — `""` when the switch had nothing to say. */
  note: string;
  /** The surface's own type scale and spacing. The tone stays the caller's,
   *  because every existing site already drew the notice in `text-secondary`:
   *  a step-down is not a failure and must not look like one. */
  className?: string;
}) {
  if (!note) return null;
  return (
    <div
      className={
        className ?? "rounded-md bg-surface text-secondary px-3 py-2 text-[13px]"
      }
    >
      {note}
    </div>
  );
}
