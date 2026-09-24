// CONSTITUTION §33: an empty screen is an invitation to act — centered, muted,
// a mono-line glyph and one primary action. Never blank, and never a bare
// "nothing here".
//
// Its partner is LoadError, and the split is the point: "there is nothing here"
// and "the request failed" must never render as the same thing (AUDIT F11-3,
// F11-5). An empty state means the fetch SUCCEEDED and returned nothing; a
// failed one gets LoadError with a retry.
export default function EmptyState({ title, body, actionLabel, onAction }: {
  title: string;
  body: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="text-center pt-16 pb-4">
      <div className="font-mono text-[14px] text-muted mb-2" aria-hidden="true">
        ◇
      </div>
      <div className="font-mono text-[12px] text-muted uppercase tracking-[0.1em] mb-2">
        {title}
      </div>
      <p className="text-secondary text-[13.5px] max-w-[420px] mx-auto">
        {body}
      </p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-3 rounded-md bg-amber/15 text-amber px-3 py-1.5 text-[13px] font-semibold"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
