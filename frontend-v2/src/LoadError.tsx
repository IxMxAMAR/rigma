// A failed list fetch, said out loud, with a way to try again.
//
// The surfaces used to render a failed request as "you have none", which sends
// the user to fix the wrong thing: the Models page invited a download for a
// model already on disk, Workflows claimed no workflows existed (AUDIT F11-3).
export default function LoadError({ message, onRetry }: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div role="alert"
         className="rounded-md bg-red/10 text-red px-3 py-2 text-[12.5px] mb-3 flex items-center gap-3">
      <span className="flex-1 min-w-0">{message}</span>
      <button
        onClick={onRetry}
        className="shrink-0 rounded-md bg-surface hover:bg-float px-2.5 py-1 text-[12px] text-secondary"
      >
        retry
      </button>
    </div>
  );
}
