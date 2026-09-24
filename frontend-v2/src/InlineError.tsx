// A refused action on a row — a delete the server would not perform.
//
// Distinct from LoadError: there is nothing to retry, and the list is unchanged.
// It exists because `await fetch(...DELETE)` followed by a refresh renders a
// refused delete as a successful one — the row simply stays, with no reason
// (AUDIT F11-4).
export default function InlineError({ message }: { message: string }) {
  return (
    <div role="alert"
         className="rounded-md bg-red/10 text-red px-3 py-2 text-[12.5px] mb-2">
      {message}
    </div>
  );
}
