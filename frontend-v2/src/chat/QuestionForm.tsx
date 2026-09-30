import { useState } from "react";

import {
  questionAnswer,
  questionFields,
  questionReady,
} from "./governance";

/** B4b: the form for an elicitation mcode is BLOCKED on.
 *
 *  WHY THIS IS A FORM AND NOT Allow/Refuse. `ask_user` arrives on the same
 *  `approval/asked` channel as a permission request, and the row fell through to
 *  the permission card — whose buttons POST `{allow}`. The route answers 409 for
 *  a question, so the controls were dead AND the turn stayed blocked until
 *  `QUESTION_WAIT_SECS` (5 s) declined it. The question needs its own answer
 *  shape: `{requestId, answer: {...}}`, where the object's keys are the
 *  `requestedSchema`'s own property names.
 *
 *  Props-only and pure in its decisions (`governance.ts` holds the schema
 *  reading), so the markup and the submitted object are both reachable by a test
 *  without a store or a fetch. */
export default function QuestionForm({ question, schema, onSubmit }: {
  /** The `data.question` sentence, shown verbatim. */
  question: string;
  /** The `data.schema` elicitation schema; `{}` when the server sent none. */
  schema: Record<string, unknown>;
  /** Send the answer. Called only when every required field is filled. */
  onSubmit: (answer: Record<string, unknown>) => void;
}) {
  const fields = questionFields(schema);
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const ready = questionReady(fields, values);

  return (
    <span className="block mt-1 flex flex-col gap-1.5">
      <span className="block text-[12px] text-primary break-words whitespace-pre-wrap">
        {question || "the backend asked a question"}
      </span>
      {fields.length === 0 && (
        /* The server is allowed to send `{}`. Saying so is honest; inventing a
           field name would answer with a key mcode never asked for. */
        <span className="block text-muted leading-snug">
          it sent no form fields, so the answer will be empty.
        </span>
      )}
      {fields.map((f) => (
        <span key={f.name} className="block">
          <label className="flex items-start gap-2 text-[12px]">
            <span className="w-28 shrink-0 text-secondary break-words">
              {f.label}
              {f.required && <span className="text-amber"> *</span>}
            </span>
            {f.kind === "boolean" ? (
              <input
                type="checkbox"
                aria-label={f.label}
                checked={values[f.name] === true}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setValues((v) => ({ ...v, [f.name]: checked }));
                }}
                className="mt-0.5 shrink-0 accent-amber"
              />
            ) : f.kind === "choice" ? (
              <select
                aria-label={f.label}
                value={typeof values[f.name] === "string"
                  ? String(values[f.name]) : ""}
                onChange={(e) => {
                  const value = e.target.value;
                  setValues((v) => ({ ...v, [f.name]: value }));
                }}
                className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                           outline-none"
              >
                <option value="">(choose)</option>
                {f.choices.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            ) : (
              <input
                aria-label={f.label}
                value={typeof values[f.name] === "string"
                  ? String(values[f.name]) : ""}
                onChange={(e) => {
                  const value = e.target.value;
                  setValues((v) => ({ ...v, [f.name]: value }));
                }}
                className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                           outline-none"
              />
            )}
          </label>
          {f.description && (
            <span className="block pl-28 text-[11px] text-muted leading-snug">
              {f.description}
            </span>
          )}
        </span>
      ))}
      <span className="flex items-center gap-2">
        <button
          type="button"
          disabled={!ready}
          onClick={() => onSubmit(questionAnswer(fields, values))}
          className="rounded bg-surface px-2 py-0.5 font-mono text-[10.5px]
                     text-moss hover:bg-panel disabled:opacity-50"
        >
          send answer
        </button>
        <span className="text-muted">
          {ready ? "— mcode is waiting on this"
                 : "— fill the required fields first"}
        </span>
      </span>
    </span>
  );
}
