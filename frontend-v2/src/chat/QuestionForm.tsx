import { useCallback, useState } from "react";

import {
  questionAnswer,
  questionDefaults,
  questionFields,
  questionProblem,
  type QuestionField,
  type QuestionInput,
  type QuestionValues,
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
 *  B4b-schema: the schema is not guaranteed flat. A property may be an `object`
 *  with its own `properties`, or an `array` with an `items` shape, and either
 *  may carry a `default`. All three are rendered here, recursively, so the form
 *  can express any answer the schema describes rather than flattening it to a
 *  text box the server would refuse.
 *
 *  Props-only and pure in its decisions (`governance.ts` holds the schema
 *  reading), so the markup and the submitted object are both reachable by a test
 *  without a store or a fetch. */

/** A path from the form's root to one control — a property name, or an array
 *  index, in order. */
type Path = (string | number)[];

/** Read the value at `path`, or undefined when nothing is there yet. */
function getIn(root: QuestionInput | undefined, path: Path): QuestionInput | undefined {
  let cur = root;
  for (const seg of path) {
    if (cur === null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, QuestionInput>)[seg];
  }
  return cur;
}

/** A copy of `root` with `path` set to `value`, creating containers as needed.
 *  Immutable, so React sees a new object and re-renders. */
function setIn(root: QuestionInput | undefined, path: Path,
               value: QuestionInput): QuestionInput {
  if (path.length === 0) return value;
  const [head, ...rest] = path;
  if (typeof head === "number") {
    const arr = Array.isArray(root) ? root.slice() : [];
    while (arr.length <= head) arr.push("");
    arr[head] = setIn(arr[head], rest, value);
    return arr;
  }
  const obj = (root !== null && typeof root === "object" && !Array.isArray(root)
    ? { ...(root as QuestionValues) } : {} as QuestionValues);
  obj[head] = setIn(obj[head], rest, value);
  return obj;
}

const INPUT_CLS = "flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px] "
  + "outline-none";
const SMALL_BTN_CLS = "rounded bg-surface px-2 py-0.5 font-mono text-[10.5px] "
  + "text-secondary hover:bg-panel";

function requiredMark(f: QuestionField) {
  return f.required ? <span className="text-amber"> *</span> : null;
}

/** One control, and for `object`/`array` the controls it contains. */
function FieldControl({ field, path, values, set, labelOverride }: {
  field: QuestionField;
  path: Path;
  values: QuestionValues;
  set: (path: Path, value: QuestionInput) => void;
  /** The row label for an array element, which has no property name of its own. */
  labelOverride?: string;
}) {
  const label = labelOverride ?? field.label;
  const value = getIn(values, path);

  if (field.kind === "object") {
    return (
      <span className="block border-l border-line pl-2">
        <span className="block text-[12px] text-secondary break-words">
          {field.label}{requiredMark(field)}
        </span>
        {field.description && (
          <span className="block text-[11px] text-muted leading-snug">
            {field.description}
          </span>
        )}
        {(field.fields ?? []).map((sub) => (
          <FieldControl key={sub.name} field={sub} path={[...path, sub.name]}
                        values={values} set={set} />
        ))}
      </span>
    );
  }

  if (field.kind === "array") {
    const rows = Array.isArray(value) ? value : [];
    return (
      <span className="block border-l border-line pl-2">
        <span className="block text-[12px] text-secondary break-words">
          {field.label}{requiredMark(field)}
        </span>
        {rows.map((_row, i) => (
          <span key={String(i)} className="flex items-start gap-2">
            <span className="flex-1 min-w-0">
              {field.items && (
                <FieldControl field={field.items} path={[...path, i]}
                              values={values} set={set}
                              labelOverride={`${field.label} ${i + 1}`} />
              )}
            </span>
            <button
              type="button"
              aria-label={`remove ${field.label} ${i + 1}`}
              onClick={() => set(path, rows.filter((_r, j) => j !== i))}
              className={SMALL_BTN_CLS}
            >
              remove
            </button>
          </span>
        ))}
        <button
          type="button"
          aria-label={`add ${field.label}`}
          onClick={() => set(path, [...rows, ""])}
          className={SMALL_BTN_CLS}
        >
          add {field.label}
        </button>
      </span>
    );
  }

  // Leaves.
  const str = typeof value === "string" ? value : "";
  return (
    <span className="block">
      <label className="flex items-start gap-2 text-[12px]">
        <span className="w-28 shrink-0 text-secondary break-words">
          {label}{requiredMark(field)}
        </span>
        {field.kind === "boolean" ? (
          <input
            type="checkbox"
            aria-label={label}
            checked={value === true}
            onChange={(e) => set(path, e.target.checked)}
            className="mt-0.5 shrink-0 accent-amber"
          />
        ) : field.kind === "choice" ? (
          <select
            aria-label={label}
            value={str}
            onChange={(e) => set(path, e.target.value)}
            className={INPUT_CLS}
          >
            <option value="">(choose)</option>
            {field.choices.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        ) : (
          <input
            type={field.kind === "number" ? "number" : "text"}
            aria-label={label}
            value={str}
            onChange={(e) => set(path, e.target.value)}
            className={INPUT_CLS}
          />
        )}
      </label>
      {field.description && (
        <span className="block pl-28 text-[11px] text-muted leading-snug">
          {field.description}
        </span>
      )}
    </span>
  );
}

export default function QuestionForm({ question, schema, onSubmit }: {
  /** The `data.question` sentence, shown verbatim. */
  question: string;
  /** The `data.schema` elicitation schema; `{}` when the server sent none. */
  schema: Record<string, unknown>;
  /** Send the answer. Called only when every required field is filled. */
  onSubmit: (answer: Record<string, unknown>) => void;
}) {
  const fields = questionFields(schema);
  const [values, setValues] = useState<QuestionValues>(
    () => questionDefaults(fields));
  const problem = questionProblem(fields, values);
  const set = useCallback((path: Path, value: QuestionInput) => {
    setValues((v) => setIn(v, path, value) as QuestionValues);
  }, []);

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
        <FieldControl key={f.name} field={f} path={[f.name]}
                      values={values} set={set} />
      ))}
      <span className="flex items-center gap-2">
        <button
          type="button"
          disabled={problem !== ""}
          onClick={() => onSubmit(questionAnswer(fields, values))}
          className="rounded bg-surface px-2 py-0.5 font-mono text-[10.5px]
                     text-moss hover:bg-panel disabled:opacity-50"
        >
          send answer
        </button>
        <span className="text-muted">
          {problem === "" ? "— mcode is waiting on this" : `— ${problem}`}
        </span>
      </span>
    </span>
  );
}
