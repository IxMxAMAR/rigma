// Reading a list endpoint without trusting it.
//
// Every surface cast the body straight to an array. `as Skill[]` is erased at
// runtime, so a FastAPI 500's `{"detail": ...}` (or a 404's error object)
// became the state, `skills.length === 0` was false (undefined !== 0) and
// `skills.map` threw during render — and with no error boundary React unmounted
// the root, so the whole app went blank. MemorySurface had this guard; this is
// that guard, once (AUDIT F11-3).

/** The server's own sentence for a bad body. Rigma's routes answer `{error}`;
 *  FastAPI's own HTTPException answers `{detail}`. Both are read. */
export function bodyError(d: unknown, r: Response): string {
  if (d && typeof d === "object") {
    const o = d as { error?: unknown; detail?: unknown };
    if (typeof o.error === "string" && o.error) return o.error;
    if (typeof o.detail === "string" && o.detail) return o.detail;
  }
  return `server replied ${r.status}`;
}

export type ListResult<T> =
  | { ok: true; rows: T[] }
  | { ok: false; error: string };

/** Read a bare-array list endpoint (`/api/skills`, `/api/presets`). */
export async function readList<T>(r: Response): Promise<ListResult<T>> {
  const d: unknown = await r.json().catch(() => null);
  if (!r.ok || !Array.isArray(d)) return { ok: false, error: bodyError(d, r) };
  return { ok: true, rows: d as T[] };
}

/** Read `{ <field>: T[] }`, e.g. `/api/methods` → `{ methods: [...] }`. */
export async function readListField<T>(
  r: Response, field: string,
): Promise<ListResult<T>> {
  const d: unknown = await r.json().catch(() => null);
  const v = d && typeof d === "object"
    ? (d as Record<string, unknown>)[field] : undefined;
  if (!r.ok || !Array.isArray(v)) return { ok: false, error: bodyError(d, r) };
  return { ok: true, rows: v as T[] };
}

/** The server's sentence for a failed response, reading its body. */
export async function responseError(r: Response): Promise<string> {
  return bodyError(await r.json().catch(() => null), r);
}
