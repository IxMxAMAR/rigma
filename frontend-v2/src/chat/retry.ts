// DSH's LLM retry, made visible.
//
// WHY THIS EXISTS. `dsh-llm-retry` is a dependency of `sdk-minimal` itself, so it
// is always loaded and always firing. When a request to the model fails and DSH
// schedules another attempt, it writes `llm/retry` and then `llm/retry-started`
// and says NOTHING on any surface Rigma reads. With a local llama-server that
// stalls, OOMs or returns a malformed tool call, the visible effect is a turn that
// sits there producing nothing — which is the same "frozen turn" this round has
// been fixing, and the one place a user most needs to be told what is happening.
//
// Both events are `Durable` in DSH's words — non-surface records, never part of the
// model transcript — which is exactly the shape of fact a UI should show and the
// conversation should not carry.
//
// Shapes quoted from DSH's own declarations
// (packages/llm/llm-retry/src/types.ts:16-42, and `LlmFailure` at
// packages/llm/llm/src/types.ts:40):
//
//   llm/retry          {retryId, turn, step, provider, mode, policyKey, retry,
//                       delayMs, failure: {message, code, status?,
//                       providerRetryAfterMs?, requestId?}, maxRetries?}
//   llm/retry-started  {retryId, turn, step, retry}
//
// TWO MODES, and the difference is not cosmetic: `normal` is bounded by
// `maxRetries`, so "attempt 3 of 5" is true; `always` is UNBOUNDED, and saying
// "of 5" there would be a lie. The `always` variant has no `maxRetries` field at
// all, which is why this is modelled rather than read as a number.

export interface RetryFailure {
  /** Human-readable provider or transport failure. */
  message: string;
  /** Stable provider-neutral machine-routing code, e.g. `TRANSPORT_ERROR`. */
  code: string;
  /** HTTP status, when the provider gave one. */
  status: number | null;
}

export interface Retry {
  retryId: string;
  /** Which attempt this is, 1-based as DSH numbers it. */
  attempt: number;
  /** The cap, or null when the policy is unbounded (`mode: "always"`). */
  maxRetries: number | null;
  /** How long DSH waited before the next attempt. */
  delayMs: number;
  provider: string;
  failure: RetryFailure;
  /** False once `llm/retry-started` says the wait is over and the attempt began.
   *  Kept rather than cleared so the line can say "retried" instead of vanishing
   *  the moment it becomes true. */
  waiting: boolean;
}

export const EMPTY_RETRY: Retry | null = null;

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function numOrNull(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

/** Fold one `llm/retry*` payload in. Returns `prev` unchanged for anything else,
 *  so the caller's reducer arm can be unconditional. */
export function foldRetry(prev: Retry | null, payload: unknown): Retry | null {
  if (!payload || typeof payload !== "object") return prev;
  const p = payload as Record<string, unknown>;
  const raw = str(p.event);
  // The prefix is REQUIRED, not merely stripped — stripping a prefix that is not
  // there is a no-op, so `{event: "goal"}` would become a retry. Same bug already
  // fixed twice in this codebase (governance.ts, compaction.ts).
  if (!raw.startsWith("llm/retry")) return prev;
  const d = (p.data && typeof p.data === "object" ? p.data : {}) as Record<string, unknown>;

  if (raw === "llm/retry-started") {
    // The wait is over. Only meaningful for the retry it names; a started event
    // for a retry we never saw scheduled is ignored rather than invented.
    const id = str(d.retryId);
    if (prev === null || (id && prev.retryId !== id)) return prev;
    return { ...prev, waiting: false };
  }

  const f = (d.failure && typeof d.failure === "object" ? d.failure : {}) as Record<string, unknown>;
  return {
    retryId: str(d.retryId),
    attempt: numOrNull(d.retry) ?? 1,
    // Absent for `mode: "always"`, and null is the honest value: an unbounded
    // policy has no cap to report.
    maxRetries: numOrNull(d.maxRetries),
    delayMs: numOrNull(d.delayMs) ?? 0,
    provider: str(d.provider),
    failure: {
      message: str(f.message),
      code: str(f.code),
      status: numOrNull(f.status),
    },
    waiting: true,
  };
}

/** One line describing the retry, or null when there is nothing to say. Built
 *  from the numbers so it cannot claim more than it knows. */
export function retryLine(r: Retry): string | null {
  const what = r.failure.message || r.failure.code;
  if (!what) return null;
  const bits: string[] = [];
  if (r.waiting && r.delayMs > 0) {
    bits.push(`retrying in ${formatDelay(r.delayMs)}`);
  } else if (r.waiting) {
    bits.push("retrying");
  } else {
    bits.push("retried");
  }
  // Only claim a bound when there IS one.
  bits.push(r.maxRetries === null
    ? `attempt ${String(r.attempt)}, no limit`
    : `attempt ${String(r.attempt)} of ${String(r.maxRetries)}`);
  return `${what} — ${bits.join(" · ")}`;
}

/** Milliseconds as something a person reads at a glance. */
export function formatDelay(ms: number): string {
  if (ms < 1000) return `${String(ms)}ms`;
  const s = ms / 1000;
  // One decimal only below 10s: "1.5s" is useful, "137.4s" is noise.
  return s < 10 ? `${s.toFixed(1)}s` : `${String(Math.round(s))}s`;
}
