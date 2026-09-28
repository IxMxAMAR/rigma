import { describe, expect, it } from "vitest";

import { EMPTY_RETRY, foldRetry, formatDelay, retryLine, type Retry } from "./retry";

const scheduled = (over: Record<string, unknown> = {}) => ({
  event: "llm/retry",
  data: {
    retryId: "r1", turn: 1, step: 1, provider: "deepseek-official",
    mode: "normal", policyKey: "default", retry: 2, maxRetries: 5, delayMs: 1500,
    failure: { message: "connection reset", code: "TRANSPORT_ERROR" },
    ...over,
  },
});
const started = (retryId = "r1") => ({
  event: "llm/retry-started",
  data: { retryId, turn: 1, step: 1, retry: 2 },
});

// `dsh-llm-retry` is a dependency of sdk-minimal itself, so it is always loaded
// and always firing. With a local llama-server that stalls or returns a malformed
// tool call, the visible effect was a turn that sat there producing nothing.
describe("foldRetry", () => {
  it("records a scheduled retry with its attempt count and failure", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled());
    expect(r).toMatchObject({
      retryId: "r1", attempt: 2, maxRetries: 5, delayMs: 1500,
      provider: "deepseek-official", waiting: true,
    });
    expect(r?.failure).toEqual({
      message: "connection reset", code: "TRANSPORT_ERROR", status: null,
    });
  });

  // `mode: "always"` has NO maxRetries field. Reading it as a number would make
  // the UI claim "attempt 2 of 0" or "of null" for an unbounded policy.
  it("leaves the cap null for an unbounded policy", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({ mode: "always", maxRetries: undefined }));
    expect(r?.maxRetries).toBeNull();
  });

  it("keeps the HTTP status when the provider gave one", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({
      failure: { message: "rate limited", code: "HTTP_ERROR", status: 429 },
    }));
    expect(r?.failure.status).toBe(429);
  });

  it("marks the wait over on llm/retry-started, without erasing what happened", () => {
    let r = foldRetry(EMPTY_RETRY, scheduled());
    r = foldRetry(r, started());
    expect(r?.waiting).toBe(false);
    // The failure is still there: the line should say "retried", not vanish the
    // moment it becomes true.
    expect(r?.failure.code).toBe("TRANSPORT_ERROR");
    expect(r?.attempt).toBe(2);
  });

  it("ignores a retry-started for a retry it never saw scheduled", () => {
    const r = foldRetry(EMPTY_RETRY, started("never-seen"));
    expect(r).toBeNull();
  });

  it("ignores a retry-started naming a different retry", () => {
    const r = foldRetry(foldRetry(EMPTY_RETRY, scheduled()), started("other"));
    expect(r?.waiting).toBe(true);
  });

  // Same bug already fixed twice: stripping a prefix that is not there is a no-op.
  it("refuses a payload that is not a retry event", () => {
    expect(foldRetry(EMPTY_RETRY, { event: "goal", data: {} })).toBe(EMPTY_RETRY);
    expect(foldRetry(EMPTY_RETRY, { event: "compaction/start", data: {} })).toBe(EMPTY_RETRY);
    expect(foldRetry(EMPTY_RETRY, {})).toBe(EMPTY_RETRY);
    expect(foldRetry(EMPTY_RETRY, null)).toBe(EMPTY_RETRY);
  });

  it("survives a malformed failure object", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({ failure: "not-an-object" }));
    expect(r?.failure).toEqual({ message: "", code: "", status: null });
  });
});

describe("formatDelay", () => {
  it("uses milliseconds below a second", () => {
    expect(formatDelay(250)).toBe("250ms");
  });

  it("uses one decimal below ten seconds", () => {
    expect(formatDelay(1500)).toBe("1.5s");
  });

  it("rounds above ten seconds, where a decimal is noise", () => {
    expect(formatDelay(137400)).toBe("137s");
  });
});

describe("retryLine", () => {
  it("says what failed, how long, and which attempt", () => {
    const line = retryLine(foldRetry(EMPTY_RETRY, scheduled()) as Retry);
    expect(line).toContain("connection reset");
    expect(line).toContain("1.5s");
    expect(line).toContain("attempt 2 of 5");
  });

  // The honest wording for an unbounded policy.
  it("says there is no limit rather than inventing one", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({ mode: "always", maxRetries: undefined }));
    expect(retryLine(r as Retry)).toContain("no limit");
  });

  it("switches to the past tense once the wait is over", () => {
    let r = foldRetry(EMPTY_RETRY, scheduled());
    r = foldRetry(r, started());
    const line = retryLine(r as Retry);
    expect(line).toContain("retried");
    expect(line).not.toContain("retrying in");
  });

  it("falls back to the code when there is no message", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({
      failure: { message: "", code: "TRANSPORT_ERROR" },
    }));
    expect(retryLine(r as Retry)).toContain("TRANSPORT_ERROR");
  });

  it("says nothing when it knows neither a message nor a code", () => {
    const r = foldRetry(EMPTY_RETRY, scheduled({ failure: {} }));
    expect(retryLine(r as Retry)).toBeNull();
  });
});
