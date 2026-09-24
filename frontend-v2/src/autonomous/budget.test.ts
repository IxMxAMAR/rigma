import { describe, expect, it } from "vitest";

import { STEP_CAP, budget, remainingTime, stopSentence } from "./budget";

describe("run budget", () => {
  it("falls back to the server's step cap when the record has no ceiling", () => {
    expect(budget({ iteration: 10, deadline: 0 }, 0).ceiling).toBe(STEP_CAP);
    expect(budget({ iteration: 10 }, 0).remainingSteps).toBe(STEP_CAP - 10);
  });

  it("prefers a ceiling the server did send (a restarted run)", () => {
    const b = budget({ iteration: 10, iter_ceiling: 2100, deadline: 0 }, 0);
    expect(b.ceiling).toBe(2100);
    expect(b.remainingSteps).toBe(2090);
  });

  it("never reports negative remaining budget", () => {
    const b = budget({ iteration: 50, iter_ceiling: 10, deadline: 100 }, 500);
    expect(b.remainingSteps).toBe(0);
    expect(b.remainingSeconds).toBe(0);
  });

  it("computes remaining time from the deadline", () => {
    expect(budget({ deadline: 1000 }, 400).remainingSeconds).toBe(600);
    expect(budget({}, 400).remainingSeconds).toBe(0);
  });

  it("formats a remaining duration", () => {
    expect(remainingTime(0)).toBe("none");
    expect(remainingTime(30)).toBe("under a minute");
    expect(remainingTime(120)).toBe("2m");
    expect(remainingTime(8 * 3600 + 5 * 60)).toBe("8h 5m");
  });
});

describe("stop reason", () => {
  it("prefers the server's own halt reason", () => {
    expect(stopSentence("stopped", "user pressed stop")).toBe("user pressed stop");
  });

  it("distinguishes finished from gave up", () => {
    expect(stopSentence("done")).toMatch(/finished/);
    expect(stopSentence("stalled")).toMatch(/stopped/);
    expect(stopSentence("budget_exhausted")).toMatch(/budget/);
    expect(stopSentence("error")).toMatch(/failed/);
  });

  it("still says something for an unknown status", () => {
    expect(stopSentence("")).toBe("stopped (unknown)");
    expect(stopSentence("weird")).toBe("stopped (weird)");
  });
});
