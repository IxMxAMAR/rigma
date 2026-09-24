import { describe, expect, it } from "vitest";

import { DEFAULT_PROFILE, PROFILES, profileSentence } from "./profiles";

describe("run profiles", () => {
  it("defaults to the server's own default", () => {
    expect(DEFAULT_PROFILE).toBe("all");
  });

  it("names exactly the four profiles runs.PROFILES accepts", () => {
    expect(PROFILES.map((p) => p.value)).toEqual(
      ["all", "no-network", "no-delete", "confined"]);
  });

  it("has a sentence for every profile", () => {
    for (const p of PROFILES) {
      expect(p.consequence.length).toBeGreaterThan(20);
      expect(profileSentence(p.value)).toBe(p.consequence);
    }
  });

  it("says nothing rather than guessing for an unknown profile", () => {
    expect(profileSentence("some-future-profile")).toBe("");
    expect(profileSentence("")).toBe("");
  });

  it("states the restriction each profile actually applies", () => {
    expect(profileSentence("no-network")).toMatch(/web_search/);
    expect(profileSentence("confined")).toMatch(/run_shell/);
    expect(profileSentence("no-delete")).toMatch(/[Dd]eletion/);
  });

  // R3-4. These sentences used to promise that code execution "still works" for
  // every profile that does not withhold it. That was false for a RUN: the run
  // needs `confirm_exec`, it defaulted off, and no surface could turn it on — so
  // the launcher described a capability the run did not have. A profile NARROWS
  // a roster; only the separate switch GRANTS execution.
  it("never claims a profile grants code execution", () => {
    for (const p of PROFILES) {
      expect(p.consequence).not.toMatch(/code execution[^.]*still works/i);
      expect(p.consequence).not.toMatch(/^Code execution, /);
    }
  });

  it("points at the separate execution switch rather than implying it", () => {
    // `all` is the one that used to make the claim, so it must now say where
    // execution actually comes from; and `confined` must say the profile wins.
    expect(profileSentence("all")).toMatch(/separate switch/i);
    expect(profileSentence("confined")).toMatch(/regardless/i);
  });
});
