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
});
