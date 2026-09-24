import { describe, expect, it } from "vitest";

import { GRANTS, NO_GRANTS, readGrants } from "./grants";

describe("chat grants", () => {
  it("defaults every grant OFF", () => {
    for (const g of GRANTS) expect(NO_GRANTS[g.key]).toBe(false);
    expect(Object.keys(NO_GRANTS).sort()).toEqual(
      GRANTS.map((g) => g.key).sort());
  });

  it("reads a missing or empty session as all OFF", () => {
    expect(readGrants(undefined)).toEqual(NO_GRANTS);
    expect(readGrants(null)).toEqual(NO_GRANTS);
    expect(readGrants({})).toEqual(NO_GRANTS);
    expect(readGrants({ title: "chapter one" })).toEqual(NO_GRANTS);
  });

  it("only `true` grants — a truthy non-boolean does not", () => {
    expect(readGrants({ confirm_exec: "yes" })).toEqual(NO_GRANTS);
    expect(readGrants({ confirm_exec: 1 })).toEqual(NO_GRANTS);
    expect(readGrants({ confirm_exec: null })).toEqual(NO_GRANTS);
  });

  it("reads each grant independently of the others", () => {
    expect(readGrants({ confirm_exec: true })).toEqual({
      ...NO_GRANTS, confirm_exec: true });
    expect(readGrants({ allow_absolute_reads: true })).toEqual({
      ...NO_GRANTS, allow_absolute_reads: true });
    expect(readGrants({ allow_outbound_post: true })).toEqual({
      ...NO_GRANTS, allow_outbound_post: true });
  });
});
