import { describe, expect, it } from "vitest";

import { bodyError, readList, readListField, responseError } from "./listFetch";

// AUDIT F11-3: a non-array body (a FastAPI 500's {detail}, a 404's error
// object) used to reach `.map` during render and blank the whole app. These
// pin the guard: a failed status OR a non-array body is an error, never rows.
function res(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

describe("reading a list endpoint", () => {
  it("returns the rows for a real array", async () => {
    expect(await readList(res(200, [{ id: "a" }])))
      .toEqual({ ok: true, rows: [{ id: "a" }] });
  });

  it("refuses a 500 whose body is an error object", async () => {
    const r = await readList(res(500, { detail: "Internal Server Error" }));
    expect(r).toEqual({ ok: false, error: "Internal Server Error" });
  });

  it("refuses an ok response whose body is not an array", async () => {
    const r = await readList(res(200, { error: "boom" }));
    expect(r).toEqual({ ok: false, error: "boom" });
  });

  it("refuses a body that is not JSON at all", async () => {
    const bad = {
      ok: false, status: 502,
      json: async () => { throw new Error("not json"); },
    } as unknown as Response;
    expect(await readList(bad)).toEqual({ ok: false, error: "server replied 502" });
  });

  it("reads an array out of a named field", async () => {
    expect(await readListField(res(200, { methods: [{ id: "m" }] }), "methods"))
      .toEqual({ ok: true, rows: [{ id: "m" }] });
    const r = await readListField(res(200, { methods: null }), "methods");
    expect(r.ok).toBe(false);
  });

  it("names the failure from either error or detail", async () => {
    expect(await responseError(res(400, { error: "bad" }))).toBe("bad");
    expect(await responseError(res(400, { detail: "bad" }))).toBe("bad");
    expect(await responseError(res(400, "plain text"))).toBe("server replied 400");
    expect(bodyError(null, res(503, null))).toBe("server replied 503");
  });

  it("names a refused action's reason, not just its status", async () => {
    // AUDIT F11-4: pause/stop/resume/restart and the per-quant actions render
    // this sentence; a bare status would still be better than the silence they
    // replaced, but the server's own words are what tells the user what to do.
    expect(await responseError(res(409, { error: "run is not restartable" })))
      .toBe("run is not restartable");
  });
});
