import { afterEach, describe, expect, it, vi } from "vitest";

import { parseBackup, postRestore, restoreSummary } from "./backup";

// D4c. `/api/restore` replaces the whole store, so the two facts this module
// must get right are: what the file actually contains (so the confirmation can
// name it), and the server's own sentence when it refuses.

const DOC = {
  rigma_backup: 1,
  app_version: "0.1.0",
  created_at: "2026-09-30T00:00:00Z",
  settings: { idle_unload_minutes: 15 },
  methods: [{ id: "a" }, { id: "b" }],
  memory: [{ id: "1" }, { id: "2" }, { id: "3" }],
};

describe("parseBackup", () => {
  it("accepts a real backup and names what it carries", () => {
    const r = parseBackup(JSON.stringify(DOC));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(restoreSummary(r.doc))
      .toBe("1 settings key, 2 methods, 3 memory rows");
  });

  it("refuses a file that is not a backup, with a reason", () => {
    expect(parseBackup("not json")).toEqual({
      ok: false, error: "that file is not JSON",
    });
    expect(parseBackup("[1,2]")).toEqual({
      ok: false,
      error: "that file is not a Rigma backup (expected a JSON object)",
    });
    expect(parseBackup('{"settings":{}}')).toEqual({
      ok: false,
      error: "that file is not a Rigma backup — it carries no "
        + "`rigma_backup` version",
    });
  });

  it("refuses the wrong-typed halves before the server has to", () => {
    expect(parseBackup(JSON.stringify({ rigma_backup: 1, settings: [] })))
      .toEqual({ ok: false, error: "settings: must be an object" });
    expect(parseBackup(JSON.stringify({ rigma_backup: 1, methods: {} })))
      .toEqual({ ok: false, error: "methods: must be a list" });
    expect(parseBackup(JSON.stringify({ rigma_backup: 1, memory: "x" })))
      .toEqual({ ok: false, error: "memory: must be a list" });
  });

  it("reads singular counts as singular", () => {
    expect(restoreSummary({ rigma_backup: 1, settings: { a: 1 },
                            methods: [{ id: "a" }], memory: [{ id: "1" }] }))
      .toBe("1 settings key, 1 method, 1 memory row");
  });
});

describe("postRestore", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the document and reports what was replaced", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return new Response(JSON.stringify({
        restored: true, version: 1, methods: 2,
        memory: { before: 3, after: 3 },
      }), { status: 200, headers: { "content-type": "application/json" } });
    }));

    const r = await postRestore(DOC);
    expect(seen[0].url).toBe("/api/restore");
    expect(seen[0].init.method).toBe("POST");
    expect(JSON.parse(String(seen[0].init.body))).toEqual(DOC);
    expect(r).toEqual({ ok: true, methods: 2, memory: { before: 3, after: 3 } });
  });

  it("keeps the server's refusal sentence verbatim", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      error: "unknown backup version 9; this build restores version 1",
    }), { status: 400, headers: { "content-type": "application/json" } })));

    const r = await postRestore({ rigma_backup: 9 });
    expect(r).toEqual({
      ok: false,
      error: "unknown backup version 9; this build restores version 1",
    });
  });

  it("falls back to the status when a refusal carries no sentence", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 500 })));
    expect(await postRestore(DOC)).toEqual({
      ok: false, error: "server replied 500",
    });
  });
});
