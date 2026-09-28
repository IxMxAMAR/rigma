import { describe, expect, it } from "vitest";

import {
  COMMANDS,
  commandQuery,
  helpText,
  isPermissionMode,
  matching,
  parseSlash,
  planFor,
} from "./commands";

// DSH's own commands cannot be reached from Rigma whatever it types: the only
// caller of `parseCommand` is a `@Remote` web-layer method, and the SDK stdio
// wire has exactly initialize / session/prompt / shutdown. So this surface is
// Rigma's own, and every command must be backed by something Rigma can do.
describe("parseSlash", () => {
  it("parses a known command and its args", () => {
    expect(parseSlash("/compact")).toEqual({ name: "compact", args: "" });
    expect(parseSlash("/permission smart")).toEqual({ name: "permission", args: "smart" });
  });

  it("trims the args", () => {
    expect(parseSlash("/permission   smart  ")?.args).toBe("smart");
  });

  // THE critical property. Swallowing a legitimate prompt is the worst thing a
  // command box can do: it is indistinguishable from the app being broken.
  it("does NOT treat a path as a command", () => {
    expect(parseSlash("/usr/bin/python is slow")).toBeNull();
    expect(parseSlash("/etc/hosts")).toBeNull();
  });

  it("does not treat prose starting with a known name as a command", () => {
    // The lookahead requires whitespace or end-of-line after the name, so this
    // is a sentence about compaction, not an invocation of it.
    expect(parseSlash("/compaction is slow")).toBeNull();
    expect(parseSlash("/newer approaches exist")).toBeNull();
  });

  it("refuses an unknown command rather than passing it through", () => {
    // Sending it as prose would just ask the model to discuss the command.
    expect(parseSlash("/nope")).toBeNull();
    expect(parseSlash("/goal ship it")).toBeNull();
  });

  it("is case-sensitive, like DSH's own parser", () => {
    expect(parseSlash("/Compact")).toBeNull();
  });

  it("ignores a line that is not a command at all", () => {
    expect(parseSlash("hello")).toBeNull();
    expect(parseSlash("")).toBeNull();
    expect(parseSlash(" /compact")).toBeNull();
  });

  it("accepts a tab or newline after the name", () => {
    expect(parseSlash("/compact\tnow")?.name).toBe("compact");
  });
});

describe("commandQuery", () => {
  it("is the name while the user is typing it, and the bare slash", () => {
    expect(commandQuery("/com")).toBe("com");
    expect(commandQuery("/")).toBe("");
  });

  // The menu must survive the moment the name is exactly complete: that is when
  // the user is looking at it, one keystroke from Enter. It closes on the SPACE,
  // which is when they have committed and moved on to the argument.
  it("still matches a complete name, and closes on the space", () => {
    expect(commandQuery("/compact")).toBe("compact");
    expect(commandQuery("/compact now")).toBeNull();
    expect(commandQuery("plain text")).toBeNull();
  });
});

describe("matching", () => {
  it("matches by prefix, in roster order", () => {
    expect(matching("c").map((c) => c.name)).toEqual(["compact"]);
    expect(matching("").map((c) => c.name)).toEqual(COMMANDS.map((c) => c.name));
  });

  it("is case-insensitive", () => {
    expect(matching("COM").map((c) => c.name)).toEqual(["compact"]);
  });

  it("returns nothing for a prefix no command has", () => {
    expect(matching("zzz")).toEqual([]);
  });
});

describe("planFor", () => {
  it("runs a command that needs no args", () => {
    expect(planFor("compact", "")).toEqual({ kind: "run", name: "compact", args: "" });
  });

  it("runs a command that has its args", () => {
    expect(planFor("permission", "smart")).toEqual({
      kind: "run", name: "permission", args: "smart",
    });
  });

  // Refusing with the acceptable values named, because "usage: /permission"
  // leaves the user guessing what to type next.
  it("refuses a command whose required argument is missing, and says which", () => {
    const plan = planFor("permission", "");
    expect(plan.kind).toBe("usage");
    if (plan.kind === "usage") {
      expect(plan.message).toContain("off");
      expect(plan.message).toContain("smart");
      expect(plan.message).toContain("full");
    }
  });

  it("refuses a command that does not exist", () => {
    const plan = planFor("nope", "");
    expect(plan.kind).toBe("usage");
    if (plan.kind === "usage") expect(plan.message).toContain("no such command");
  });
});

describe("helpText", () => {
  it("lists every command, built from the roster so it cannot drift", () => {
    const help = helpText();
    for (const c of COMMANDS) expect(help).toContain(`/${c.name}`);
  });

  it("shows the argument hint for a command that needs one", () => {
    expect(helpText()).toContain("/permission <off | smart | full>");
  });
});

describe("isPermissionMode", () => {
  it("accepts the modes the server accepts", () => {
    expect(isPermissionMode("off")).toBe(true);
    expect(isPermissionMode("smart")).toBe(true);
    expect(isPermissionMode("full")).toBe(true);
  });

  it("refuses anything else, before a request is made", () => {
    expect(isPermissionMode("")).toBe(false);
    expect(isPermissionMode("SMART")).toBe(false);
    expect(isPermissionMode("yolo")).toBe(false);
  });
});

describe("the roster itself", () => {
  it("gives every command a summary, because the menu shows one", () => {
    for (const c of COMMANDS) {
      expect(c.summary.length, c.name).toBeGreaterThan(0);
    }
  });

  it("gives every argument-taking command a hint", () => {
    for (const c of COMMANDS) {
      if (c.takesArgs) expect(c.argHint, c.name).toBeTruthy();
    }
  });

  it("has no duplicate names", () => {
    const names = COMMANDS.map((c) => c.name);
    expect(new Set(names).size).toBe(names.length);
  });
});

// R6-EXPORT: `/export` wraps a route Rigma has had all along.
//
// The invariant this file states at the top is that "every command must be backed by
// something Rigma can do" — and this one is, which is why it was worth adding: the
// capability existed and only the composer could not reach it.
describe("R6-EXPORT: /export", () => {
  it("parses, and is in the roster", () => {
    expect(parseSlash("/export")).toEqual({ name: "export", args: "" });
    expect(COMMANDS.some((c) => c.name === "export")).toBe(true);
  });

  it("takes no args, because there is one sensible format", () => {
    // The route also serves `fmt=json`, but that is reachable by URL. A command that
    // took a format argument would be a second way to say what the rail already says.
    const cmd = COMMANDS.find((c) => c.name === "export");
    expect(cmd?.takesArgs).toBe(false);
  });

  it("is offered by /help, so it is discoverable", () => {
    // A command nobody can find is not a feature. `/help` is generated from the
    // roster, so this checks the roster rather than a hand-written list.
    expect(helpText()).toContain("/export");
  });

  it("does NOT swallow a path that merely starts with the same letters", () => {
    // The parser's strictness is the feature: `/exports/2026/report.md` is a path, and
    // eating it would be indistinguishable from the app being broken.
    expect(parseSlash("/exports/2026/report.md")).toBeNull();
  });
});
