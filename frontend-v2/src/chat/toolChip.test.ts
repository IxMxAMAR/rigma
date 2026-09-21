import { describe, expect, it } from "vitest";

import { formatArgs, previewArgs } from "./toolChip";

// A chip is how a reader judges a tool call they did not write. These are the
// calls an EXTERNAL agent makes, with that agent's vocabulary — which is not
// Rigma's, and assuming it was is why `bash` rendered as a bare name.
describe("what identifies a call", () => {
  it("shows the command for a bash call", () => {
    // mcode says `command`; Rigma's own chips say `cmd`. Only one of those
    // was in the list, so the single most important call showed nothing.
    expect(previewArgs({ command: "pytest -q" })).toBe("pytest -q");
  });

  it("shows the path for a read or write", () => {
    expect(previewArgs({ path: "src/rigma/serve.py" })).toBe("src/rigma/serve.py");
    expect(previewArgs({ path: "a.py", content: "x".repeat(300) }))
      .toBe("a.py · 300 chars");
  });

  it("shows the url for a fetch and the query for a search", () => {
    expect(previewArgs({ url: "https://example.com" })).toBe("https://example.com");
    expect(previewArgs({ query: "rigma harness" })).toBe("rigma harness");
  });

  it("shows the objective for a goal call and the name for a skill", () => {
    expect(previewArgs({ objective: "ship the arm" })).toBe("ship the arm");
    expect(previewArgs({ name: "runpod" })).toBe("runpod");
  });

  it("counts a list of objects instead of printing [object Object]", () => {
    // `todowrite` takes a list of todo objects. Joining them is worse than
    // useless — it looks like a real value.
    expect(previewArgs({ todos: [{ a: 1 }, { a: 2 }] })).toBe("2 items");
    expect(previewArgs({ paths: ["a.py", "b.py"] })).toBe("a.py, b.py");
  });

  it("falls back to the argument names when nothing identifies the call", () => {
    expect(previewArgs({ alpha: 1, beta: 2 })).toBe("alpha, beta");
  });

  it("says nothing for no arguments rather than something misleading", () => {
    expect(previewArgs(null)).toBe("");
    expect(previewArgs(undefined)).toBe("");
    expect(previewArgs("not an object")).toBe("");
  });
});

describe("what a chip shows when it is opened", () => {
  it("gives the command a full line, because truncation is the whole problem", () => {
    // 56 chars is right for a one-line summary and wrong for the thing a
    // reader is deciding whether to trust.
    const cmd = "python -m pytest tests/test_harness_mcode.py -k permission -q";
    expect(previewArgs({ command: cmd }).endsWith("…")).toBe(true);
    expect(formatArgs({ command: cmd })).toBe(cmd);
  });

  it("keeps the remaining arguments, so nothing is hidden", () => {
    expect(formatArgs({ command: "ls", cwd: "/tmp" }))
      .toBe('ls\n{\n  "cwd": "/tmp"\n}');
  });

  it("does not repeat the identifying argument in the JSON tail", () => {
    expect(formatArgs({ path: "a.py" })).toBe("a.py");
  });

  it("falls back to plain JSON when no argument identifies the call", () => {
    expect(formatArgs({ alpha: 1 })).toBe('{\n  "alpha": 1\n}');
  });

  it("survives a value that cannot be serialised", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => formatArgs(circular)).not.toThrow();
  });

  it("says nothing rather than 'null' when there are no arguments", () => {
    expect(formatArgs(null)).toBe("");
    expect(formatArgs(undefined)).toBe("");
  });
});

describe("a subagent call is identified by its agent, not its prose", () => {
  // Measured off the wire: mcode's `task` requires description, prompt AND
  // agent_name. Showing the description told the reader what the subagent was
  // asked to do without telling them WHICH one was asked.
  const call = {
    agent_name: "explore",
    description: "find every place the seam is called",
    prompt: "long instructions that do not belong on one line",
    run_in_background: false,
  };

  it("previews the agent name", () => {
    expect(previewArgs(call)).toBe("explore");
  });

  it("keeps the description on the opened chip", () => {
    const out = formatArgs(call);
    expect(out.split("\n")[0]).toBe("explore");
    expect(out).toContain("find every place the seam is called");
  });

  it("still falls back to description when there is no agent", () => {
    // A tool with a description and no agent_name must not regress.
    expect(previewArgs({ description: "do the thing" })).toBe("do the thing");
  });
});
