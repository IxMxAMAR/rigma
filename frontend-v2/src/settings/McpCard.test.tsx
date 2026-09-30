// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import McpCard from "./McpCard";

// D4c. `GET /api/mcp` is a report, and its handler STARTS the configured
// servers (`manager().status()` → `_ensure()`, mcp_client.py:433-444). So the
// first assertion here is the one that matters most: mounting the Settings page
// must not call the route at all.

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

const STATUS = {
  configured: ["fs", "git"],
  running: ["fs", "git"],
  failed: { git: "spawn ENOENT" },
  dead: {},
  wedged: {},
  oversize_frames: {},
  tools: ["read_file", "write_file", "git_status"],
};

describe("the MCP status card", () => {
  let container: HTMLDivElement;
  let root: Root;
  let calls: string[];

  function serve(body: () => Response) {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(String(url));
      return body();
    }));
  }

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    calls = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  const button = (label: string) =>
    [...container.querySelectorAll("button")]
      .find((b) => b.textContent === label);

  it("does not touch the route until asked", async () => {
    serve(() => reply(200, STATUS));
    await act(async () => {
      root.render(<McpCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(calls).toEqual([]);
    expect(container.textContent).toContain("Checking starts them");
  });

  it("reports the servers and the tools they contribute", async () => {
    serve(() => reply(200, STATUS));
    await act(async () => {
      root.render(<McpCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await act(async () => {
      button("check mcp servers")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(calls).toEqual(["/api/mcp"]);
    expect(container.textContent)
      .toContain("2 configured · 2 running · 3 tools · 1 failed");
    expect(container.textContent).toContain("read_file");
    expect(container.textContent).toContain("git failed — spawn ENOENT");
  });

  it("shows the server's own error sentence on a 500", async () => {
    serve(() => reply(500, { error: "mcp config is unreadable" }));
    await act(async () => {
      root.render(<McpCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await act(async () => {
      button("check mcp servers")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(container.textContent).toContain("mcp config is unreadable");
  });

  it("renders the unconfigured hint the route sends", async () => {
    serve(() => reply(200, {
      configured: [], running: [], failed: {}, tools: [],
      hint: "add servers in C:\\Users\\x\\.rigma\\mcp.json",
    }));
    await act(async () => {
      root.render(<McpCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await act(async () => {
      button("check mcp servers")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(container.textContent).toContain("mcp.json");
    expect(container.textContent).toContain("0 configured · 0 running · 0 tools");
  });
});
