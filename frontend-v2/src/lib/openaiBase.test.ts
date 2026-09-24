import { describe, expect, it } from "vitest";

import { openaiBase } from "./openaiBase";

// AUDIT F11-6: Settings advertised the engine's upstream port (11499) as
// "Rigma's OpenAI API"; the server's own public port is 11500 and it publishes
// it as openai_base. The fallback must be the page's origin, never a literal
// port — a hardcoded one is wrong on any non-default launch.
describe("the OpenAI base URL", () => {
  it("uses the server's published base", () => {
    expect(openaiBase({ openai_base: "http://127.0.0.1:11500/v1" },
                      "http://127.0.0.1:11500"))
      .toBe("http://127.0.0.1:11500/v1");
  });

  it("trims a padded value", () => {
    expect(openaiBase({ openai_base: "  http://127.0.0.1:8080/v1  " },
                      "http://127.0.0.1:8080"))
      .toBe("http://127.0.0.1:8080/v1");
  });

  it("falls back to the origin that served the page", () => {
    expect(openaiBase(null, "http://127.0.0.1:9000"))
      .toBe("http://127.0.0.1:9000/v1");
    expect(openaiBase({}, "http://localhost:9000/"))
      .toBe("http://localhost:9000/v1");
  });

  it("ignores a non-string or empty openai_base", () => {
    expect(openaiBase({ openai_base: 11500 }, "http://127.0.0.1:9000"))
      .toBe("http://127.0.0.1:9000/v1");
    expect(openaiBase({ openai_base: "   " }, "http://127.0.0.1:9000"))
      .toBe("http://127.0.0.1:9000/v1");
  });

  it("never invents a port when the info is missing", () => {
    expect(openaiBase(undefined, "http://127.0.0.1:11500"))
      .not.toContain("11499");
  });
});
