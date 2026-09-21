import { describe, expect, it } from "vitest";

import { SANITIZE_CONFIG, sanitizeHtml } from "./sanitizeConfig";

// AUDIT F58. Model output is untrusted HTML by definition — it arrives from
// RAG-indexed documents, MCP results and files the model read — and marked
// passes raw HTML straight through. DOMPurify WAS applied, but with its stock
// config, whose allowlist includes <style> and <form>.
//
// WHAT THIS DOES NOT DO: run the real sanitiser. This environment has no DOM
// (no jsdom, no happy-dom), so `DOMPurify.sanitize` cannot execute here. The
// boundary and its APPLICATION are pinned; that DOMPurify honours FORBID_TAGS
// is its own documented contract, and a browser check is still owed.

describe("the sanitizer boundary (AUDIT F58)", () => {
  it("forbids the elements whose reach escapes the transcript", () => {
    // <style> is document-GLOBAL: one arriving in model output restyles the
    // whole app, and background:url(https://…) is an outbound request on
    // render. The form elements are the phishing shape.
    for (const tag of ["style", "form", "input", "button", "textarea",
                       "select", "iframe", "object", "embed"]) {
      expect(SANITIZE_CONFIG.FORBID_TAGS).toContain(tag);
    }
  });

  it("forbids the attributes that carry the same reach", () => {
    // `style` is the inline equivalent of the <style> element; `action` and
    // `formaction` are where a form posts; `srcset` is a second image URL.
    for (const attr of ["style", "action", "formaction", "srcset"]) {
      expect(SANITIZE_CONFIG.FORBID_ATTR).toContain(attr);
    }
  });

  it("applies the boundary rather than the stock config", () => {
    // The original bug was a sanitiser called with NO config. Defining the
    // config is not the fix; passing it is. A spy stands in for DOMPurify
    // because there is no DOM here — which is the reason this seam exists.
    const seen: (unknown)[] = [];
    const spy = (html: string, cfg?: unknown) => {
      seen.push(cfg);
      return html;
    };

    const out = sanitizeHtml("<style>body{display:none}</style>hi", spy);

    expect(out).toBe("<style>body{display:none}</style>hi");
    expect(seen).toHaveLength(1);
    expect(seen[0]).toBe(SANITIZE_CONFIG);
    expect(seen[0]).not.toBeUndefined();
  });
});
