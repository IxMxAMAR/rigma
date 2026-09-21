import type { Config } from "dompurify";

/** What the sanitiser may leave behind (AUDIT F58).
 *
 *  Kept in its own module, free of DOM and CSS imports, so a test can load it:
 *  `Markdown.tsx` pulls in highlight.js stylesheets and touches `document`, and
 *  neither belongs in a security-boundary test.
 *
 *  DOMPurify's stock config already strips the things people expect it to —
 *  `<script>`, event handlers, `javascript:` URLs — and, MEASURED rather than
 *  assumed, the `<style>`, `<iframe>`, `<object>` and `<embed>` ELEMENTS too. The
 *  audit that prompted this file said the stock allowlist "includes `style`"; it
 *  does not, and neither does it include the other three. That part of the finding
 *  was wrong, and it was the part the finding led with. Running the real sanitiser
 *  under jsdom is what showed it (dompurify 3.4.12).
 *
 *  What stock really does allow, same measurement:
 *
 *    `style` attribute + `background:url(…)`   KEPT — outbound request on render
 *    `style` attribute + `position:fixed`      KEPT — full-viewport overlay
 *    `-moz-binding:url(…)`                     KEPT — outbound (Firefox-only, and
 *                                                     Firefox has since removed it)
 *    `srcset` attribute                        KEPT — a second image URL
 *    `<form action=…>` and `<input>`           KEPT — the phishing shape
 *    `<button>`, `<textarea>`, `<select>`      KEPT — the elements, not just text
 *
 *  So the real reach is per-ELEMENT, not document-global: an inline `style` cannot
 *  restyle the app the way a `<style>` element would have, and it cannot run
 *  script. The damage it does allow is a render beacon, a defacement overlay and a
 *  credential-shaped form — not code execution, and nothing leaves the origin that
 *  was not already going to be requested. Still worth closing, because model
 *  output is untrusted HTML by definition (see the header of `Markdown.tsx`): it
 *  arrives from RAG-indexed documents, MCP results, and files the model read.
 *
 *  A boundary a later edit can quietly widen is not a boundary, so the lists are
 *  named here and pinned by a test — and that test RUNS the real sanitiser, so a
 *  config DOMPurify ignored could no longer pass it.
 */
export const SANITIZE_CONFIG: Config = {
  FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select",
                "iframe", "object", "embed"],
  FORBID_ATTR: ["style", "action", "formaction", "srcset"],
};

type Sanitizer = (html: string, cfg?: Config) => string;

/** Sanitise marked's output through the boundary above.
 *
 *  The config is applied HERE rather than left to each call site, for two
 *  reasons. A second renderer cannot forget it, because there is no second place
 *  to pass it. And the boundary stays testable by handing in a spy instead of the
 *  real sanitiser — which was the only option when this project's test environment
 *  had no DOM. It has jsdom now, so the same seam carries both a spy that pins the
 *  config is PASSED and real DOMPurify that proves it WORKS.
 */
export function sanitizeHtml(raw: string, sanitize: Sanitizer): string {
  return sanitize(raw, SANITIZE_CONFIG);
}
