import type { Config } from "dompurify";

/** What the sanitiser may leave behind (AUDIT F58).
 *
 *  Kept in its own module, free of DOM and CSS imports, so a test can load it:
 *  `Markdown.tsx` pulls in highlight.js stylesheets and touches `document`, and
 *  neither belongs in a security-boundary test.
 *
 *  DOMPurify's stock allowlist is generous, and marked passes raw HTML through
 *  from wherever the text came from — a RAG-indexed document, an MCP result, a
 *  file the model read. It includes `style`, `form`, `input`, `button` and the
 *  `style`/`action` attributes. A `<style>` element is document-GLOBAL, so one
 *  arriving in model output restyles the whole app, and `background:
 *  url(https://…)` fires an outbound request the moment it renders. No script
 *  runs and the origin holds no secret, so the realistic damage is defacement
 *  plus a render beacon — but "model output is untrusted HTML by definition"
 *  (see the header of Markdown.tsx) means the allowlist should be the small
 *  thing, not the generous one.
 *
 *  A boundary a later edit can quietly widen is not a boundary, so the lists are
 *  named here and pinned by a test.
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
 *  to pass it. And the boundary becomes testable by handing in a spy instead of
 *  the real sanitiser — which matters, because this project's test environment
 *  has no DOM and `DOMPurify.sanitize` cannot be executed in it. Pinning that the
 *  config is DEFINED is not the same as pinning that it is USED, and the original
 *  bug was precisely a sanitiser called with no config at all.
 */
export function sanitizeHtml(raw: string, sanitize: Sanitizer): string {
  return sanitize(raw, SANITIZE_CONFIG);
}
