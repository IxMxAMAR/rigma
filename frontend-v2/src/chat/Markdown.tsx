// Markdown rendering: marked -> DOMPurify -> highlight.js. Sanitisation is
// non-negotiable — model output is untrusted HTML by definition.
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { Marked } from "marked";
import { useMemo } from "react";
import "highlight.js/styles/github-dark-dimmed.css";

const marked = new Marked({
  gfm: true,
  breaks: true,
});

export default function Markdown({ text }: { text: string }) {
  const html = useMemo(() => {
    const raw = marked.parse(text, { async: false }) as string;
    const clean = DOMPurify.sanitize(raw);
    const el = document.createElement("div");
    el.innerHTML = clean;
    el.querySelectorAll("pre code").forEach((block) => {
      try {
        hljs.highlightElement(block as HTMLElement);
      } catch {
        /* unknown language: leave plain */
      }
    });
    return el.innerHTML;
  }, [text]);
  return (
    <div
      className="md prose-rigma min-w-0 break-words"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
