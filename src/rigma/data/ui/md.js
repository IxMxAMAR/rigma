// md.js — tiny markdown renderer. Pure: string in, HTML string out.
// The ENTIRE input is HTML-escaped first; markdown re-introduces safe tags only.
"use strict";
function mdEscape(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function mdInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\s)\*([^*\n]+)\*(?=\s|[.,!?;:]|$)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function renderMarkdown(src) {
  const lines = mdEscape(src).split("\n");
  let html = "", i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    const fence = ln.match(/^```+\s*([\w+-]*)/);
    if (fence) {                       // unclosed fence renders what it has
      const buf = [];
      for (i++; i < lines.length && !/^```+\s*$/.test(lines[i]); i++)
        buf.push(lines[i]);
      i++;
      html += '<pre data-lang="' + fence[1] + '"><code>' + buf.join("\n") +
              "</code></pre>";
      continue;
    }
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h) {                           // h3..h6: replies never outrank UI headings
      const n = h[1].length + 2;
      html += "<h" + n + ">" + mdInline(h[2]) + "</h" + n + ">"; i++; continue;
    }
    if (/^\s*[-*]\s+/.test(ln)) {
      html += "<ul>";
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        html += "<li>" + mdInline(lines[i].replace(/^\s*[-*]\s+/, "")) + "</li>";
        i++;
      }
      html += "</ul>"; continue;
    }
    if (/^\s*\d+\.\s+/.test(ln)) {
      html += "<ol>";
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        html += "<li>" + mdInline(lines[i].replace(/^\s*\d+\.\s+/, "")) + "</li>";
        i++;
      }
      html += "</ol>"; continue;
    }
    if (/^\s*&gt;\s?/.test(ln)) {      // '>' arrives escaped
      const buf = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*&gt;\s?/, "")); i++;
      }
      html += "<blockquote>" + mdInline(buf.join("<br>")) + "</blockquote>";
      continue;
    }
    if (ln.trim() === "") { i++; continue; }
    const buf = [];
    while (i < lines.length && lines[i].trim() !== "" &&
           !/^(```|#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*&gt;)/.test(lines[i])) {
      buf.push(lines[i]); i++;
    }
    if (!buf.length) { buf.push(lines[i]); i++; }   // never stall the scanner
    html += "<p>" + mdInline(buf.join("<br>")) + "</p>";
  }
  return html;
}
