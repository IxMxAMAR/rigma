"""Tools the local model can call, and the machinery to run them.

A tool is a typed function the model may invoke: Rigma advertises the schema to
llama-server, the model emits a tool_call, Rigma runs the handler here and feeds
the result back. Tools are tiered by risk:

  safe  — read-only, no side effects (search, fetch, math, time, doc lookup).
          Run automatically.
  gated — touches the filesystem or runs code. Only offered when the session
          explicitly opts in (session["allow_code"] / a workspace root), so
          there's never surprise code execution.
"""
from __future__ import annotations

import ast
import html
import json
import operator
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema for the arguments
    handler: Callable[..., str]
    safe: bool = True         # safe -> auto-run; gated -> needs opt-in
    needs: str = ""           # optional capability the session must grant


_REGISTRY: dict[str, Tool] = {}


def tool(name, description, parameters, safe=True, needs=""):
    def wrap(fn):
        _REGISTRY[name] = Tool(name, description, parameters, fn, safe, needs)
        return fn
    return wrap


def tool_specs(allow_code: bool = False, has_rag: bool = False,
               workspace: str | None = None) -> list[dict]:
    """OpenAI-format tool definitions to hand the model, filtered to what this
    session actually permits."""
    out = []
    for t in _REGISTRY.values():
        if t.needs == "code" and not allow_code:
            continue
        if t.needs == "rag" and not has_rag:
            continue
        if t.needs == "workspace" and not workspace:
            continue
        out.append({"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": t.parameters}})
    return out


def run_tool(name: str, args: dict, ctx: dict | None = None) -> str:
    """Execute a tool by name. Returns a plain-text result the model reads;
    never raises — errors come back as text so the model can react."""
    t = _REGISTRY.get(name)
    if t is None:
        return f"error: no such tool '{name}'"
    ctx = ctx or {}
    if t.needs == "code" and not ctx.get("allow_code"):
        return "error: code execution is not enabled for this chat"
    try:
        return t.handler(args or {}, ctx)
    except Exception as e:   # a broken tool must not kill the turn
        return f"error running {name}: {e}"


# ---- safe tools --------------------------------------------------------------

@tool("web_search",
      "Search the web and return the top results (title, URL, snippet). Use "
      "for current events, facts, docs, or anything you don't already know.",
      {"type": "object", "properties": {
          "query": {"type": "string", "description": "the search query"},
          "count": {"type": "integer", "description": "results to return (1-8)"}},
       "required": ["query"]})
def _web_search(args, ctx):
    import httpx
    q = str(args.get("query", "")).strip()
    if not q:
        return "error: empty query"
    n = max(1, min(int(args.get("count", 5) or 5), 8))
    # a real key beats scraping when present
    tav = os.environ.get("TAVILY_API_KEY", "")
    if tav:
        r = httpx.post("https://api.tavily.com/search", timeout=20, json={
            "api_key": tav, "query": q, "max_results": n})
        r.raise_for_status()
        hits = r.json().get("results", [])[:n]
        return _fmt_results([(h.get("title", ""), h.get("url", ""),
                              h.get("content", "")) for h in hits], q)
    # keyless fallback: DuckDuckGo's HTML endpoint
    r = httpx.post("https://html.duckduckgo.com/html/", timeout=20,
                   data={"q": q}, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    hits = _parse_ddg(r.text)[:n]
    if not hits:
        return f"no results for '{q}'."
    return _fmt_results(hits, q)


def _parse_ddg(page: str):
    out = []
    for m in re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page,
            re.S):
        url, title = m.group(1), _strip(m.group(2))
        um = re.search(r"uddg=([^&]+)", url)   # unwrap DDG redirect
        if um:
            from urllib.parse import unquote
            url = unquote(um.group(1))
        out.append((title, url, ""))
    snips = [_strip(s) for s in re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)]
    return [(t, u, snips[i] if i < len(snips) else "")
            for i, (t, u, _) in enumerate(out)]


def _fmt_results(hits, q):
    lines = [f"Search results for '{q}':"]
    for i, (title, url, snip) in enumerate(hits, 1):
        lines.append(f"\n{i}. {title}\n   {url}"
                     + (f"\n   {snip[:300]}" if snip else ""))
    return "\n".join(lines)


@tool("fetch_url",
      "Fetch a web page and return its readable text (tags stripped). Use to "
      "read a page a search turned up.",
      {"type": "object", "properties": {
          "url": {"type": "string", "description": "the http(s) URL to fetch"}},
       "required": ["url"]})
def _fetch_url(args, ctx):
    import httpx
    url = str(args.get("url", "")).strip()
    if not re.match(r"^https?://", url):
        return "error: url must start with http:// or https://"
    r = httpx.get(url, timeout=25, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    body = re.sub(r"(?is)<(script|style|noscript|head)[^>]*>.*?</\1>", " ",
                  r.text)
    text = _strip(re.sub(r"(?s)<[^>]+>", " ", body))
    text = re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", text)).strip()
    return text[:6000] + ("\n…(truncated)" if len(text) > 6000 else "")


@tool("calculator",
      "Evaluate an arithmetic expression exactly (+, -, *, /, //, %, **, "
      "parentheses). Use instead of doing math in your head.",
      {"type": "object", "properties": {
          "expression": {"type": "string",
                         "description": "e.g. (1234 * 5.5) / 3"}},
       "required": ["expression"]})
def _calculator(args, ctx):
    expr = str(args.get("expression", ""))
    try:
        return str(_safe_eval(ast.parse(expr, mode="eval").body))
    except Exception:
        return f"error: couldn't evaluate '{expr}'"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow):        # block giant exponents (DoS)
            r = _safe_eval(node.right)
            if isinstance(r, (int, float)) and r > 1000:
                raise ValueError("exponent too large")
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool("current_datetime",
      "Get the current local date and time. Use for anything time-relative "
      "('today', 'now', 'this year').",
      {"type": "object", "properties": {}})
def _current_datetime(args, ctx):
    now = datetime.now()
    return (now.strftime("%A, %d %B %Y, %H:%M:%S")
            + " (local) · " + datetime.now(timezone.utc).strftime("%H:%M UTC"))


@tool("search_my_documents",
      "Search the user's own indexed documents (their private files). Use when "
      "the question is about their notes/codebase/papers, not the open web.",
      {"type": "object", "properties": {
          "query": {"type": "string"}}, "required": ["query"]},
      needs="rag")
def _search_docs(args, ctx):
    from . import rag
    q = str(args.get("query", "")).strip()
    if not q:
        return "error: empty query"
    port = rag.recorded_sidecar_port()
    if not port:
        return "no documents are indexed yet."
    a = rag.ask(q, port=port)
    if not isinstance(a, dict):
        return "documents unavailable."
    cites = a.get("citations") or []
    out = a.get("answer", "") or "(no answer)"
    if cites:
        out += "\n\nsources: " + ", ".join(
            c.get("source", "") if isinstance(c, dict) else str(c)
            for c in cites[:5])
    return out


# ---- gated tools (filesystem + code) ----------------------------------------

def _ws_path(ctx, rel: str) -> Path:
    """Resolve a path INSIDE the session's workspace root; refuse escapes."""
    root = Path(ctx.get("workspace", "")).resolve()
    p = (root / rel).resolve()
    if root == Path("") or not str(p).startswith(str(root)):
        raise ValueError("path is outside the workspace")
    return p


@tool("read_file",
      "Read a text file inside the chat's workspace folder.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "path relative to the "
                   "workspace"}}, "required": ["path"]},
      needs="workspace")
def _read_file(args, ctx):
    p = _ws_path(ctx, str(args.get("path", "")))
    if not p.is_file():
        return f"error: no such file: {args.get('path')}"
    if p.stat().st_size > 400_000:
        return "error: file too large to read"
    return p.read_text(encoding="utf-8", errors="replace")[:20000]


@tool("list_directory",
      "List files and folders inside the chat's workspace folder.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "folder path relative to "
                   "the workspace (default: root)"}}},
      needs="workspace")
def _list_dir(args, ctx):
    p = _ws_path(ctx, str(args.get("path", "") or "."))
    if not p.is_dir():
        return f"error: not a folder: {args.get('path')}"
    items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    return "\n".join(("📄 " if x.is_file() else "📁 ") + x.name
                     for x in items[:200]) or "(empty)"


@tool("run_python",
      "Run a short Python 3 snippet and return its stdout/stderr. For "
      "calculations, data wrangling, quick checks.",
      {"type": "object", "properties": {
          "code": {"type": "string", "description": "the Python source to run"}},
       "required": ["code"]},
      safe=False, needs="code")
def _run_python(args, ctx):
    code = str(args.get("code", ""))
    return _run_subprocess(["python", "-I", "-c", code], ctx)


@tool("run_shell",
      "Run a shell command and return its output. Use sparingly.",
      {"type": "object", "properties": {
          "command": {"type": "string"}}, "required": ["command"]},
      safe=False, needs="code")
def _run_shell(args, ctx):
    cmd = str(args.get("command", ""))
    return _run_subprocess(cmd, ctx, shell=True)


def _run_subprocess(cmd, ctx, shell=False):
    cwd = ctx.get("workspace") or None
    try:
        r = subprocess.run(cmd, shell=shell, cwd=cwd, capture_output=True,
                           text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "error: timed out after 30s"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    out = out.strip() or f"(no output, exit {r.returncode})"
    return out[:8000] + ("\n…(truncated)" if len(out) > 8000 else "")


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"(?s)<[^>]+>", "", s)).strip()
