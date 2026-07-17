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
    url = str(args.get("url", "")).strip()
    if not re.match(r"^https?://", url):
        return "error: url must start with http:// or https://"
    with _public_client() as c:
        r = c.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
    except ValueError as e:
        return f"error: {e}"                    # e.g. 'number too large'
    except Exception:
        return f"error: couldn't evaluate '{expr}'"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _bounded(v):
    # a single guard on the RESULT magnitude catches nested-pow DoS
    # (((2**999)**999)**999) that a per-node exponent check misses
    if isinstance(v, int) and v.bit_length() > 4096:
        raise ValueError("number too large")
    if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
        raise ValueError("number too large")
    return v


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _bounded(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _bounded(_OPS[type(node.op)](_safe_eval(node.left),
                                            _safe_eval(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _bounded(_OPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError("unsupported expression")


def _is_public_host(host: str) -> bool:
    """True only if `host` resolves entirely to public addresses — blocks the
    model (possibly prompt-injected by a fetched page) from reaching localhost,
    cloud metadata (169.254.169.254), or the LAN."""
    import ipaddress
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


def _public_client():
    """httpx client that refuses private/loopback targets on EVERY hop (the
    request hook fires again on each redirect, so a public URL can't bounce
    the fetch to an internal address)."""
    import httpx
    from urllib.parse import urlparse

    def guard(request):
        if not _is_public_host(urlparse(str(request.url)).hostname or ""):
            raise ValueError("refusing to reach a private/loopback address")
    return httpx.Client(follow_redirects=True, timeout=25,
                        event_hooks={"request": [guard]})


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
    """Resolve a path INSIDE the session's workspace root; refuse escapes.

    Uses is_relative_to, NOT a string prefix — `str(p).startswith(str(root))`
    would let /workspace2/evil escape a /workspace root."""
    ws = (ctx.get("workspace") or "").strip()
    if not ws:
        raise ValueError("no workspace folder is set for this chat")
    root = Path(ws).resolve()
    if not root.is_dir():
        raise ValueError("the workspace folder doesn't exist")
    p = (root / rel).resolve()
    if p != root and not p.is_relative_to(root):
        raise ValueError("path is outside the workspace")
    return p


@tool("http_request",
      "Make an HTTP request to any API (GET or POST with headers/JSON body) "
      "and return the response. Use for APIs, not just reading web pages.",
      {"type": "object", "properties": {
          "url": {"type": "string"},
          "method": {"type": "string", "description": "GET or POST"},
          "headers": {"type": "object"},
          "json": {"type": "object", "description": "JSON body for POST"}},
       "required": ["url"]})
def _http_request(args, ctx):
    url = str(args.get("url", ""))
    if not re.match(r"^https?://", url):
        return "error: url must start with http:// or https://"
    method = str(args.get("method", "GET")).upper()
    with _public_client() as c:
        r = c.request(method, url, headers=args.get("headers") or None,
                      json=args.get("json") if method == "POST" else None)
    body = r.text
    return (f"HTTP {r.status_code}\n"
            + body[:6000] + ("\n…(truncated)" if len(body) > 6000 else ""))


@tool("system_info",
      "Report the machine's OS, CPU, RAM, disk, and GPU. Use to reason about "
      "what will run well here.",
      {"type": "object", "properties": {}})
def _system_info(args, ctx):
    import platform as _p

    import psutil
    m = psutil.virtual_memory()
    du = psutil.disk_usage(str(Path.home()))
    lines = [f"OS: {_p.system()} {_p.release()}",
             f"CPU: {_p.processor() or _p.machine()} · {psutil.cpu_count()} cores",
             f"RAM: {m.available / 2**30:.1f} free / {m.total / 2**30:.1f} GB",
             f"Disk: {du.free / 2**30:.0f} free / {du.total / 2**30:.0f} GB"]
    try:
        from .probe import probe_hardware
        from .registry import Registry
        for g in probe_hardware(Registry.load().gpus):
            lines.append(f"GPU: {g.name} · {g.vram_mb / 1024:.0f} GB VRAM "
                         f"· {'/'.join(g.backends)}")
    except Exception:
        pass
    return "\n".join(lines)


@tool("remember",
      "Save a durable note to your own memory so you can recall it in future "
      "chats (facts about the user, preferences, ongoing work).",
      {"type": "object", "properties": {
          "key": {"type": "string"}, "value": {"type": "string"}},
       "required": ["key", "value"]})
def _remember(args, ctx):
    from .runtime import rigma_home
    f = rigma_home() / "model_memory.json"
    mem = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    mem[str(args.get("key"))] = str(args.get("value"))
    f.write_text(json.dumps(mem, indent=1), encoding="utf-8")
    return f"remembered '{args.get('key')}'"


@tool("recall",
      "Look up something you saved earlier with `remember` (omit key to list "
      "everything you remember).",
      {"type": "object", "properties": {"key": {"type": "string"}}})
def _recall(args, ctx):
    from .runtime import rigma_home
    f = rigma_home() / "model_memory.json"
    if not f.exists():
        return "(nothing remembered yet)"
    mem = json.loads(f.read_text(encoding="utf-8"))
    key = args.get("key")
    if key:
        return mem.get(str(key), f"(nothing remembered for '{key}')")
    return "\n".join(f"{k}: {v}" for k, v in mem.items()) or "(empty)"


@tool("find_files",
      "Find files by glob pattern inside the workspace (e.g. '**/*.py', "
      "'src/*.js'). Like a file search.",
      {"type": "object", "properties": {
          "pattern": {"type": "string"}}, "required": ["pattern"]},
      needs="workspace")
def _find_files(args, ctx):
    root = _ws_path(ctx, ".")
    pat = str(args.get("pattern", "*"))
    hits = [str(p.relative_to(root).as_posix())
            for p in sorted(root.glob(pat))[:200] if p.is_file()]
    return "\n".join(hits) if hits else f"no files match {pat}"


@tool("grep",
      "Search file contents for a regex inside the workspace. Returns matching "
      "lines with file:line.",
      {"type": "object", "properties": {
          "pattern": {"type": "string"},
          "glob": {"type": "string", "description": "limit to files matching "
                   "this glob (default all text files)"}},
       "required": ["pattern"]},
      needs="workspace")
def _grep(args, ctx):
    root = _ws_path(ctx, ".")
    try:
        rx = re.compile(str(args.get("pattern", "")))
    except re.error as e:
        return f"error: bad regex: {e}"
    glob = str(args.get("glob", "") or "**/*")
    out, seen = [], 0
    for p in sorted(root.glob(glob)):
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8",
                                                 errors="ignore").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{p.relative_to(root).as_posix()}:{i}: "
                               + line.strip()[:200])
                    seen += 1
                    if seen >= 100:
                        return "\n".join(out) + "\n…(more matches)"
        except OSError:
            continue
    return "\n".join(out) if out else "no matches"


@tool("edit_file",
      "Replace an exact string in a workspace file with a new string (the old "
      "string must appear exactly once). Use for surgical edits.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "old": {"type": "string"},
          "new": {"type": "string"}},
       "required": ["path", "old", "new"]},
      safe=False, needs="code")
def _edit_file(args, ctx):
    p = _ws_path(ctx, str(args.get("path", "")))
    if not p.is_file():
        return f"error: no such file: {args.get('path')}"
    text = p.read_text(encoding="utf-8")
    old = str(args.get("old", ""))
    n = text.count(old)
    if n == 0:
        return "error: the old string wasn't found in the file"
    if n > 1:
        return f"error: the old string appears {n} times — make it unique"
    p.write_text(text.replace(old, str(args.get("new", "")), 1),
                 encoding="utf-8")
    return f"edited {args.get('path')}"


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


@tool("write_file",
      "Create or overwrite a text file inside the chat's workspace folder.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "path relative to the "
                   "workspace"},
          "content": {"type": "string", "description": "the file's contents"}},
       "required": ["path", "content"]},
      safe=False, needs="code")
def _write_file(args, ctx):
    p = _ws_path(ctx, str(args.get("path", "")))
    p.parent.mkdir(parents=True, exist_ok=True)
    content = str(args.get("content", ""))
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {args.get('path')}"


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
