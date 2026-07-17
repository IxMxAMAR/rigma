"""Tool registry + handlers (safe + gated)."""
import pytest

from rigma import tools


def test_specs_filtered_by_permission():
    base = {t["function"]["name"] for t in tools.tool_specs()}
    assert {"web_search", "fetch_url", "calculator",
            "current_datetime"} <= base
    assert "run_python" not in base and "read_file" not in base  # gated hidden
    withcode = {t["function"]["name"]
                for t in tools.tool_specs(allow_code=True, workspace="/w")}
    assert "run_python" in withcode and "read_file" in withcode
    assert "search_my_documents" not in base                     # rag off
    assert "search_my_documents" in {
        t["function"]["name"] for t in tools.tool_specs(has_rag=True)}


def test_calculator_exact_and_safe():
    assert tools.run_tool("calculator", {"expression": "(1234 * 5.5) / 2"}) \
        == "3393.5"
    assert tools.run_tool("calculator", {"expression": "2 ** 10"}) == "1024"
    # no code execution / attribute access
    assert "error" in tools.run_tool(
        "calculator", {"expression": "__import__('os').system('echo hi')"})
    assert "error" in tools.run_tool("calculator",
                                     {"expression": "9 ** 99999"})  # DoS guard


def test_datetime_returns_now():
    out = tools.run_tool("current_datetime", {})
    assert "UTC" in out and "202" in out


def test_unknown_tool_and_broken_tool_return_text_not_raise():
    assert "no such tool" in tools.run_tool("nope", {})
    # a handler that raises comes back as an error string
    assert tools.run_tool("fetch_url", {"url": "not-a-url"}).startswith("error")


def test_web_search_tavily_and_ddg(monkeypatch):
    import types
    # Tavily path when a key is set
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(tools, "run_tool", tools.run_tool)  # noqa (keep import)

    class _R:
        def raise_for_status(self): pass
        def json(self): return {"results": [
            {"title": "Hello Wikipedia", "url": "https://en.wikipedia.org/Hello",
             "content": "Hello is a greeting."}]}
    import rigma.tools as T
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    out = tools.run_tool("web_search", {"query": "hello"})
    assert "Hello Wikipedia" in out and "wikipedia.org" in out

    # keyless DDG scrape path
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    page = ('<a class="result__a" href="https://duckduckgo.com/l/?uddg='
            'https%3A%2F%2Fexample.com">Example Site</a>'
            '<a class="result__snippet">an example page</a>')

    class _H:
        text = page
        def raise_for_status(self): pass
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _H())
    out = tools.run_tool("web_search", {"query": "example"})
    assert "Example Site" in out and "example.com" in out


def test_workspace_tools_confined(tmp_path):
    (tmp_path / "a.txt").write_text("hello file", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    ctx = {"workspace": str(tmp_path)}
    assert tools.run_tool("read_file", {"path": "a.txt"}, ctx) == "hello file"
    assert "a.txt" in tools.run_tool("list_directory", {}, ctx)
    # traversal is refused
    assert "outside the workspace" in tools.run_tool(
        "read_file", {"path": "../../etc/passwd"}, ctx)


def test_code_tools_gated(tmp_path):
    ctx = {"workspace": str(tmp_path), "allow_code": True}
    out = tools.run_tool("run_python", {"code": "print(6*7)"}, ctx)
    assert out.strip() == "42"
    # without the opt-in, it refuses
    assert "not enabled" in tools.run_tool(
        "run_python", {"code": "print(1)"}, {"workspace": str(tmp_path)})
