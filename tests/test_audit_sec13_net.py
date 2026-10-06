"""13-1: the vetted address is the address dialed; work-doing GETs are guarded.

The resolver is stubbed and the transport is recorded, so no name is looked up
and no socket is opened.
"""
import socket

import pytest

from rigma import serve, tools

# --- 13-1: the vetted address is the address that is dialed -------------------
#
# The resolver is stubbed, so no name is ever looked up and no socket is opened.

def _fake_addrinfo(*ips):
    def _res(host, port=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
                for ip in ips]
    return _res


def test_every_address_a_name_answers_is_validated(monkeypatch):
    # the split-horizon answer that defeats check-then-connect: one public, one
    # loopback. Validating "the first" or "the last" is what the bug did.
    monkeypatch.setattr(tools, "_resolve_addresses",
                        _fake_addrinfo("8.8.8.8", "127.0.0.1"))
    assert tools._vetted_ip("evil.example") is None
    monkeypatch.setattr(tools, "_resolve_addresses",
                        _fake_addrinfo("8.8.8.8"))
    assert tools._vetted_ip("evil.example") == "8.8.8.8"


def test_the_transport_dials_the_vetted_ip_not_a_second_resolution(monkeypatch):
    import httpx
    monkeypatch.setattr(tools, "_resolve_addresses",
                        _fake_addrinfo("8.8.8.8"))
    seen = {}

    def _record(self, request):
        seen["url"] = str(request.url)
        seen["host"] = request.headers.get("host")
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _record)
    req = httpx.Request("GET", "https://evil.example/secret")
    tools._pin_request(req)
    tools._pinned_transport().handle_request(req)
    # connected to the vetted address, while Host/SNI keep the real name
    assert "8.8.8.8" in seen["url"]
    assert "evil.example" not in seen["url"].split("/")[2]
    assert seen["host"] == "evil.example"
    assert seen["sni"] == "evil.example"


def test_a_private_target_is_refused_before_any_connect(monkeypatch):
    import httpx
    monkeypatch.setattr(tools, "_resolve_addresses",
                        _fake_addrinfo("127.0.0.1"))
    with pytest.raises(ValueError):
        tools._pin_request(httpx.Request("GET", "http://evil.example/"))


# --- 13-1/13-5: work-doing GETs need same-origin proof and a rate limit -------

def test_a_cross_site_get_to_a_work_route_is_refused():
    h = "127.0.0.1:11500"
    assert serve.guard_request(h, "", sec_fetch_site="cross-site",
                               path="/api/hf/repo") != ""
    assert serve.guard_request(h, "", sec_fetch_site="same-site",
                               path="/api/rag/discover") != ""
    # a non-browser client sends no Sec-Fetch-* — the fallback
    assert serve.guard_request(h, "", path="/api/hf/repo") == ""
    # the UI's own same-origin fetch
    assert serve.guard_request(h, "", sec_fetch_site="same-origin",
                               path="/api/hf/repo") == ""
    # a cheap route is unaffected
    assert serve.guard_request(h, "", sec_fetch_site="cross-site",
                               path="/api/sessions") == ""
    # the existing two-argument contract still holds
    assert serve.guard_request(h, "") == ""


def test_a_work_route_is_rate_limited():
    path = "/api/hf/repo"
    serve._rate_hits.pop(path, None)
    n = serve._RATE_MAX_PER_WINDOW
    results = [serve.work_route_rate_limited(path, now=1000.0 + i * 0.1)
               for i in range(n + 2)]
    assert results[:n] == [False] * n
    assert results[-1] is True
    # the window slides: a hit after the window is admitted again
    assert serve.work_route_rate_limited(
        path, now=1000.0 + serve._RATE_WINDOW_S + 1) is False


def test_the_middleware_refuses_a_cross_site_work_get():
    import asyncio
    called = []

    async def _app(scope, receive, send):
        called.append(True)

    out = {}

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]

    scope = {"type": "http", "method": "GET", "path": "/api/hf/repo",
             "headers": [(b"host", b"127.0.0.1:11500"),
                         (b"sec-fetch-site", b"cross-site")]}
    asyncio.run(serve.LocalOriginGuard(_app)(scope, receive, send))
    assert out["status"] == 403
    assert called == []          # the route handler never ran


def test_the_middleware_admits_the_address_the_connection_arrived_on():
    # The wiring half of the same fix: the guard only admits the proxy's Host
    # when the middleware hands it scope["server"], so a pod behind Runpod's
    # edge reaches the UI instead of a 403 on every request.
    import asyncio
    called = []

    async def _app(scope, receive, send):
        called.append(True)

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        pass

    scope = {"type": "http", "method": "GET", "path": "/",
             "server": ("100.65.29.116", 11500),
             "headers": [(b"host", b"100.65.29.116:11500")]}
    asyncio.run(serve.LocalOriginGuard(_app)(scope, receive, send))
    assert called == [True]      # admitted: the pod's own address, not a name

    # A scope with no "server" key at all (older callers, bare test scopes)
    # must still be refused rather than crash.
    called.clear()
    scope.pop("server")
    asyncio.run(serve.LocalOriginGuard(_app)(scope, receive, send))
    assert called == []

