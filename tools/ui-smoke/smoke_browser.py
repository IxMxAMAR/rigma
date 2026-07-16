"""Headless-browser smoke of the Cockpit UI against the smoke server."""
import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:18500"
SHOT = sys.argv[1] if len(sys.argv) > 1 else "cockpit-smoke.png"
failures = []
console_errors = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        failures.append(name)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 860})
    page.on("console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))

    page.goto(BASE)
    page.wait_for_timeout(600)
    check("boot: no console/page errors", not console_errors)
    check("boot: preset picker populated",
          page.locator("#preset-pick option").count() >= 4)  # none + 3 builtins

    # always exercise a FRESH chat (the server may carry prior-run state)
    page.click("#new-chat")
    page.wait_for_timeout(300)

    # send a message end-to-end
    page.fill("#in", "tell me a story")
    page.press("#in", "Enter")
    page.wait_for_timeout(300)
    stop_mid = page.locator("#send").text_content()
    page.wait_for_selector(".bot .body strong", timeout=8000)  # **smoke** rendered
    page.wait_for_timeout(400)
    body = page.locator(".bot .body").last.text_content()
    check("turn: reply streamed + markdown rendered",
          "smoke" in body and "Hello" in body)
    check("turn: send morphed to Stop mid-stream", stop_mid == "Stop")
    check("turn: send restored after", page.locator("#send").text_content() == "Send")
    check("turn: rail shows the chat",
          page.locator(".chat-item .title").first.text_content().startswith("tell me"))
    check("turn: tok/s from meta",
          "42.5" in (page.locator("#tps").text_content() or ""))
    check("turn: ctx bar live",
          "live" in (page.locator("#ctx-bar").get_attribute("class") or ""))

    # message actions appear (hover the bot message)
    page.hover(".bot")
    page.wait_for_timeout(150)
    acts = page.locator(".actions").last.locator("button")
    labels = [acts.nth(i).text_content() for i in range(acts.count())]
    check("actions: copy/edit/delete/regenerate/continue",
          {"copy", "edit", "delete", "regenerate", "continue"} <= set(labels))

    # regenerate -> variant flipper appears
    page.locator(".actions button", has_text="regenerate").last.click()
    page.wait_for_timeout(1500)
    flips = page.locator(".actions .flip")
    check("regenerate: variant flipper present", flips.count() >= 1)

    # sys bar + notes + preset select
    page.click("#sys-toggle")
    check("sys editor opens", page.locator("#sys-edit").is_visible())
    page.click("#sys-toggle")
    page.click("#notes-toggle")
    check("notes editor opens", page.locator("#notes-edit").is_visible())
    page.fill("#notes-edit", "Ember is the dragon.")
    page.click("#log")  # blur -> save
    page.wait_for_timeout(400)
    page.select_option("#preset-pick", "usecase:creative")
    page.wait_for_timeout(400)
    sysprev = page.locator("#sys-preview").text_content() or ""
    check("preset applied to session", "reative" in sysprev or "preset" in sysprev)

    # docs panel holds the RAG toggle now
    page.click("#docs-toggle")
    check("rag toggle lives in docs panel", page.locator("#use-rag").is_visible())


    # ---- Phase 3: drawer, params, presets manager, search, branch ----
    page.click("#gear")
    check("drawer opens on gear", page.locator("#drawer").is_visible())
    check("param sliders present (5 core + 6 advanced)",
          page.locator(".param-row").count() == 11)
    page.fill(".param-row input.val >> nth=0", "1.2")   # temperature number box
    page.wait_for_timeout(700)                            # debounce + save
    sess = page.evaluate("current && current.params")
    check("param persisted to session", sess and abs(sess.get("temperature", 0) - 1.2) < 1e-6)
    page.click("#drawer-tabs button[data-tab=presets]")
    check("presets manager lists built-ins",
          page.locator(".preset-row.builtin").count() >= 3)
    page.click("#drawer-close")

    # search
    page.fill("#rail-search", "tell me")
    page.wait_for_timeout(600)
    check("search finds the chat", page.locator(".chat-item .snippet").count() >= 1)
    page.fill("#rail-search", "")
    page.wait_for_timeout(500)

    # branch from first message
    n_before = page.locator(".chat-item").count()
    page.hover(".msg >> nth=0")
    page.locator(".actions button", has_text="branch").first.click()
    page.wait_for_timeout(800)
    check("branch created + opened",
          "(branch)" in (page.locator(".chat-item.active .title").first.text_content() or ""))


    # ---- Phase 4: engine chip, server tab, fit advisor ----
    check("engine chip rendered", page.locator("#engine-chip").is_visible())
    page.click("#engine-chip")
    page.wait_for_timeout(600)
    check("server tab opens from chip",
          "verdict" in (page.locator("#drawer-body").text_content() or ""))
    check("log tail rendered",
          len((page.locator("pre.srv-log").text_content() or "")) > 0)
    page.click("#drawer-close")

    # fit advisor on ctx overflow
    page.click("#new-chat")
    page.wait_for_timeout(300)
    page.fill("#in", "OVERFLOW please")
    page.press("#in", "Enter")
    page.wait_for_selector(".advisor", timeout=8000)
    check("fit advisor appears on ctx overflow",
          "fit advisor" in (page.locator(".advisor").text_content() or ""))
    check("error bubble preserved with advisor",
          "exceeds the available context size"
          in (page.locator(".bot.error").last.text_content() or ""))

    # drawer must not cross-write params after a session switch (P3 critical)
    page.click("#gear")
    page.wait_for_timeout(300)
    page.fill(".param-row input.val >> nth=0", "3.7")
    page.click("#new-chat")
    page.wait_for_timeout(400)
    page.fill(".param-row input.val >> nth=1", "0.4")   # stale editor nudge
    page.wait_for_timeout(700)
    fresh = page.evaluate("current && current.params")
    check("stale drawer cannot cross-write params",
          not fresh or "temperature" not in (fresh or {}))
    page.click("#drawer-close")


    # ---- Forge: think block, effort chip, compact, advanced samplers ----
    page.click("#new-chat")
    page.wait_for_timeout(300)
    page.fill("#in", "THINK about this")
    page.press("#in", "Enter")
    page.wait_for_selector(".think", timeout=8000)
    page.wait_for_timeout(1500)
    check("think block rendered", page.locator(".think").count() >= 1)
    check("think collapses when reply starts",
          "closed" in (page.locator(".think").last.get_attribute("class") or ""))
    check("effort chip visible (thinking-capable model)",
          page.locator("#effort-toggle").is_visible())
    page.click("#effort-toggle")
    page.wait_for_timeout(400)
    eff = page.evaluate("current && current.effort")
    check("effort cycles to off", eff == "off")

    # compact via drawer (needs >6 messages -> seed via API)
    seed = page.evaluate("""async () => {
      const msgs = [];
      for (let i = 0; i < 12; i++)
        msgs.push({role: i % 2 ? "assistant" : "user", content: "turn " + i});
      await api("POST", "/api/sessions/" + current.id, {messages: msgs});
      return true; }""")
    check("seeded 12 turns", bool(seed))
    page.click("#gear")
    page.wait_for_timeout(400)
    page.locator("#drawer-body button", has_text="Compact now").click()
    page.wait_for_timeout(2500)
    remaining = page.evaluate("current && current.messages.length")
    digest = page.evaluate("current && current.digest")
    check("compact kept 6 + wrote digest",
          remaining == 6 and bool(digest))

    # image attach chip (no vision model in smoke -> server must 400 politely)
    ok400 = page.evaluate("""async () => {
      const parts = [{type: "text", text: "see"},
                     {type: "image_url", image_url: {url: "data:image/png;base64,AA"}}];
      const r = await fetch("/api/sessions/" + current.id + "/chat", {
        method: "POST", headers: {"content-type": "application/json"},
        body: JSON.stringify({message: parts})});
      return r.status; }""")
    check("image to non-vision model politely 400s", ok400 == 400)
    # the 400 above was deliberate; drop its console echo before the final gate
    console_errors[:] = [e for e in console_errors if "400" not in e]

    check("no console/page errors at end", not console_errors)
    page.screenshot(path=SHOT, full_page=False)
    b.close()

if console_errors:
    print("CONSOLE ERRORS:", *console_errors[:10], sep="\n  ", flush=True)
print(f"screenshot: {SHOT}", flush=True)
sys.exit(1 if failures else 0)
