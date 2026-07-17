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
    check("param sliders present (6 core + 6 advanced)",
          page.locator(".param-row").count() == 12)
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

    # image to the (vision-capable) smoke model passes the guard: engine answers
    okimg = page.evaluate("""async () => {
      const parts = [{type: "text", text: "see"},
                     {type: "image_url", image_url: {url: "data:image/png;base64,AA"}}];
      const r = await fetch("/api/sessions/" + current.id + "/chat", {
        method: "POST", headers: {"content-type": "application/json"},
        body: JSON.stringify({message: parts})});
      return r.status; }""")
    check("image to vision-capable model passes guard", okimg == 200)


    # ---- Hangar: models tab, drag-drop install, caps, remove ----
    import base64
    import struct

    def _s(bs):
        return struct.pack("<Q", len(bs)) + bs
    kvs = [
        _s(b"general.architecture") + struct.pack("<I", 8) + _s(b"qwen3"),
        _s(b"general.name") + struct.pack("<I", 8) + _s(b"Smoke Drop 1B"),
        _s(b"qwen3.block_count") + struct.pack("<II", 4, 4),
        _s(b"qwen3.context_length") + struct.pack("<II", 4, 8192),
        _s(b"qwen3.embedding_length") + struct.pack("<II", 4, 512),
        _s(b"qwen3.attention.head_count") + struct.pack("<II", 4, 8),
        _s(b"qwen3.attention.head_count_kv") + struct.pack("<II", 4, 2),
    ]
    gguf64 = base64.b64encode(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"\x00" * 128).decode()

    page.click("#open-models")                 # full main-area Models view
    page.wait_for_timeout(900)
    check("models view: opens full-area", page.locator("#models-view")
          .is_visible())
    check("models view: disk readout in header",
          "GB free" in (page.locator("#mv-disk").text_content() or ""))
    check("models view: drop zone", page.locator(".drop-zone").is_visible())
    check("models view: model grid cards",
          page.locator(".model-grid .model-card").count() >= 3)
    running_card = page.locator(".model-grid .model-card.running")
    check("models view: running model badged",
          "qwen3.6-35b-a3b" in (running_card.text_content() or ""))
    check("models view: vision cap chip on running model",
          running_card.locator(".cap.vision").count() == 1)
    check("models view: not-downloaded model shows a Download button",
          page.locator(".model-grid .model-card button", has_text="Download")
          .count() >= 1)
    check("models view: vision model offers its projector for download",
          page.locator(".model-grid .mmproj-row", has_text="vision projector")
          .count() >= 1)

    # drag-drop a synthetic gguf onto the zone (full upload->install->rerender)
    page.evaluate("""async (b64) => {
      const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const f = new File([bin], "SmokeDrop-Q4_0.gguf");
      const dt = new DataTransfer();
      dt.items.add(f);
      const zone = document.querySelector(".drop-zone");
      zone.dispatchEvent(new DragEvent("drop",
        {bubbles: true, dataTransfer: dt}));
    }""", gguf64)
    page.wait_for_timeout(1500)
    check("hangar: dropped gguf installed",
          page.locator(".model-card", has_text="smoke-drop-1b").count() == 1)
    dropped = page.locator(".model-card", has_text="smoke-drop-1b")
    check("hangar: custom badge + on-disk quant shown",
          dropped.locator(".badge", has_text="custom").count() == 1
          and dropped.locator(".dot.on").count() >= 1)

    # capability editor round-trip
    dropped.locator("button", has_text="Edit capabilities").click()
    page.wait_for_timeout(300)
    page.locator(".cap-row", has_text="tools").locator("input").check()
    page.locator("#mv-body button", has_text="Save").click()
    page.wait_for_timeout(900)
    check("hangar: capability saved",
          page.locator(".model-card", has_text="smoke-drop-1b")
              .locator(".cap", has_text="tools").count() == 1)

    # remove it again (keeps reruns clean); confirm dialog auto-accepted
    page.on("dialog", lambda d: d.accept())
    page.locator(".model-card", has_text="smoke-drop-1b") \
        .locator("button", has_text="Remove").click()
    page.wait_for_timeout(900)
    check("hangar: custom model removed",
          page.locator(".model-card", has_text="smoke-drop-1b").count() == 0)


    # ---- Bazaar: HF search -> fit verdicts -> add to library ----
    page.fill("#mv-body .mv-panel input >> nth=0", "webtune")   # HF search box
    page.locator(".mv-panel button", has_text="Search").click()
    page.wait_for_timeout(700)
    check("bazaar: search result listed",
          page.locator(".hf-row", has_text="cool/WebTune-GGUF").count() == 1)
    page.locator(".hf-row").first.click()
    page.wait_for_timeout(700)
    check("bazaar: fit verdicts rendered (gpu-fast vs slow offload)",
          page.locator(".fit.ok").count() == 1
          and page.locator(".fit.warn").count() == 1)
    check("bazaar: caps + mmproj surfaced",
          "mmproj included" in (page.locator(".hf-results").text_content() or ""))
    check("bazaar: quant tooltip present",
          bool(page.locator(".hf-results .quant-row .q").first
               .get_attribute("title")))
    check("bazaar: Recommended tag on the sweet-spot quant (not the biggest)",
          page.locator(".hf-results .rec-tag").count() == 1
          and "Q4_K_M" in (page.locator(".hf-results .rec-row")
                           .text_content() or ""))
    check("bazaar: offload quant flagged slow (warn), gpu quant ok",
          page.locator(".hf-results .fit.warn").count() == 1)
    check("bazaar: quant glossary present",
          page.locator(".quant-legend").count() >= 1)
    page.locator(".hf-results button", has_text="Add to library").click()
    page.wait_for_timeout(1000)
    # the browse panel is preserved (search state not nuked); button confirms
    check("bazaar: add confirms without nuking browse panel",
          page.locator(".hf-results button", has_text="Added").count() == 1)
    # the added model shows in Your Models IN REAL TIME (no reload needed)
    page.wait_for_timeout(1000)
    added = page.locator(".model-grid .model-card", has_text="web-tune-7b")
    check("bazaar: added model appears in grid live (no reload)",
          added.count() == 1
          and added.locator(".badge", has_text="custom").count() == 1)
    check("bazaar: browse panel still intact after add",
          page.locator(".hf-results .model-card").count() >= 1)
    added.locator("button", has_text="Remove").click()   # keep reruns clean
    page.wait_for_timeout(700)
    page.click("#mv-close")                     # back to chat
    page.wait_for_timeout(300)

    # ---- Unload: free VRAM/RAM without killing the UI ----
    page.click("#gear")
    page.wait_for_timeout(200)
    page.click("#drawer-tabs button[data-tab=server]")
    page.wait_for_timeout(600)
    unload_btn = page.locator("#drawer-body button",
                              has_text="Unload engine")
    if unload_btn.count():                      # rerun-tolerant
        unload_btn.click()
        page.wait_for_timeout(900)
    check("unload: server tab offers reload",
          page.locator("#drawer-body button", has_text="again").count() == 1)
    check("unload: engine chip shows unloaded",
          "unloaded" in (page.locator("#engine-label").text_content() or ""))
    page.click("#drawer-close")

    # ---- B6 features: palette, author's note, prefill, stats ----
    page.keyboard.press("Control+k")
    page.wait_for_timeout(300)
    check("palette: opens on Ctrl+K",
          not page.locator("#palette").get_attribute("hidden") is not None
          or page.locator("#palette").is_visible())
    page.fill("#palette-input", "new")
    page.wait_for_timeout(200)
    check("palette: filters to matching action",
          page.locator(".palette-row", has_text="New chat").count() >= 1)
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    page.click("#new-chat")
    page.wait_for_timeout(200)
    page.click("#an-toggle")
    check("author's note editor opens", page.locator("#an-edit").is_visible())
    page.fill("#an-text", "Keep the tone ominous.")
    page.fill("#an-depth", "2")
    page.click("#log")
    page.wait_for_timeout(400)
    an = page.evaluate("current && current.authors_note")
    check("author's note persists", an == "Keep the tone ominous.")

    page.click("#tools-toggle")
    page.wait_for_timeout(400)
    check("tools toggle turns on",
          page.evaluate("current && current.use_tools") is True
          and "on" in (page.locator("#tools-toggle").text_content() or ""))
    page.click("#tools-toggle")
    page.wait_for_timeout(300)

    page.click("#prefill-toggle")
    check("prefill row opens", page.locator("#prefill-row").is_visible())
    page.fill("#prefill-text", "Sure, here goes:")
    page.click("#log")
    page.wait_for_timeout(400)
    check("prefill persists",
          page.evaluate("current && current.prefill") == "Sure, here goes:")

    page.click("#engine-chip")
    page.wait_for_timeout(500)
    # stats only render once a turn has recorded predicted_n; smoke upstream
    # sends timings so at least the endpoint must not error
    check("stats endpoint reachable",
          page.evaluate("""async () => {
            const r = await fetch('/api/server/stats');
            return r.ok; }"""))
    page.click("#drawer-close")

    check("no console/page errors at end", not console_errors)
    page.screenshot(path=SHOT, full_page=False)
    b.close()

if console_errors:
    print("CONSOLE ERRORS:", *console_errors[:10], sep="\n  ", flush=True)
print(f"screenshot: {SHOT}", flush=True)
sys.exit(1 if failures else 0)
