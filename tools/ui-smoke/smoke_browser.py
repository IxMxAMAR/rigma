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

    check("no console/page errors at end", not console_errors)
    page.screenshot(path=SHOT, full_page=False)
    b.close()

if console_errors:
    print("CONSOLE ERRORS:", *console_errors[:10], sep="\n  ", flush=True)
print(f"screenshot: {SHOT}", flush=True)
sys.exit(1 if failures else 0)
