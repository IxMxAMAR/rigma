"""Headless-browser smoke of the v2 React shell — what `/` actually serves.

smoke_browser.py drives the legacy DOM, which moved to `/rizz` at the cutover,
so it no longer exercises the shipping UI at all. This is the browser-level
guard for `/`: boot the shell, walk every sidebar surface, and fail on any
console/page error or on the error boundary rendering.

Deliberately shallow. The v2 surfaces have their own vitest suite; the point of
a real browser here is the class of fault only a browser shows — a
strict-mode ReferenceError, a mount that throws, a bundle that 404s.
"""
import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:18500"          # `/` serves the v2 shell
SHOT = sys.argv[1] if len(sys.argv) > 1 else "v2-smoke.png"
SURFACES = ["Chat", "Autonomous", "Models", "Engine", "Memory", "Skills",
            "Workflows", "Settings"]
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
    page.wait_for_selector("#root")
    page.wait_for_selector('nav[aria-label="Primary"]', timeout=10000)
    check("boot: v2 shell mounted at /",
          "RIGMA" in (page.locator('nav[aria-label="Primary"]').text_content() or ""))
    check("boot: chat composer rendered",
          page.locator('textarea[aria-label="Message"]').is_visible())
    check("boot: no console/page errors", not console_errors)

    # every sidebar surface must render without taking the shell down
    for label in SURFACES:
        page.locator('nav[aria-label="Primary"] button',
                     has_text=label).first.click()
        page.wait_for_timeout(350)
        heading = (page.locator("header h1").text_content() or "").strip()
        crashed = "this screen crashed" in (page.locator("body").text_content() or "")
        check(f"surface {label}: renders", heading == label and not crashed)
    check("no console/page errors after walking every surface",
          not console_errors)

    page.screenshot(path=SHOT, full_page=False)
    b.close()

if console_errors:
    print("CONSOLE ERRORS:", *console_errors[:10], sep="\n  ", flush=True)
print(f"screenshot: {SHOT}", flush=True)
sys.exit(1 if failures else 0)
