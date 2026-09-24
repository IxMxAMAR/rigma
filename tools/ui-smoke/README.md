# UI browser smoke (dev-only)

Real-browser check of the chat UI against a fake OpenAI upstream. Catches the
class of app.js runtime faults that `node --check` and pytest cannot (e.g.
strict-mode ReferenceErrors) — added after exactly such a bug shipped.

Two scripts, because the cutover split the UI:

- `smoke_browser.py` — the LEGACY Cockpit UI, now served at `/rizz` (it used to
  be at `/`). Deep end-to-end walk of the legacy DOM.
- `smoke_browser_v2.py` — the v2 React shell that `/` actually serves. Shallow:
  boot, walk every sidebar surface, fail on any console error or error-boundary
  render. The v2 surfaces have their own vitest suite; this is the browser-level
  guard for the shipping page.

Needs Playwright OUTSIDE the project venv (keep the project venv lean):

    python -m venv pwenv && pwenv/Scripts/pip install playwright
    pwenv/Scripts/python -m playwright install chromium-headless-shell

Run (two terminals, repo root):

    .venv/Scripts/python tools/ui-smoke/smoke_server.py          # serves :18500
    <pwenv>/Scripts/python tools/ui-smoke/smoke_browser.py out.png
    <pwenv>/Scripts/python tools/ui-smoke/smoke_browser_v2.py v2.png

Exit code 0 = all checks pass. The screenshot is a bonus design artifact.
