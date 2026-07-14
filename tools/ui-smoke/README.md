# UI browser smoke (dev-only)

Real-browser check of the chat UI against a fake OpenAI upstream. Catches the
class of app.js runtime faults that `node --check` and pytest cannot (e.g.
strict-mode ReferenceErrors) — added after exactly such a bug shipped.

Needs Playwright OUTSIDE the project venv (keep the project venv lean):

    python -m venv pwenv && pwenv/Scripts/pip install playwright
    pwenv/Scripts/python -m playwright install chromium-headless-shell

Run (two terminals, repo root):

    .venv/Scripts/python tools/ui-smoke/smoke_server.py          # serves :18500
    <pwenv>/Scripts/python tools/ui-smoke/smoke_browser.py out.png

Exit code 0 = all checks pass. The screenshot is a bonus design artifact.
