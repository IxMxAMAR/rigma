"""Things llama-server says once, quietly, that change what you are running.

The engine reports several decisions as a single WARN line at load and then
never mentions them again. They sit in a log nobody opens while the behaviour
they describe persists for the whole session.

The one that prompted this module, found in this machine's own logs:

    W srv load_model: cache_reuse is not supported by this context,
                      it will be disabled

Rigma passes `--cache-reuse 256`. On a hybrid / DeltaNet architecture the
engine cannot KV-shift, so it turns the flag off. Everything still works; edits
that are not a clean prefix just reprocess from scratch, silently, forever. The
flag being *set* and the flag being *in effect* are different facts, and only
the log knew which one was true.

Pure function over log text, so the patterns are testable without an engine.
"""
from __future__ import annotations

import re

# (id, compiled pattern, severity, what it means for the user)
#
# `confirmed` marks patterns actually observed in this machine's logs. The
# others come from llama.cpp's source and upstream reports and are matched
# best-effort — a pattern that never fires is invisible, which is the right
# failure mode for a diagnostic.
_PATTERNS: list[tuple[str, re.Pattern, str, str, bool]] = [
    ("cache_reuse_disabled",
     re.compile(r"cache[_ ]reuse is not supported", re.I),
     "warn",
     "KV-shift reuse is OFF — this model's architecture does not support it, "
     "so the engine disabled --cache-reuse. Edits that are not a clean prefix "
     "reprocess from scratch. Nothing to fix: it is an architecture limit, not "
     "a setting.",
     True),
    ("swa_disabled",
     re.compile(r"n_swa\s*=\s*0|swa.*will be disabled", re.I),
     "info",
     "--swa-full is a no-op here: the engine reports no sliding window, so the "
     "flag disables itself. On hybrid models the reprocessing trigger is the "
     "recurrent component, which that flag does not touch.",
     False),
    ("checkpoint_cascade",
     re.compile(r"erasing.*checkpoint|checkpoint.*eras|no checkpoint found",
                re.I),
     "warn",
     "Context checkpoints were discarded, which forces a full prompt "
     "reprocess. Hybrid and recurrent layers cannot be rewound, so when no "
     "checkpoint matches the resume position the engine drops them all. "
     "Lowering --checkpoint-min-step toward your turn length makes this rarer.",
     False),
    ("kv_cache_full",
     re.compile(r"KV cache is full|context (?:is )?full|slot context shift",
                re.I),
     "warn",
     "The context filled and the engine shifted or dropped tokens. The oldest "
     "part of the conversation is gone from the model's view.",
     False),
    ("fallback_template",
     re.compile(r"chat template.*not found|using default chat template", re.I),
     "warn",
     "No chat template in the gguf, so the engine is using a generic one. The "
     "model is being prompted in a format it was not trained on.",
     False),
]


def findings(log_text: str) -> list[dict]:
    """Significant one-off engine statements found in a log.

    Deduplicated: these fire once per load, and a log spanning several restarts
    would otherwise report the same fact many times. Ordered as listed, so the
    confirmed and consequential ones come first.
    """
    out = []
    for key, pat, severity, message, confirmed in _PATTERNS:
        hits = [ln.strip() for ln in (log_text or "").splitlines()
                if pat.search(ln)]
        if not hits:
            continue
        out.append({
            "id": key,
            "severity": severity,
            "message": message,
            "count": len(hits),
            "confirmed_here": confirmed,
            # the last occurrence: on a log covering several launches it is the
            # one describing the engine currently running
            "example": hits[-1][:300],
        })
    return out
