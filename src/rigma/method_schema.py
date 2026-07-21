"""The Method document: one shape, validated in one place.

A Method used to be a preset with a nicer name. It is now an authorable
document whose Rules, Macros and Workflows are three presentations of ONE
primitive -- a step list. That is what makes a single validator and a single
interpreter possible, and it is why this module has no rigma imports: it is
pure data rules, so both the save path and the builder tools can lean on it
without dragging the server in.

Spec: docs/superpowers/specs/2026-07-21-custom-methods-design.md
"""
from __future__ import annotations

import re

# --- the safety allowlist -------------------------------------------------
# Deliberately explicit, and deliberately NOT tools.Tool.safe: `remember` and
# `http_request` are declared safe=True yet write memory and can issue
# mutating POSTs. A macro runs unattended after one click, so this list is
# read-only-or-nothing.
SAFE_TOOLS = frozenset({
    "read_file", "list_directory", "find_files", "grep", "sample_files",
    "recall", "current_datetime", "calculator", "search_my_documents",
    "web_search", "fetch_url"})

STEP_KINDS = ("tool", "prompt", "settings", "note", "new_chat")
RULE_KINDS = ("standing", "trigger")
TRIGGER_EVENTS = ("tool_ran", "turn_ended", "every_n_turns", "method_applied")
TRIGGER_MODES = ("nudge", "run")
CARRY_FIELDS = ("notes", "method", "workspace", "params", "use_rag",
                "system_prompt")
SETTINGS_FIELDS = ("effort", "params", "use_tools", "allow_code")
EFFORTS = ("", "off", "auto", "on")
VAR_KINDS = ("text", "path", "number")

# placeholders: {{name}} or {{name:argument}}
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)(?::([^}]*))?\}\}")
# resolved from the run, not from `vars` -- these never need declaring
BUILTIN_PLACEHOLDERS = frozenset({
    "last_reply", "selection", "transcript", "title_next", "step", "ask"})

_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_MAX_STEPS = 20


def find_placeholders(obj) -> list[tuple[str, str]]:
    """Every (name, argument) placeholder in any string leaf of `obj`."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, str):
        out.extend((m.group(1), m.group(2) or "")
                   for m in PLACEHOLDER.finditer(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_placeholders(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(find_placeholders(v))
    return out


def slugify(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")[:40]
    base = base or "item"
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _components(doc: dict) -> list[dict]:
    return [c for key in ("rules", "macros", "workflows")
            for c in doc.get(key) or []]


def normalize(doc: dict) -> dict:
    """Return a full document: every optional key present, every component
    carrying an id. Never mutates the input -- built-ins are module-level
    literals and a normalize() that edited them in place would corrupt the
    catalog for the life of the process."""
    out = {
        "id": doc.get("id", ""),
        "name": doc.get("name", ""),
        "tagline": doc.get("tagline", ""),
        "builtin": bool(doc.get("builtin", False)),
        "version": int(doc.get("version", 1) or 1),
        "extends": doc.get("extends") or None,
        "apply": dict(doc.get("apply") or {}),
        "guide": list(doc.get("guide") or []),
        "vars": {k: dict(v) for k, v in (doc.get("vars") or {}).items()},
    }
    taken: set[str] = set()
    for key in ("rules", "macros", "workflows"):
        items = []
        for raw in doc.get(key) or []:
            c = {k: (list(v) if isinstance(v, list)
                     else dict(v) if isinstance(v, dict) else v)
                 for k, v in raw.items()}
            cid = str(c.get("id") or "")
            if not _ID_RE.match(cid) or cid in taken:
                cid = slugify(c.get("label") or c.get("kind") or key, taken)
            c["id"] = cid
            taken.add(cid)
            if key != "rules":
                c.setdefault("label", cid)
                c.setdefault("hint", "")
                c.setdefault("steps", [])
            items.append(c)
        out[key] = items
    return out


def _validate_steps(steps, where: str, var_names: set[str],
                    tool_names: set[str], errs: list[str]) -> None:
    if not isinstance(steps, list):
        errs.append(f"{where}: steps must be a list")
        return
    if len(steps) > _MAX_STEPS:
        errs.append(f"{where}: {len(steps)} steps is over the {_MAX_STEPS} "
                    "limit -- split it into two macros")
    for i, step in enumerate(steps):
        at = f"{where} step {i}"
        if not isinstance(step, dict):
            errs.append(f"{at}: must be an object")
            continue
        kind = step.get("kind")
        if kind not in STEP_KINDS:
            errs.append(f"{at}: unknown step kind '{kind}' -- use one of "
                        f"{', '.join(STEP_KINDS)}")
            continue
        if kind == "tool":
            name = str(step.get("name") or "")
            if name not in tool_names:
                errs.append(f"{at}: no such tool '{name}'")
            if not isinstance(step.get("args", {}), dict):
                errs.append(f"{at}: args must be an object")
        elif kind == "prompt":
            if not str(step.get("text") or "").strip():
                errs.append(f"{at}: prompt needs text")
            if step.get("to", "chat") not in ("chat", "aux"):
                errs.append(f"{at}: 'to' must be 'chat' or 'aux'")
        elif kind == "settings":
            st = step.get("set")
            if not isinstance(st, dict) or not st:
                errs.append(f"{at}: settings needs a non-empty 'set' object")
            else:
                for k in st:
                    if k not in SETTINGS_FIELDS:
                        errs.append(f"{at}: '{k}' is not settable -- use "
                                    f"{', '.join(SETTINGS_FIELDS)}")
                if "effort" in st and st["effort"] not in EFFORTS:
                    errs.append(f"{at}: effort must be one of "
                                f"{', '.join(repr(e) for e in EFFORTS)}")
        elif kind == "note":
            if step.get("op") not in ("append", "replace"):
                errs.append(f"{at}: note op must be 'append' or 'replace'")
        elif kind == "new_chat":
            for f in step.get("carry") or []:
                if f not in CARRY_FIELDS:
                    errs.append(f"{at}: cannot carry '{f}' -- carry one of "
                                f"{', '.join(CARRY_FIELDS)}")
        # placeholders resolve against vars, earlier steps, or the builtins
        for pname, parg in find_placeholders(step):
            if pname == "step":
                if not parg.isdigit() or int(parg) >= i:
                    errs.append(f"{at}: {{{{step:{parg}}}}} does not refer to "
                                "an earlier step in this list")
            elif pname in BUILTIN_PLACEHOLDERS:
                continue
            elif pname not in var_names:
                errs.append(f"{at}: {{{{{pname}}}}} is not a declared "
                            "variable -- add it with set_var or fix the name")


def validate(doc: dict, tool_names: set[str]) -> list[str]:
    """Every problem with `doc`, phrased so a model can fix it in one turn.
    Empty list means the document is safe to save."""
    errs: list[str] = []
    if not _ID_RE.match(str(doc.get("id") or "")):
        errs.append("id must be 1-40 characters of a-z, 0-9 and underscore")
    if not str(doc.get("name") or "").strip():
        errs.append("name is required")
    apply = doc.get("apply")
    if not isinstance(apply, dict) or not str(
            apply.get("system_prompt") or "").strip():
        errs.append("apply.system_prompt is required")
    elif apply.get("effort", "") not in EFFORTS:
        errs.append("apply.effort must be one of "
                    + ", ".join(repr(e) for e in EFFORTS))

    var_names = set(doc.get("vars") or {})
    for k, v in (doc.get("vars") or {}).items():
        if not _ID_RE.match(k):
            errs.append(f"var '{k}': name must be a-z, 0-9 and underscore")
        if (v or {}).get("kind", "text") not in VAR_KINDS:
            errs.append(f"var '{k}': kind must be one of "
                        + ", ".join(VAR_KINDS))

    seen: set[str] = set()
    for c in _components(doc):
        cid = str(c.get("id") or "")
        if cid in seen:
            errs.append(f"duplicate component id '{cid}'")
        seen.add(cid)

    macro_ids = {str(m.get("id")) for m in doc.get("macros") or []}
    for r in doc.get("rules") or []:
        rid = r.get("id")
        kind = r.get("kind")
        if kind not in RULE_KINDS:
            errs.append(f"rule '{rid}': kind must be 'standing' or 'trigger'")
            continue
        if kind == "standing":
            if not str(r.get("text") or "").strip():
                errs.append(f"rule '{rid}': a standing rule needs text")
            continue
        on, do = r.get("on") or {}, r.get("do") or {}
        if on.get("event") not in TRIGGER_EVENTS:
            errs.append(f"rule '{rid}': unknown event "
                        f"'{on.get('event')}' -- use one of "
                        + ", ".join(TRIGGER_EVENTS))
        if on.get("event") == "tool_ran" and on.get("tool") \
                and on["tool"] not in tool_names:
            errs.append(f"rule '{rid}': no such tool '{on['tool']}'")
        if on.get("event") == "every_n_turns" and int(on.get("n", 0) or 0) < 1:
            errs.append(f"rule '{rid}': every_n_turns needs n >= 1")
        mode = do.get("mode", "nudge")
        if mode not in TRIGGER_MODES:
            errs.append(f"rule '{rid}': mode must be 'nudge' or 'run'")
        elif mode == "nudge" and not str(do.get("text") or "").strip():
            errs.append(f"rule '{rid}': a nudge needs text")
        elif mode == "run" and do.get("macro") not in macro_ids:
            errs.append(f"rule '{rid}': no macro named "
                        f"'{do.get('macro')}' in this method")

    for key in ("macros", "workflows"):
        for c in doc.get(key) or []:
            if not str(c.get("label") or "").strip():
                errs.append(f"{key[:-1]} '{c.get('id')}': label is required")
            _validate_steps(c.get("steps"), f"{key[:-1]} '{c.get('id')}'",
                            var_names, tool_names, errs)
    return errs
