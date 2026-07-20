"""Mission compiler: raw prose in, a systematic spec out.

A free-text mission ("go through my images, work out my taste, then write 100
prompts in batches") forces a weak model to re-interpret intent on every turn —
and it interprets differently each time, which is how runs restart phases, skip
batching, and declare victory early.

So we compile it ONCE, at run start, into an explicit spec: objective,
deliverables with real paths, constraints, and numbered steps that each name the
artifact they must produce. The steps seed plan.json, so the model never invents
its own plan, and each artifact is checked ON DISK, so "done" is something the
server can verify rather than something the model claims.

Compilation is best-effort by design: if the local model returns unusable JSON we
fall back to a single-step spec and the run still starts. A degraded run beats a
refused one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAX_STEPS = 12

COMPILE_PROMPT = """You are a mission compiler. Turn the user's request into a \
strict JSON plan for an autonomous agent that will execute it step by step.

Rules:
- Break the work into 3-8 concrete, ordered steps. Each step must produce ONE
  artifact (a file) or gather information needed by a later step.
- If the user asks for output "in batches" or "separately", make EACH BATCH ITS
  OWN STEP with its own output file. Never collapse batches into one step.
- Artifact paths are RELATIVE to the agent's workspace folder — write
  "naming.md", never an absolute path, and NEVER invent folders. The single
  exception: if the user themselves wrote an absolute path, keep it exactly.
- `verification` says how a step is checked: "file_min_size" with a byte count
  for anything written, "none" for pure exploration steps.
- Output ONLY the JSON object. No commentary, no markdown fence.

JSON shape:
{"objective": "one sentence",
 "deliverables": [{"path": "file.txt", "description": "..."}],
 "constraints": ["..."],
 "steps": [{"id": 1, "description": "...", "artifact": "file.txt",
            "verification": {"type": "file_min_size", "value": 500}}]}

User's request:
"""


def _clean_json(text: str) -> str:
    """Strip fences/prose around a JSON object."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if start != -1 and end > start else t


def parse_spec(text: str):
    """Best-effort spec from model output. None if it isn't usable."""
    from . import tools
    obj, _ = tools.repair_json_args(_clean_json(text))
    if not isinstance(obj, dict):
        return None
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    clean = []
    for i, s in enumerate(steps[:MAX_STEPS], 1):
        if not isinstance(s, dict):
            continue
        desc = str(s.get("description") or s.get("task") or "").strip()
        if not desc:
            continue
        ver = s.get("verification") if isinstance(s.get("verification"), dict) else {}
        clean.append({
            "id": i,
            "description": desc[:300],
            "artifact": str(s.get("artifact") or s.get("expected_artifact") or ""),
            "verification": {"type": str(ver.get("type", "none")),
                             "value": int(ver.get("value", 0) or 0)},
        })
    if not clean:
        return None
    return {
        "objective": str(obj.get("objective") or "")[:500],
        "deliverables": [d for d in (obj.get("deliverables") or [])
                         if isinstance(d, dict) and d.get("path")][:10],
        "constraints": [str(c)[:200] for c in (obj.get("constraints") or [])][:10],
        "steps": clean,
        "compiled": True,
    }


def fallback_spec(raw: str) -> dict:
    """Never block a run on a failed compile — one step, the mission as given."""
    return {"objective": str(raw)[:500], "deliverables": [], "constraints": [],
            "steps": [{"id": 1, "description": str(raw)[:300], "artifact": "",
                       "verification": {"type": "none", "value": 0}}],
            "compiled": False}


def spec_block(spec: dict, raw: str) -> str:
    """The systematic form pinned in the system prompt, replacing raw prose.

    Same information every turn, in the same order, so the model reads its
    objective the same way each time instead of re-interpreting a paragraph."""
    if not spec.get("compiled"):
        return raw
    out = ["OBJECTIVE: " + spec["objective"]]
    if spec["deliverables"]:
        out.append("DELIVERABLES (these files must exist when you finish):")
        out += [f"  - {d['path']}" + (f" — {d.get('description','')}" if
                                      d.get("description") else "")
                for d in spec["deliverables"]]
    if spec["constraints"]:
        out.append("CONSTRAINTS:")
        out += [f"  - {c}" for c in spec["constraints"]]
    out.append("STEPS (work through them IN ORDER, one at a time):")
    for s in spec["steps"]:
        line = f"  {s['id']}. {s['description']}"
        if s["artifact"]:
            line += f"  -> writes: {s['artifact']}"
        out.append(line)
    return "\n".join(out)


def verify_step(step: dict, workspace: str = "") -> tuple[bool, str]:
    """Check a step's artifact ON DISK. (ok, reason) — the point is that the
    server decides a step is done, not the model."""
    ver = step.get("verification") or {}
    kind = str(ver.get("type", "none"))
    target = str(step.get("artifact") or "")
    if kind == "none" or not target:
        return True, ""
    p = Path(target)
    if not p.is_absolute() and workspace:
        p = Path(workspace) / target
    if not p.exists():
        return False, f"{p} does not exist"
    if kind == "file_min_size":
        size = p.stat().st_size
        want = int(ver.get("value", 0) or 0)
        if size < want:
            return False, f"{p.name} is only {size} bytes (expected >= {want})"
    return True, ""


def anchor_spec(spec: dict, workspace: str = "") -> dict:
    """Strip hallucinated absolute artifact paths down to workspace-relative.

    Live run #3 (2026-07-20): the compiler invented C:\\workspace\\naming.md —
    a folder that does not exist — because the old prompt's EXAMPLE showed
    absolute paths, and a weak model imitates the example over the rule. The
    server could then never verify the artifact (wrong path), never advanced
    the plan, and never credited the injected memories. Defence in depth: even
    with the prompt fixed, any absolute artifact whose parent directory does
    not actually exist is reduced to its basename, which verify_step resolves
    against the real workspace.
    """
    for coll, key in ((spec.get("steps") or [], "artifact"),
                      (spec.get("deliverables") or [], "path")):
        for item in coll:
            raw_p = str(item.get(key) or "")
            if not raw_p:
                continue
            p = Path(raw_p)
            if p.is_absolute() and not p.parent.exists():
                item[key] = p.name
    return spec


async def compile_mission(raw: str, post) -> dict:
    """Compile raw prose into a spec using the loaded model. `post` is an async
    callable taking the chat-completions payload. Falls back on any failure.

    The first attempt CONSTRAINS the output to JSON (llama.cpp builds a grammar
    from response_format, so the model physically cannot emit prose). A weak
    model asked politely for "only JSON" returns markdown and commentary — which
    is exactly why compilation kept failing and every run got a single-step
    fallback plan. If the server rejects response_format we retry unconstrained
    rather than lose the compile entirely."""
    base = {"messages": [{"role": "user", "content": COMPILE_PROMPT + raw}],
            "stream": False, "temperature": 0.2, "max_tokens": 2000}
    for payload in ({**base, "response_format": {"type": "json_object"}}, base):
        try:
            resp = await post(payload)
            text = resp["choices"][0]["message"]["content"] or ""
        except Exception:
            continue
        spec = parse_spec(text)
        if spec:
            return anchor_spec(spec)
    return fallback_spec(raw)
