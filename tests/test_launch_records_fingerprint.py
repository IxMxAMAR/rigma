"""A launch must record the cache fingerprint it launched under.

The bug this pins: `rigma up` wrote state without `kv_fp`, and `write_state`
reverts unnamed fields to their defaults BY DESIGN, so it wrote an empty string.
Nothing failed. `serve._prefix_ctx` returns None on a falsy fingerprint, so
prefix warm/snapshot became silent no-ops and `perform_unload` skipped its KV
save — both caches were off for every CLI-started run, and the only symptom was
that long conversations were mysteriously slow to start.

An empty fingerprint is the SAFE default (fail closed: no cache is restored
rather than the wrong one), which is exactly why the mistake is invisible. So
the invariant is checked structurally: any `write_state` call that records a LIVE
engine must name `kv_fp`. `engine_pid=-1` means "no engine" and is exempt.

Parsed with `ast` rather than matched with a regex, because the argument lists
are multi-line and keyword-based, and a regex that only sometimes sees the whole
call is worse than no guard.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rigma"


def _write_state_calls():
    """Every `write_state(...)` call in the package, as (file, node)."""
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name == "write_state":
                yield path, node


def _kw(node: ast.Call, key: str):
    for k in node.keywords:
        if k.arg == key:
            return k.value
    return None


def test_every_call_that_records_a_live_engine_names_its_cache_fingerprint():
    offenders = []
    for path, node in _write_state_calls():
        pid = _kw(node, "engine_pid")
        if pid is None:
            continue
        # engine_pid=-1 is the explicit "no engine" record (the UI-up-without-a-
        # model path); there is no cache to name and blank is correct.
        if isinstance(pid, ast.UnaryOp) and isinstance(pid.operand, ast.Constant):
            if pid.operand.value == 1:
                continue
        if _kw(node, "kv_fp") is None:
            offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these write_state calls record a running engine but no kv_fp, which "
        "silently disables prefix caching AND restore-on-unload: "
        + ", ".join(offenders))


def test_both_launch_paths_agree_by_construction():
    """`rigma up` and the UI reload must derive the fingerprint the SAME way.

    Two copies of the expression is how they drift, and a drifted fingerprint
    does not error — it restores a cache taken under a different configuration,
    which generates fluent text from a history that never happened.
    """
    for name in ("cli.py", "server_ops.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert "launch_fingerprint(" in src, (
            f"{name} launches an engine but does not use the shared "
            "kvcache.launch_fingerprint")
        assert "kvcache.fingerprint(kvcache.config_of(" not in src, (
            f"{name} derives the fingerprint inline again — use the shared "
            "helper so the two launch paths cannot disagree")


def test_the_guard_would_have_caught_the_original_bug():
    """The guard is only worth having if it fails on the code that caused it.

    Reproduces the pre-fix call verbatim, minus `kv_fp`.
    """
    bug = ast.parse(
        "st.write_state(rp.model_slug, rp.gguf.quant, port,\n"
        "               engine_pid=sp.proc.pid, ui_pid=os.getpid(),\n"
        "               backend=rp.backend, ctx=rp.flags.ctx)\n")
    call = next(n for n in ast.walk(bug) if isinstance(n, ast.Call))
    assert _kw(call, "kv_fp") is None
    assert _kw(call, "engine_pid") is not None
