"""The focused tool surface: what a Run advertises, and the way back.

An autonomous Run used to advertise all 32 permitted tools. Measured
2026-09-20 on the reference machine, that is 16,490 chars of JSON schema
(~4,100 tok) — 14.6% of a 32K window spent before the mission, the run-state
block, or one message of history. These tests pin the saving AND the two
properties that make it safe: a tier is never a permission, and `use_tools` is
the way back to everything the tier table held down.
"""
import json

from rigma import tools


def _run_specs(**kw):
    """The full Run surface: code, workspace, vision and RAG all available."""
    base = dict(allow_code=True, has_rag=True, workspace="C:/x",
                has_vision=True, has_run=True)
    return tools.tool_specs(**{**base, **kw})


def _chars(specs):
    return len(json.dumps(specs))


def test_focused_surface_is_materially_smaller_than_all():
    full = _run_specs(surface="all")
    focused = _run_specs(surface="focused")
    assert len(focused) < len(full)
    # the saving is the point: assert it is a real reduction, not a token one
    assert _chars(focused) < _chars(full) * 0.85, (
        f"focused {_chars(focused)} vs all {_chars(full)}")


def test_default_surface_is_unchanged():
    # nothing changes until it is asked for — the default is the old behaviour
    assert _run_specs() == _run_specs(surface="all")


def test_focused_keeps_the_workhorse_tools():
    names = {s["function"]["name"] for s in _run_specs(surface="focused")}
    for must in ("read_file", "write_file", "edit_file", "list_directory",
                 "find_files", "grep", "run_shell", "run_python",
                 "manage_plan", "task_complete", "sample_files",
                 "view_images", "delegate", "undo_last_change", "ask_user"):
        assert must in names, must


def test_focused_holds_back_the_tail():
    names = {s["function"]["name"] for s in _run_specs(surface="focused")}
    for held in ("calculator", "current_datetime", "system_info",
                 "start_job", "job_output", "kill_job", "remember", "recall",
                 "view_image", "http_request", "ask_gemini",
                 "web_search", "fetch_url", "move_files", "copy_files"):
        assert held not in names, held


def test_use_tools_is_offered_only_on_a_focused_surface():
    assert "use_tools" not in {s["function"]["name"]
                               for s in _run_specs(surface="all")}
    assert "use_tools" in {s["function"]["name"]
                           for s in _run_specs(surface="focused")}


def test_unlocked_tools_come_back():
    focused = {s["function"]["name"]
               for s in _run_specs(surface="focused")}
    assert "move_files" not in focused
    after = {s["function"]["name"] for s in _run_specs(
        surface="focused", unlocked=["move_files", "web_search"])}
    assert {"move_files", "web_search"} <= after


def test_unlock_never_widens_permission():
    """A tier is not a permission, and neither is an unlock.

    `view_image` is vision-gated. Asking for it on a session whose model has no
    vision must not put it on the wire — otherwise `use_tools` becomes a way to
    talk a model into a tool the session deliberately withheld.
    """
    names = {s["function"]["name"] for s in tools.tool_specs(
        allow_code=False, has_rag=False, workspace=None, has_vision=False,
        has_run=False, surface="focused", unlocked=["view_image",
                                                    "search_my_documents",
                                                    "run_shell"])}
    assert "view_image" not in names
    assert "search_my_documents" not in names      # rag not indexed
    assert "run_shell" not in names                # code not permitted


def test_builder_chat_is_unaffected_by_surface():
    """The creation chat's whole safety property is that it gets builder tools
    and NOTHING else; a surface must not become a way around that."""
    names = {s["function"]["name"] for s in tools.tool_specs(
        allow_code=True, workspace="C:/x", builder_only=True,
        surface="focused", unlocked=["write_file", "run_shell"])}
    assert "write_file" not in names and "run_shell" not in names


def test_lockable_names_are_exactly_the_permitted_set():
    """`use_tools` may only reach what permission already allowed, so the two
    must be derived from one source of truth."""
    kw = dict(allow_code=True, has_rag=True, workspace="C:/x",
              has_vision=True, has_run=True)
    allowed = tools.lockable_names(**kw)
    full = {s["function"]["name"] for s in tools.tool_specs(**kw)}
    assert allowed == full
    assert "use_tools" not in allowed


def test_unknown_tool_defaults_to_core():
    """A tool added without a tier must land in the always-on set rather than
    silently disappearing from every focused surface."""
    assert tools.tier_of("a_tool_that_does_not_exist") == "core"
    assert tools.tier_of("read_file") == "core"
    assert tools.tier_of("calculator") == "specialist"


def test_use_tools_routes_to_the_loop():
    """serve.py routes this by NAME, before the handler runs — the same way
    `delegate` is routed. So the contract to pin is the handler's payload, not
    the sentinel surviving run_tool (which defuses control bytes on purpose:
    the sentinel is only a fallback for callers that never intercept)."""
    handler = tools._REGISTRY["use_tools"].handler
    out = handler({"names": ["move_files"]}, {})
    assert out.startswith(tools.USE_TOOLS_SENTINEL)
    payload = json.loads(out[len(tools.USE_TOOLS_SENTINEL):])
    assert payload["names"] == ["move_files"]
    # a single name as a bare string is accepted, not iterated into characters
    out = handler({"names": "web_search"}, {})
    payload = json.loads(out[len(tools.USE_TOOLS_SENTINEL):])
    assert payload["names"] == ["web_search"]
    assert json.loads(
        handler({"help": "list"}, {})[len(tools.USE_TOOLS_SENTINEL):]
    )["help"] == "list"
