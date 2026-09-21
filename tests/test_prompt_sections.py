"""The agent system prompt, as named ordered sections.

The refactor from one inline string to a section registry is only safe if it is
PROVABLY a refactor, so the first test compares the assembly against a golden
copy of exactly what shipped before (`tests/golden/agent_system_prompt.txt`,
captured from the constant immediately before the split). Everything after that
pins the properties the split buys: stable order, real names, and a prompt that
does not describe tools the tool surface withheld.
"""
from pathlib import Path

import pytest

from rigma import prompt as _prompt
from rigma import serve

GOLDEN = Path(__file__).parent / "golden" / "agent_system_prompt.txt"


def _golden() -> str:
    # newline="" — compare bytes, not a platform's idea of a line ending
    with open(GOLDEN, encoding="utf-8", newline="") as fh:
        return fh.read()


def test_the_assembled_prompt_is_byte_identical_to_what_shipped():
    """A refactor, not a rewrite. If this fails the prompt changed, and a
    prompt change is a model-behaviour change that needs a live A/B.

    The caps spell out what "what shipped" WAS: a vision model on the full tool
    surface, because before the tier table existed every run carried every tool
    and the prompt named `web_search` accordingly. Dropping "extended" here does
    not reproduce the old prompt — it reproduces the focused one.
    """
    assert _prompt.agent_prompt(caps=("vision", "extended")) == _golden()


def test_serve_still_exposes_the_same_constant():
    """serve.AGENT_SYSTEM_PROMPT is read by the run-start path and by nothing
    else; it must keep its value and its name."""
    assert serve.AGENT_SYSTEM_PROMPT == _golden()


def test_sections_are_named_and_ordered():
    p = _prompt._registry(("vision",))
    assert p.names() == [
        "agent.identity",
        "agent.every_turn",
        "agent.loop",
        "agent.orientation",
        "agent.no_promise",
        "agent.no_fabrication",
        "agent.recovery",
    ]


def test_order_is_the_section_order_not_the_registration_order():
    p = _prompt.Prompt()
    p.add("third", 30, "C")
    p.add("first", 10, "A")
    p.add("second", 20, "B")
    assert p.names() == ["first", "second", "third"]
    assert p.render() == "A\n\nB\n\nC"


def test_equal_orders_are_stable_not_dict_order():
    """Two sections claiming one position must not produce a prompt whose
    order depends on insertion — that would silently reshape the cached
    prefix between processes."""
    a = _prompt.Prompt()
    a.add("zebra", 10, "Z")
    a.add("alpha", 10, "A")
    b = _prompt.Prompt()
    b.add("alpha", 10, "A")
    b.add("zebra", 10, "Z")
    assert a.names() == b.names() == ["alpha", "zebra"]


def test_a_duplicate_section_name_is_an_error():
    p = _prompt.Prompt()
    p.add("rules", 10, "one")
    with pytest.raises(_prompt.PromptError, match="already registered"):
        p.add("rules", 20, "two")


def test_a_section_requiring_a_missing_capability_is_omitted():
    p = _prompt.Prompt(caps=set())
    p.add("always", 10, "here")
    p.add("needs_vision", 20, "not here", requires=("vision",))
    assert p.names() == ["always"]
    assert p.render() == "here"

    with_vision = _prompt.Prompt(caps={"vision"})
    with_vision.add("always", 10, "here")
    with_vision.add("needs_vision", 20, "and here", requires=("vision",))
    assert with_vision.render() == "here\n\nand here"


def test_empty_text_contributes_nothing():
    """The memory block is empty on a machine that has learned nothing; an
    empty section must not leave a blank line behind."""
    p = _prompt.Prompt()
    p.add("a", 10, "A")
    p.add("gap", 20, "")
    p.add("b", 30, "B")
    assert p.render() == "A\n\nB"


def test_without_vision_the_prompt_never_names_a_vision_tool():
    """The vision tools are withheld from a model without vision
    (`tools.tool_specs`, needs="vision"). A prompt that names them is asking
    for a tool that is not on the wire — including in the example tool list."""
    text = _prompt.agent_prompt(caps=set())
    assert "view_images" not in text
    assert "view_sample" not in text
    # it must still give a remedy the sightless model can actually use
    assert "sample_files" in text
    assert "read_file" in text


def test_with_vision_the_prompt_still_names_them():
    text = _prompt.agent_prompt(caps=("vision",))
    assert "view_images(folder=..., count=N)" in text
    assert "view_sample()" in text


def test_a_run_prompt_agrees_with_the_tool_surface():
    """The two halves of one question: does the engine have vision. The prompt
    must name vision tools exactly when the tool surface offers them."""
    from rigma import tools as toolkit
    for has_vision in (True, False):
        offered = {s["function"]["name"] for s in toolkit.tool_specs(
            allow_code=True, workspace="C:/x", has_vision=has_vision,
            has_run=True)}
        text = _prompt.agent_prompt(
            caps={"vision"} if has_vision else set())
        for vision_tool in ("view_images", "view_sample", "view_image"):
            assert (vision_tool in offered) == (vision_tool in text), (
                f"vision={has_vision}: {vision_tool} offered={vision_tool in offered} "
                f"named={vision_tool in text}")


def test_learned_pitfalls_is_the_last_section():
    """What earlier runs learned reads as the final word, not as an
    interruption in the middle of the instructions."""
    base = _prompt.agent_prompt(caps=("vision",))
    with_mem = _prompt.agent_prompt_with_memory(
        "you cannot retype long filenames", caps=("vision",))
    assert with_mem.startswith(base)
    assert with_mem.endswith("you cannot retype long filenames")
    assert _prompt.MEMORY_ORDER > 70        # after every agent.* section


def test_an_empty_memory_block_changes_nothing():
    base = _prompt.agent_prompt(caps=("vision",))
    assert _prompt.agent_prompt_with_memory("", caps=("vision",)) == base
    assert _prompt.agent_prompt_with_memory("   \n ", caps=("vision",)) == base


def test_the_prompt_never_names_a_tool_the_surface_withheld():
    """The menu in `agent.loop` IS the model's menu, so every tool it names has
    to be on the wire for that session.

    The vision half of this was got right; the tier half was not. A focused Run
    withholds `web_search` (tools._TIER, extended) and the prompt named it
    anyway — the same prompt/tool-surface disagreement the module exists to
    stop, in a second place. Both axes are checked here, and the tool vocabulary
    is read from `tool_specs` rather than hand-listed, so a tool added later
    cannot quietly escape the check.
    """
    import re

    from rigma import tools as toolkit
    base = dict(allow_code=True, workspace="C:/x", has_run=True, has_rag=True)
    vocabulary = {s["function"]["name"] for s in toolkit.tool_specs(
        surface="all", has_vision=True, **base)}
    for has_vision in (True, False):
        for surface in ("all", "focused"):
            caps = ({"vision"} if has_vision else set())
            if surface == "all":
                caps.add("extended")
            text = _prompt.agent_prompt(caps=caps)
            offered = {s["function"]["name"] for s in toolkit.tool_specs(
                surface=surface, has_vision=has_vision, **base)}
            named = {w for w in re.findall(r"[a-z_][a-z0-9_]{3,}", text)
                     if w in vocabulary}
            assert named <= offered, (
                f"vision={has_vision} surface={surface}: prompt names "
                f"{sorted(named - offered)}, which that surface withholds")


def test_the_surface_decision_has_one_home():
    """`_tool_surface` is what the round loop and the Run's prompt both read.
    Computing the rule twice in two handlers is how they drift apart."""
    assert serve._tool_surface("run-1") == "focused"
    assert serve._tool_surface("") == "all"          # plain chat stays lean-all


def test_an_unknown_surface_env_falls_back_to_all(monkeypatch):
    """A typo must not silently produce a third, undefined surface."""
    monkeypatch.setenv("RIGMA_TOOL_SURFACE", "focussed")   # sic
    assert serve._tool_surface("run-1") == "all"
    monkeypatch.setenv("RIGMA_TOOL_SURFACE", "all")
    assert serve._tool_surface("run-1") == "all"
