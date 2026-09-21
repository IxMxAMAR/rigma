"""The system prompt, as named ordered sections instead of one growing string.

WHY THIS EXISTS.

`AGENT_SYSTEM_PROMPT` had grown one clause per observed failure into a 20-line
block. Every rule in it was earned and every rule is load-bearing — but as ONE
string the only thing a test could assert was a substring of the whole, nothing
could be reordered without re-reading the entire paragraph, and the only way to
say something different for a different model was to write a second block by
hand and let the two drift apart.

A section is now a named, ordered, individually testable unit. Assembly is
deterministic: sections sort by `order`, then by name, so two sections claiming
one position still produce a stable prompt rather than a dict-order lottery that
would silently reshape the cached prefix.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

1. It does not own per-turn text. Anything that changes every turn belongs in
   the run-state block appended AFTER history (serve.py) — moving it up here
   would rewrite the first tokens of the prompt every turn and invalidate the
   whole KV-cache prefix. That is the same distinction DSH draws between a
   prompt section and dynamic runtime context
   (docs/subsystems/system-prompt.md), and Rigma reached it independently in
   docs/superpowers/specs/2026-07-20-context-policy-design.md.

2. It does not decide WHICH tools a session gets. That is the tool surface's
   job (`tools.tool_specs`). A section may only describe what the surface has
   already promised — see `requires`, which is what keeps the two honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


class PromptError(ValueError):
    """A prompt that cannot be assembled — a duplicate or unknown section."""


@dataclass(frozen=True)
class Section:
    """One named, ordered piece of a system prompt.

    `requires` names the capabilities the section is only true for. A section
    requiring "vision" is omitted for a model that has none — which is not
    decoration: the vision tools are withheld by `tools.tool_specs` for exactly
    such a model, and a prompt that tells it to call `view_images` is asking for
    a tool it cannot see.
    """

    name: str
    order: int
    text: str | Callable[[frozenset], str]
    requires: tuple[str, ...] = ()


class Prompt:
    """An ordered registry of sections, assembled for one set of capabilities."""

    def __init__(self, caps: Iterable[str] = ()) -> None:
        self.caps = frozenset(caps)
        self._sections: dict[str, Section] = {}

    def add(self, name: str, order: int,
            text: str | Callable[[frozenset], str],
            requires: Iterable[str] = ()) -> "Prompt":
        """Register a section. A duplicate name is an error, not a silent
        overwrite: two authors reaching for "rules" should find out."""
        if name in self._sections:
            raise PromptError(f"section {name!r} is already registered")
        self._sections[name] = Section(name, order, text, tuple(requires))
        return self

    def names(self) -> list[str]:
        """The section names that will be rendered, in assembly order."""
        return [s.name for s in self._ordered() if self._applies(s)]

    def _applies(self, s: Section) -> bool:
        return set(s.requires) <= self.caps

    def _ordered(self) -> list[Section]:
        # order, then name — so equal orders are stable rather than dict order
        return sorted(self._sections.values(), key=lambda s: (s.order, s.name))

    def render(self) -> str:
        """The assembled prompt, sections joined by a blank line."""
        parts = []
        for s in self._ordered():
            if not self._applies(s):
                continue
            body = s.text(self.caps) if callable(s.text) else s.text
            body = body.strip("\n")
            if body:
                parts.append(body)
        return "\n\n".join(parts)


# --- the autonomous agent's prompt ------------------------------------------
#
# Split out of the single AGENT_SYSTEM_PROMPT that stood in serve.py. The text
# is unchanged; tests/test_prompt_sections.py proves the assembly is byte-for-
# byte what shipped before, so this is a refactor and not a rewrite.

_LOOP_HEAD_A = (
    "Your loop:\n"
    "1. No plan yet? Call `manage_plan(action=\"add\", task=\"…\")` 3-5 times to "
    "break the mission into concrete, verifiable steps.\n"
    "2. Otherwise DO the next pending step with the right tool (read_file, "
    "write_file, run_shell, run_python, find_files, sample_files, ")
_LOOP_HEAD_B = (
    ").\n"
    "   • File tools take ABSOLUTE paths (D:\\Art\\pic.png) — you do NOT need "
    "run_shell to reach a folder outside the workspace.\n"
    "   • NEVER RETYPE A FILENAME. You will get long names wrong (you cannot "
    "reliably reproduce ComfyUI_00428_.png).")
# Even this example list is capability-gated: it is the model's menu, so it may
# only name tools the surface actually advertises. Two things withhold one:
# `view_images` is not offered to a model with no vision, and `web_search` is
# extended-tier, which a focused Run holds back (tools._TIER). Naming either at
# a session that cannot call it is the prompt/tool-surface disagreement this
# module exists to stop.
def _tool_menu(caps: frozenset) -> str:
    names = []
    if "vision" in caps:
        names.append("view_images")
    if "extended" in caps:
        names.append("web_search")
    return "".join(n + ", " for n in names) + "…"
_LOOP_TAIL = (
    "\n"
    "   • NEVER dump a big folder. list_directory summarises large folders; use "
    "`sample_files(path, count)` when you need examples, and `find_files` with a "
    "glob when you need specific ones. Dumping thousands of filenames wastes "
    "the context you need for the actual work.\n"
    "3. After a real step, call `manage_plan(action=\"complete\", id=N)`. Your "
    "progress log is written for you automatically — never narrate it.\n"
    "4. Only when the WHOLE mission is genuinely finished, call "
    "`task_complete(summary=\"…\")` — you will be asked to verify with tools.")

# The remedy for a retyped filename depends on what the model can actually do.
# Vision tools are withheld from a model without vision (`tools.tool_specs`,
# needs="vision"), so naming them at one is naming a tool that is not on the
# wire — the precise prompt/tool-surface disagreement this module exists to
# stop. The vision text is byte-identical to what shipped.
_REMEDY_VISION = (
    " After sample_files, call `view_sample()` — it uses the files you were "
    "just given, by reference. To look at a folder directly use "
    "`view_images(folder=..., count=N)`.")
_REMEDY_TEXT_ONLY = (
    " After sample_files, copy the names it returned EXACTLY as written — do "
    "not retype them from memory — or call `read_file` with the path it gave "
    "you.")


def _loop_text(caps: frozenset) -> str:
    remedy = _REMEDY_VISION if "vision" in caps else _REMEDY_TEXT_ONLY
    return (_LOOP_HEAD_A + _tool_menu(caps) + _LOOP_HEAD_B
            + remedy + _LOOP_TAIL)


def _agent_sections() -> list[tuple[str, int, object, tuple]]:
    """The agent's sections as (name, order, text, requires).

    One copy of every rule: both entry points below read this list, so a rule
    cannot exist in the prompt and be missing from the memory variant.
    """
    return [
        ("agent.identity", 10,
         "You are Rigma's AUTONOMOUS AGENT. You pursue one MISSION over many "
         "steps, entirely on your own, by CALLING TOOLS. Writing prose "
         "accomplishes NOTHING — only tool calls change anything. There is no "
         "human to answer; act.", ()),
        ("agent.every_turn", 20,
         "EVERY TURN you MUST call at least one tool. Never reply with only "
         "text or only thinking. Think briefly, then ACT.", ()),
        ("agent.loop", 30, _loop_text, ()),
        ("agent.orientation", 40,
         "NEVER SPEND TOOL CALLS FINDING YOUR PLACE. Every turn you are "
         "already told how many steps are done, your last logged progress, "
         "and your ONE next step. Do NOT call manage_plan(list), do NOT read "
         "progress.md / core_directive.md to catch up, and do NOT re-list "
         "folders you have already listed. Trust what you are given and spend "
         "your tool calls on the WORK.", ()),
        ("agent.no_promise", 50,
         "NEVER PROMISE AN ACTION INSTEAD OF TAKING IT. If you write \"I will "
         "now write the file\", \"let me check X\", or \"next I'll generate "
         "the prompts\", the tool call for it MUST be in that same response. "
         "Never end a turn on a promise — the promise accomplishes nothing "
         "and the next turn starts over. Either you called a tool this turn, "
         "or the turn was wasted.", ()),
        ("agent.no_fabrication", 60,
         "NEVER INVENT RESULTS. Do not write file contents, directory "
         "listings, counts or outputs you did not actually get back from a "
         "tool. If something is blocked, say what blocked you and try another "
         "route — reporting a blocker honestly is always better than "
         "fabricating a result that looks plausible.", ()),
        ("agent.recovery", 70,
         "If a tool errors, read it and try a different approach. Keep going "
         "until task_complete. Do not stop to ask permission — you already "
         "have it.", ()),
    ]


def _registry(caps: Iterable[str]) -> Prompt:
    p = Prompt(caps)
    for name, order, text, requires in _agent_sections():
        p.add(name, order, text, requires)
    return p


# Pinned last, after every rule, so what earlier runs learned reads as the final
# word rather than interrupting the instructions.
MEMORY_ORDER = 90


def agent_prompt(caps: Iterable[str] = ("vision", "extended")) -> str:
    """The autonomous agent's system prompt.

    `caps=("vision", "extended")` reproduces exactly what shipped before this
    module existed — that is a vision model on the full tool surface, which is
    what every run got then. Drop "vision" and the model is not told to call
    tools it cannot see; drop "extended" and it is not told about the
    extended-tier tools a focused Run holds back.
    """
    return _registry(caps).render()


def agent_prompt_with_memory(memory_block: str = "",
                             caps: Iterable[str] = ("vision", "extended")) -> str:
    """The agent prompt plus what earlier runs learned, as one more section.

    This used to be `AGENT_SYSTEM_PROMPT + "\\n\\n" + block` at the call site,
    so the block's position was an accident of where the concatenation sat
    rather than a decision. Now it is a named section with an order, and an
    empty block adds nothing at all.
    """
    p = _registry(caps)
    block = str(memory_block or "").strip()
    if block:
        p.add("agent.learned_pitfalls", MEMORY_ORDER,
              "What earlier runs on this machine learned — treat as fact:\n"
              + block)
    return p.render()
