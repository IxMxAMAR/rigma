"""R3-5: the ReAct rescue's "only a Thought line may precede it" test was
vacuous, so prose-with-an-Action-block still executed.

05-2 anchored the ReAct shape to "the Action block ends the reply" and required
the part before it to match `_REACT_PREFIX`. That pattern was
`^(?:\\s*(?:Thought|Thinking|Reasoning)\\s*:.*)?\\s*$` compiled with `re.S`, so
the `.*` swallowed NEWLINES: any reply that merely STARTED with "Thought:" could
hold arbitrary explanatory prose and still match, and `rescue_tool_call`'s
result is executed by serve.py's loop.
"""
from rigma import tools

CALL = 'Action: run_shell\nAction Input: {"command": "echo pwned"}'


def test_prose_after_a_thought_marker_is_not_a_call():
    reply = ("Thought: The user is asking HOW to delete a folder. Here is the "
             "syntax you would use — do not run it, it is only an example.\n"
             "It takes a -Recurse flag and a path.\n" + CALL)
    assert tools.rescue_tool_call(reply) == (None, None)


def test_a_multi_line_thought_is_not_a_call():
    reply = "Thinking: first I consider the options\nsecond, a caveat\n" + CALL
    assert tools.rescue_tool_call(reply) == (None, None)


def test_a_single_thought_line_is_still_a_call():
    reply = "Thought: I should list the workspace\n" + CALL
    assert tools.rescue_tool_call(reply)[0] == "run_shell"


def test_a_bare_action_block_is_still_a_call():
    assert tools.rescue_tool_call(CALL)[0] == "run_shell"


def test_blank_lines_after_the_thought_are_still_a_call():
    assert tools.rescue_tool_call("Thought: ok\n\n" + CALL)[0] == "run_shell"


def test_prose_with_no_thought_marker_is_not_a_call():
    reply = "Here is how you would do it:\n" + CALL
    assert tools.rescue_tool_call(reply) == (None, None)


def test_trailing_text_after_the_action_block_is_not_a_call():
    assert tools.rescue_tool_call(CALL + "\nThat is all.") == (None, None)
