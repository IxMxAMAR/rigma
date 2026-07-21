"""run_python heals single-escaped control chars in string literals.

Live owner report 2026-07-21: "a huge number of syntax errors when working
with python". Reproduced: the model writes  print('a\\nb')  in the JSON
`code` argument with ONE backslash. JSON decoding turns that into a REAL
newline, which splits the Python string literal across two lines:

    print('a
    b')

...which is `SyntaxError: unterminated string literal`. The model then sees a
bare traceback, cannot tell that the transport mangled its code, and rewrites
it with the same mistake. On a model that takes minutes per turn that is an
expensive loop, so -- exactly like repair_json_args and edit_file's healing
ladder -- repair it and SAY so.
"""
from rigma import tools as toolkit

NL = chr(10)


def test_the_bug_reproduces_through_the_json_layer():
    """Guards the premise: single-escaped \\n really does arrive as a newline."""
    args, _ = toolkit.repair_json_args(
        '{"code": "print(' + chr(39) + 'a' + chr(92) + 'nb' + chr(39) + ')"}')
    assert args["code"] == "print('a" + NL + "b')"


def test_heal_rejoins_a_split_string_literal():
    broken = "print('a" + NL + "b')"
    healed, n = toolkit.heal_python_escapes(broken)
    assert n == 1
    assert healed == "print('a" + chr(92) + "nb')"
    compile(healed, "<string>", "exec")


def test_heal_leaves_legitimate_multiline_code_alone():
    good = "x = 1" + NL + "y = 2" + NL + "print(x + y)"
    healed, n = toolkit.heal_python_escapes(good)
    assert healed == good and n == 0


def test_heal_leaves_triple_quoted_strings_alone():
    good = 'text = """line one' + NL + 'line two"""' + NL + "print(text)"
    healed, n = toolkit.heal_python_escapes(good)
    assert healed == good and n == 0


def test_heal_fixes_several_split_literals():
    broken = ("a = 'x" + NL + "y'" + NL + "b = 'p" + NL + "q'" + NL
              + "print(a, b)")
    healed, n = toolkit.heal_python_escapes(broken)
    assert n == 2
    compile(healed, "<string>", "exec")


def test_heal_gives_up_on_genuinely_broken_code_without_hanging():
    broken = "def (((:" + NL + "  ???"
    healed, n = toolkit.heal_python_escapes(broken)
    assert n == 0 and healed == broken


def test_run_python_repairs_and_reports(monkeypatch):
    seen = {}

    def fake(cmd, ctx, shell=False, python_src=None, timeout=30):
        seen["code"] = python_src
        return "exit 0 (ok)" + NL + "a" + NL + "b"
    monkeypatch.setattr(toolkit, "_run_subprocess", fake)
    out = toolkit.run_tool("run_python",
                           {"code": "print('a" + NL + "b')"},
                           {"allow_code": True})
    # it ran the REPAIRED source...
    assert seen["code"] == "print('a" + chr(92) + "nb')"
    # ...and told the model, so the next call is written correctly
    assert "repaired" in out.lower()
    assert "exit 0" in out


def test_run_python_reports_a_syntax_error_it_cannot_heal(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not run code that does not compile")
    monkeypatch.setattr(toolkit, "_run_subprocess", boom)
    out = toolkit.run_tool("run_python", {"code": "def ((("},
                           {"allow_code": True})
    assert out.startswith("error:")
    assert "line 1" in out


def test_valid_code_is_untouched(monkeypatch):
    seen = {}

    def fake(cmd, ctx, shell=False, python_src=None, timeout=30):
        seen["code"] = python_src
        return "exit 0 (ok)"
    monkeypatch.setattr(toolkit, "_run_subprocess", fake)
    src = "print('hello')"
    out = toolkit.run_tool("run_python", {"code": src}, {"allow_code": True})
    assert seen["code"] == src
    assert "repaired" not in out.lower()
