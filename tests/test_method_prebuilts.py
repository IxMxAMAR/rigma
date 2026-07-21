"""Every built-in ships real rules, macros and workflows (spec §10)."""
import pytest

from rigma import macros, method_schema as ms, methods


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_every_builtin_validates():
    names = methods.tool_names()
    for m in methods.builtins():
        assert ms.validate(m, names) == [], m["id"]


def test_every_builtin_has_a_standing_rule_and_macros():
    for m in methods.builtins():
        assert any(r["kind"] == "standing" for r in m["rules"]), m["id"]
        assert len(m["macros"]) >= 3, m["id"]


def test_book_ships_the_finish_chapter_macro():
    book = methods.get("book")
    ids = {x["id"] for x in book["macros"]}
    assert {"finish_chapter", "recap_so_far"} <= ids
    fc = next(x for x in book["macros"] if x["id"] == "finish_chapter")
    kinds = [s["kind"] for s in fc["steps"]]
    assert kinds == ["prompt", "note", "new_chat"]
    assert fc["steps"][0]["to"] == "aux"       # no transcript pollution


def test_book_has_a_chapter_trigger_rule():
    book = methods.get("book")
    trig = [r for r in book["rules"] if r["kind"] == "trigger"]
    assert trig and trig[0]["on"]["event"] == "tool_ran"
    assert trig[0]["on"]["tool"] == "write_file"


def test_recap_is_read_only_but_finish_chapter_is_not():
    book = methods.get("book")
    by_id = {x["id"]: x for x in book["macros"]}
    assert macros.is_effectful(by_id["recap_so_far"]["steps"]) is False
    assert macros.is_effectful(by_id["finish_chapter"]["steps"]) is True


def test_no_organize_macro_declares_a_destructive_tool_step():
    """Organize acts through the model, never through a hardcoded move: the
    paths only exist once it has explored. So the safety property to assert
    is that no step NAMES a mutating tool -- not that the macro is inert."""
    org = methods.get("organize")
    for mac in org["macros"]:
        for step in mac["steps"]:
            if step["kind"] == "tool":
                assert step["name"] in ms.SAFE_TOOLS, mac["id"]


def test_organize_macros_confirm_when_the_session_can_run_code():
    org = methods.get("organize")
    by_id = {x["id"]: x for x in org["macros"]}
    for mid in ("preview_plan", "execute_plan"):
        steps = by_id[mid]["steps"]
        # a chat prompt can reach move_files on the model's own initiative
        assert macros.is_effectful(steps, allow_code=True) is True
        assert macros.is_effectful(steps, allow_code=False) is False


def test_book_and_research_ship_workflows():
    assert methods.get("book")["workflows"]
    assert methods.get("research")["workflows"]


def test_no_builtin_macro_asks_a_question_it_cannot_answer():
    for m in methods.builtins():
        for mac in m["macros"]:
            for label in macros.asks(mac["steps"]):
                assert label and len(label) < 60, (m["id"], mac["id"])
