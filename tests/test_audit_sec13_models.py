"""13-4: one containment helper for every `models_dir()/<remote name>` join.

Calls the real helper / sink function on a NAME. Nothing is downloaded, deleted
or written outside the pytest temp dir.
"""
import pytest

from rigma import hangar


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("name", [
    "../../evil.gguf",
    "..\\..\\evil.gguf",
    "Q4_K_M/../../evil.gguf",
    "/etc/passwd",
    "//host/share/evil.gguf",
    "\\\\host\\share\\evil.gguf",
    "C:\\Windows\\evil.gguf",
    "D:evil.gguf",                       # drive-relative, no separator
    "",
    "model\x00.gguf",
    "sub/",
])
def test_a_remote_file_name_cannot_escape_the_models_directory(home, name):
    with pytest.raises(hangar.HangarError):
        hangar.model_file_path(name)


@pytest.mark.parametrize("name", ["CON.gguf", "nul.gguf", "sub/com1.gguf",
                                  "LPT1.Q4_K_M.gguf"])
def test_a_reserved_device_name_is_refused(home, name):
    with pytest.raises(hangar.HangarError):
        hangar.model_file_path(name)


def test_a_nested_quant_is_allowed_and_stays_inside(home):
    root = hangar.models_dir().resolve()
    p = hangar.model_file_path("Q4_K_M/model.gguf")
    assert p.resolve().is_relative_to(root)
    assert p.name == "model.gguf"


def test_pull_progress_routes_the_name_through_the_helper(home):
    # the fallback sink in pull_progress builds models_dir()/<file>; a
    # traversing name must be refused there, not stat'd outside the directory
    with pytest.raises(hangar.HangarError):
        hangar.pull_progress("..\\..\\evil.gguf", 10)
    assert hangar.pull_progress("no-such-file.gguf", 10) == 0
