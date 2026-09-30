"""Mission compiler: prose in, a systematic spec the SERVER can verify out."""
from rigma import mission


RAW = ("Go through images in D:\Good Stuff and D:\DevArt, extract my taste, "
       "write a Core Directive doc, then generate 100 prompts in batches of 25.")


def test_parse_spec_accepts_a_good_compile():
    text = '''```json
    {"objective": "analyse art and write prompts",
     "deliverables": [{"path": "D:\\out\\core.md", "description": "directive"}],
     "constraints": ["do not modify originals"],
     "steps": [
       {"id": 1, "description": "sample images", "artifact": "",
        "verification": {"type": "none"}},
       {"id": 2, "description": "write directive",
        "artifact": "D:\\out\\core.md",
        "verification": {"type": "file_min_size", "value": 500}}]}
    ```'''
    spec = mission.parse_spec(text)
    assert spec and spec["compiled"] is True
    assert len(spec["steps"]) == 2
    assert spec["steps"][1]["artifact"].endswith("core.md")
    assert spec["steps"][1]["verification"]["value"] == 500


def test_parse_spec_rejects_junk_so_we_can_fall_back():
    assert mission.parse_spec("sorry, I cannot do that") is None
    assert mission.parse_spec('{"objective": "x"}') is None      # no steps
    assert mission.parse_spec('{"steps": []}') is None


def test_fallback_never_blocks_a_run():
    spec = mission.fallback_spec(RAW)
    assert spec["compiled"] is False
    assert len(spec["steps"]) == 1 and spec["steps"][0]["description"]


def test_spec_block_is_systematic_and_lists_steps_in_order():
    spec = mission.parse_spec(
        '{"objective": "O", "deliverables": [{"path": "P", "description": "D"}],'
        ' "constraints": ["C"], "steps": ['
        '{"id": 1, "description": "one", "artifact": "A1",'
        ' "verification": {"type": "file_min_size", "value": 10}},'
        '{"id": 2, "description": "two", "artifact": "A2",'
        ' "verification": {"type": "none"}}]}')
    block = mission.spec_block(spec, RAW)
    assert "OBJECTIVE: O" in block and "DELIVERABLES" in block
    assert "CONSTRAINTS" in block and "STEPS" in block
    assert block.index("1. one") < block.index("2. two")   # order preserved
    assert "-> writes: A1" in block
    # an uncompiled spec falls back to the raw prose rather than an empty block
    assert mission.spec_block(mission.fallback_spec(RAW), RAW) == RAW


def test_verify_step_checks_the_artifact_on_disk(tmp_path):
    small = tmp_path / "out.txt"
    small.write_text("tiny", encoding="utf-8")
    step = {"id": 1, "artifact": str(small),
            "verification": {"type": "file_min_size", "value": 500}}
    ok, why = mission.verify_step(step)
    assert not ok and "only 4 bytes" in why
    small.write_text("x" * 900, encoding="utf-8")
    assert mission.verify_step(step)[0] is True
    # a missing file is never "done", whatever the model claimed
    gone = {"id": 2, "artifact": str(tmp_path / "nope.txt"),
            "verification": {"type": "file_min_size", "value": 1}}
    ok, why = mission.verify_step(gone)
    assert not ok and "does not exist" in why
    # exploration steps need no artifact
    assert mission.verify_step({"id": 3, "artifact": "",
                                "verification": {"type": "none"}})[0] is True


def _spec_with_artifact(artifact: str) -> dict:
    return {"objective": "o", "deliverables": [], "constraints": [],
            "steps": [{"id": 1, "description": "d", "artifact": artifact,
                       "verification": {"type": "none", "value": 0}}],
            "compiled": True}


def test_anchor_spec_keeps_an_absolute_path_when_its_parent_exists(tmp_path):
    """The normal case must not change: an absolute artifact whose parent is
    really there stays exactly as the compiler wrote it."""
    real = tmp_path / "real"
    real.mkdir()
    spec = _spec_with_artifact(str(real / "a.md"))
    assert mission.anchor_spec(spec)["steps"][0]["artifact"] == str(real / "a.md")


def test_anchor_spec_treats_an_unstatable_parent_as_missing(monkeypatch,
                                                            tmp_path):
    """A BitLocker-locked volume, an offline network share or a device-not-ready
    answer makes Path.exists() RAISE (errno 22 is outside pathlib's ignored set)
    instead of returning False. anchor_spec's contract is "anchor to an EXISTING
    parent", so a parent the OS will not stat must take the SAME basename branch
    as a genuinely absent one — not propagate and abort the whole compile, which
    left `run["spec"] = None` and the run executing with no plan."""
    import pathlib
    locked = tmp_path / "locked"        # really exists; we make stat fail on it
    locked.mkdir()
    missing = tmp_path / "missing"      # genuinely absent
    real_exists = pathlib.Path.exists

    def exists(self):
        if "locked" in str(self):
            raise OSError(22, "BitLocker locked")
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", exists)

    got = mission.anchor_spec(_spec_with_artifact(str(locked / "a.md")))
    want = mission.anchor_spec(_spec_with_artifact(str(missing / "a.md")))
    assert got["steps"][0]["artifact"] == "a.md"       # returned, did not raise
    assert got["steps"][0]["artifact"] == want["steps"][0]["artifact"]


def test_anchor_spec_also_guards_deliverable_paths(monkeypatch, tmp_path):
    """The deliverables collection is anchored by the same loop and the same
    hazard, so it gets the same guard."""
    import pathlib
    locked = tmp_path / "locked"
    locked.mkdir()
    real_exists = pathlib.Path.exists

    def exists(self):
        if "locked" in str(self):
            raise OSError(22, "locked")
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", exists)
    spec = {"objective": "o", "constraints": [], "steps": [],
            "deliverables": [{"path": str(locked / "core.md"),
                              "description": "d"}], "compiled": True}
    got = mission.anchor_spec(spec)
    assert got["deliverables"][0]["path"] == "core.md"


class _Recorder:
    """Captures the payloads compile_mission sends, and replies per-attempt."""
    def __init__(self, replies):
        self.replies, self.payloads = list(replies), []

    async def __call__(self, payload):
        self.payloads.append(payload)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"choices": [{"message": {"content": r}}]}


GOOD = ('{"objective": "o", "deliverables": [], "constraints": [],'
        ' "steps": [{"id": 1, "description": "do it", "artifact": "",'
        ' "verification": {"type": "none"}}]}')


def test_compile_constrains_output_to_json_first():
    import asyncio
    rec = _Recorder([GOOD])
    spec = asyncio.run(mission.compile_mission(RAW, rec))
    assert spec["compiled"] is True
    # the FIRST attempt must ask llama.cpp to grammar-constrain the output —
    # asking a weak model politely for "only JSON" returns markdown instead
    assert rec.payloads[0]["response_format"] == {"type": "json_object"}


def test_compile_retries_unconstrained_if_the_server_rejects_it():
    import asyncio
    rec = _Recorder([RuntimeError("400 response_format unsupported"), GOOD])
    spec = asyncio.run(mission.compile_mission(RAW, rec))
    assert spec["compiled"] is True                    # recovered
    assert len(rec.payloads) == 2
    assert "response_format" not in rec.payloads[1]    # plain retry


def test_compile_falls_back_when_both_attempts_fail():
    import asyncio
    rec = _Recorder(["not json", "still not json"])
    spec = asyncio.run(mission.compile_mission(RAW, rec))
    assert spec["compiled"] is False and len(spec["steps"]) == 1
