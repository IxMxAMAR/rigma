"""D1 / S3 — the benchmark prompt must be varied, and it must carry its depth.

The old prompt was ONE repeated sentence::

    "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 8)

Two things follow from that:

* a MoE router sends every token of repetitive text to the same narrow expert
  set. The owner measured this on 2026-08-23: ``n_cpu_moe 18`` was crowned on
  filler, and on varied text ``n_cpu_moe 0`` beat it by 24% — so the sweep can
  pick the wrong placement off this prompt.
* there is no depth dimension. llama.cpp attends only over OCCUPIED cells
  (``get_n_kv`` pads to 256), so a short run at a huge ctx measures nothing
  about that ctx. ``BenchResult`` recorded no depth, so "tok/s at 131K" was
  measured with an almost-empty window and said nothing about a filled one.

These tests use pure functions and fakes only — no engine, no model.
"""

import json

from rigma import bench
from rigma.bench import BenchResult
from rigma.models import ComboFlags, GgufFile, RunPlan


def _legacy_words(prompt_tokens: int) -> int:
    """Word count of the prompt the unmodified module built, for comparison."""
    return len(("The quick brown fox jumps over the lazy dog. "
                * (prompt_tokens // 8)).split())


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


_TIMINGS = {"timings": {"prompt_per_second": 100.0, "predicted_per_second": 50.0}}


def _capture(monkeypatch, sent):
    def _post(url, json=None, timeout=None):
        sent.update(json)
        return _Resp(_TIMINGS)
    monkeypatch.setattr(bench.httpx, "post", _post)


# --- the text builder --------------------------------------------------------

def test_the_same_budget_builds_byte_identical_text():
    assert bench.bench_text(1024) == bench.bench_text(1024)
    assert bench.bench_text(4096) == bench.bench_text(4096)


def test_the_text_is_not_one_repeated_sentence():
    text = bench.bench_text(1024)
    sentences = [s for s in text.split(".") if s.strip()]
    words = text.split()
    # the old filler was ONE sentence repeated ~128 times
    assert len(sentences) > 20, f"only {len(sentences)} sentences"
    assert len(set(sentences)) > 20, "sentences repeat"
    assert len(set(words)) > 60, f"only {len(set(words))} distinct words"


def test_the_text_is_sized_exactly_like_the_legacy_filler():
    """Replacing the filler must not move the prompt SIZE: the old filler was
    ``9 * (n // 8)`` words, so the new builder is too. Equality, not a
    tolerance — a tolerance here would hide an incidental change of size riding
    along with the intended change of text."""
    for n in (8, 16, 64, 512, 1024, 2048, 4096, 131072):
        assert len(bench.bench_text(n).split()) == _legacy_words(n), n


def test_different_budgets_build_different_text():
    short, long = bench.bench_text(512), bench.bench_text(1024)
    assert short != long[:len(short)]


def test_a_sub_eight_budget_is_empty_like_the_legacy_filler():
    # the old code did `"..." * (n // 8)`, which is "" for n < 8
    for n in (0, 1, 7):
        assert bench.bench_text(n) == ""


# --- the default path is unchanged -------------------------------------------

def test_no_depth_keeps_the_pre_depth_prompt_size():
    """The default path must send the SAME size prompt the old filler did — the
    variety change is isolated from the size change. The old formula produced
    9 words per 8-token sentence."""
    assert len(bench.bench_text(2048).split()) == _legacy_words(2048)


def test_run_bench_default_sends_no_fill_and_records_no_depth(monkeypatch):
    sent = {}
    _capture(monkeypatch, sent)
    r = bench.run_bench(11500, prompt_tokens=2048, gen_tokens=8)
    content = sent["messages"][0]["content"]
    # exactly the legacy prompt: 9*(2048//8) filler words + the 4-word
    # "Summarize in one sentence." instruction the old code appended
    assert len(content.split()) == _legacy_words(2048) + 4
    assert r.prompt_tokens == 2048
    assert r.depth is None
    assert r.ctx is None
    assert r.filler == bench.FILLER_GENERATION


# --- the depth dimension -----------------------------------------------------

def test_depth_fills_the_window_and_is_recorded(monkeypatch):
    sent = {}
    _capture(monkeypatch, sent)
    r = bench.run_bench(11500, prompt_tokens=512, gen_tokens=8,
                        depth=4096, ctx=131072)
    words = len(sent["messages"][0]["content"].split())
    assert words == _legacy_words(4096) + 4, words
    assert words >= 4096, f"window only {words} deep, asked for 4096"
    assert r.depth == 4096
    assert r.ctx == 131072
    # the occupancy that was requested is what gets recorded
    assert r.prompt_tokens == 4096
    assert r.filler == bench.FILLER_GENERATION


def test_depth_never_shrinks_a_larger_prompt(monkeypatch):
    sent = {}
    _capture(monkeypatch, sent)
    r = bench.run_bench(11500, prompt_tokens=2048, gen_tokens=8, depth=256)
    assert len(sent["messages"][0]["content"].split()) == _legacy_words(2048) + 4
    assert r.prompt_tokens == 2048
    assert r.depth == 256


# --- depth round-trips through the result store ------------------------------

def test_depth_round_trips_through_the_calibration_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    r = BenchResult(pp_tps=100, tg_tps=50, prompt_tokens=4096,
                    gen_tokens=8, depth=4096, ctx=131072,
                    filler=bench.FILLER_GENERATION)
    bench.save_calibration("m:q:vulkan", r.as_measured())
    entry = bench.load_calibration()["m:q:vulkan"]
    assert bench.measured_depth(entry) == 4096
    assert entry["measured"]["ctx"] == 131072
    assert bench.measured_filler(entry) == bench.FILLER_GENERATION


def test_as_measured_omits_an_unknown_depth_and_ctx():
    """A missing key and a null must not be two spellings of the same fact: a
    caller doing ``"depth" in measured`` has to be able to tell "not recorded"
    from a recorded value."""
    d = BenchResult(pp_tps=1, tg_tps=2, prompt_tokens=8, gen_tokens=8,
                    filler=bench.FILLER_GENERATION).as_measured()
    assert "depth" not in d and "ctx" not in d
    assert d["filler"] == bench.FILLER_GENERATION
    d2 = BenchResult(pp_tps=1, tg_tps=2, prompt_tokens=8, gen_tokens=8,
                     depth=4096, ctx=131072).as_measured()
    assert d2["depth"] == 4096 and d2["ctx"] == 131072


def test_a_legacy_entry_without_depth_reads_as_unknown(tmp_path, monkeypatch):
    """An entry written before depth existed has no ``depth`` key. That is
    UNKNOWN, not 0 — 0 would claim the window was empty, which the old entries
    cannot support. It also predates the filler marker, so its number came from
    the old repeated sentence and must not be compared with a new one."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    bench.save_calibration("m:q:vulkan",
                           {"tg_tps": 50, "pp_tps": 100,
                            "prompt_tokens": 2048, "gen_tokens": 8})
    entry = bench.load_calibration()["m:q:vulkan"]
    assert "depth" not in entry["measured"]
    assert bench.measured_depth(entry) is None
    # and no filler marker either: it was measured with the old repeated
    # sentence, so its number is not comparable with a new one
    assert "filler" not in entry["measured"]
    assert bench.measured_filler(entry) is None


def test_measured_filler_reads_the_generator_marker():
    assert bench.measured_filler(
        {"measured": {"filler": bench.FILLER_GENERATION}}) == bench.FILLER_GENERATION
    assert bench.measured_filler({"measured": {"filler": ""}}) is None
    assert bench.measured_filler({"measured": {"filler": None}}) is None
    assert bench.measured_filler({"measured": {}}) is None
    assert bench.measured_filler({}) is None


def test_a_stored_null_or_zero_depth_is_unknown_not_zero():
    assert bench.measured_depth({"measured": {"depth": None}}) is None
    assert bench.measured_depth({"measured": {"depth": 0}}) is None
    assert bench.measured_depth({"measured": {}}) is None
    assert bench.measured_depth({}) is None


# --- the sweep surfaces it ---------------------------------------------------

def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=131072, **fl),
                   origin="calculator")


def test_run_sweep_records_and_logs_depth(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    class _Srv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _Srv())
    monkeypatch.setattr(bench, "run_bench",
                        lambda port, **k: BenchResult(
                            pp_tps=120, tg_tps=70,
                            prompt_tokens=k.get("prompt_tokens"),
                            gen_tokens=8, depth=k.get("depth"),
                            ctx=k.get("ctx")))
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): [("fa-off", {"flash_attn": "off"})])

    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601, prompt_tokens=512, gen_tokens=8,
                           depth=4096)
    assert rows[0]["depth"] == 4096
    assert rows[0]["filler"] == bench.FILLER_GENERATION
    logged = [json.loads(x) for x in
              (tmp_path / "logs" / "bench-rows.jsonl").read_text(
                  encoding="utf-8").splitlines() if x]
    assert logged[0]["depth"] == 4096
    assert logged[0]["filler"] == bench.FILLER_GENERATION
    entry = next(iter(bench.load_calibration().values()))
    assert bench.measured_depth(entry) == 4096
    assert bench.measured_filler(entry) == bench.FILLER_GENERATION


def test_run_sweep_default_records_unknown_depth(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))

    class _Srv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _Srv())
    monkeypatch.setattr(bench, "run_bench",
                        lambda port, **k: BenchResult(
                            pp_tps=120, tg_tps=70, prompt_tokens=8,
                            gen_tokens=8, depth=k.get("depth")))
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): [("fa-off", {"flash_attn": "off"})])
    rows = bench.run_sweep(_plan(), tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    # a depth-less row must OMIT the key, not carry a null a reader could take
    # for a recorded value
    assert "depth" not in rows[0]
    logged = [json.loads(x) for x in
              (tmp_path / "logs" / "bench-rows.jsonl").read_text(
                  encoding="utf-8").splitlines() if x]
    assert "depth" not in logged[0]
    entry = next(iter(bench.load_calibration().values()))
    assert "depth" not in entry["measured"]
    assert bench.measured_depth(entry) is None
    # the generator marker is always written by the new path
    assert bench.measured_filler(entry) == bench.FILLER_GENERATION
