"""The fingerprint is the safety mechanism, so it is what gets tested.

A KV cache restored under a configuration it was not taken under does not
error — the model generates fluent, subtly wrong text from a history that never
happened. Nothing downstream catches that. The only thing standing between the
user and it is that the filename encodes the configuration, so a mismatch asks
for a file that does not exist.
"""
import json

import pytest

from rigma import kvcache


BASE = {
    "model": "qwen38-ara-v5",
    "quant": "Q3_K_M [mtp]",
    "gguf": "RVN-Q3_K_M-mtp.gguf",
    "backend": "vulkan",
    "engine": "C:/engines/b9867/vulkan/llama-server.exe",
    "engine_version": "b9867+152d337fa",
    "ctx": 122880,
    "cache_type_k": "q5_1",
    "cache_type_v": "q5_1",
    "ngl": 63,
    "n_cpu_moe": 0,
    "spec_type": "draft-mtp",
    "spec_n_max": 1,
    "spec_draft": "",
    "spec_conf_min": 0.0,
    "spec_p_min": 0.0,
    "flash_attn": "on",
}


def test_the_same_configuration_always_names_the_same_cache():
    assert kvcache.fingerprint(BASE) == kvcache.fingerprint(dict(BASE))


def test_key_order_does_not_change_the_name():
    shuffled = {k: BASE[k] for k in reversed(list(BASE))}
    assert kvcache.fingerprint(shuffled) == kvcache.fingerprint(BASE)


@pytest.mark.parametrize("field,value", [
    ("model", "ornith-1.5-35b-a3b-heretic-t62"),
    ("quant", "Q3_K_L [mtp]"),
    ("gguf", "RVN-Q3_K_M.gguf"),
    ("backend", "rocm"),
    ("engine", "C:/engines/b9900/vulkan/llama-server.exe"),
    ("engine_version", "b10709+9a9394a89"),
    ("ctx", 131072),
    ("cache_type_k", "q4_0"),
    ("cache_type_v", "q4_0"),
    ("ngl", 62),
    ("n_cpu_moe", 18),
    ("spec_type", ""),
    ("spec_n_max", 2),
    # The draft ARTEFACT and DSpark's confidence cut change what the cache was
    # built against even when the KV layout is identical, so each must move the
    # name — a snapshot restored across a draft swap describes a run that never
    # happened.
    ("spec_draft", "C:/models/dflash2-draft.gguf"),
    ("spec_conf_min", 0.4),
    ("spec_p_min", 0.2),
    ("flash_attn", "off"),
])
def test_every_field_that_invalidates_a_cache_changes_the_name(field, value):
    # Each of these changes what the KV cache means. If any stopped affecting
    # the hash, a stale cache would become restorable under the new config and
    # the model would answer from a history it never had.
    other = {**BASE, field: value}
    assert other[field] != BASE[field], "test would not prove anything"
    assert kvcache.fingerprint(other) != kvcache.fingerprint(BASE)


def test_absent_is_not_the_same_as_empty():
    # A spec_type that is missing and one that is "" describe different
    # launches; collapsing them would let one restore the other's cache.
    missing = {k: v for k, v in BASE.items() if k != "spec_type"}
    empty = {**BASE, "spec_type": ""}
    assert kvcache.fingerprint(missing) != kvcache.fingerprint(empty)


def test_fields_outside_the_list_do_not_change_the_name():
    # use_case does not alter the KV cache, so it must not fragment it.
    assert kvcache.fingerprint({**BASE, "use_case": "creative"}) \
        == kvcache.fingerprint(BASE)


def test_restore_refuses_when_no_cache_matches(tmp_path):
    # No file, no HTTP call, no error — this is the normal case after any
    # config change, and it must be silent rather than noisy.
    restored, note = kvcache.restore(1, tmp_path, kvcache.fingerprint(BASE))
    assert restored is False
    assert note is None


def test_restore_of_a_cache_from_another_config_is_not_even_attempted(tmp_path):
    # A cache exists, but for a DIFFERENT context. The lookup is by name, so
    # there is nothing to compare and nothing to get wrong: the file this
    # config would need simply is not there.
    other = {**BASE, "ctx": 32768}
    (tmp_path / kvcache.cache_name(kvcache.fingerprint(other))).write_bytes(b"x")
    restored, note = kvcache.restore(1, tmp_path, kvcache.fingerprint(BASE))
    assert restored is False
    assert note is None


def test_prune_keeps_the_newest_and_removes_their_metadata(tmp_path):
    import os
    names = []
    for i in range(5):
        fp = kvcache.fingerprint({**BASE, "ctx": 1024 * (i + 1)})
        blob = tmp_path / kvcache.cache_name(fp)
        blob.write_bytes(b"x" * 16)
        (tmp_path / f"kv-{fp}.json").write_text(json.dumps({"ctx": i}))
        os.utime(blob, (1_000_000 + i * 60, 1_000_000 + i * 60))
        names.append(blob.name)

    removed = kvcache.prune(tmp_path, keep=2)

    assert sorted(removed) == sorted(names[:3])          # oldest three
    assert {p.name for p in tmp_path.glob("kv-*.bin")} == set(names[3:])
    # metadata follows its blob, or the directory fills with orphans
    assert len(list(tmp_path.glob("kv-*.json"))) == 2


def test_prune_on_a_missing_directory_is_not_an_error(tmp_path):
    assert kvcache.prune(tmp_path / "nope") == []


# --- VLLM-1: the fingerprint must identify the engine RUNTIME, not a path -----
#
# `engine` is the BINARY PATH, and a path is not a build: replacing the binary
# in place (which is exactly how a hand-installed fork arrives) left the name
# unchanged, so a cache taken under the old build was restored under the new
# one. The fingerprint now also carries the measured engine identity, from the
# SAME source the rest of the code uses for "which build is this"
# (`server_ops.engine_version`, via `resolve._engine_now`/`bench._engine_version`),
# so a build change invalidates the cache exactly when calibration would go stale.


def _plan(backend="vulkan"):
    """The plan attributes `config_of` reads, without building a whole RunPlan."""
    from types import SimpleNamespace
    flags = SimpleNamespace(ctx=122880, cache_type_k="q5_1", cache_type_v="q5_1",
                            ngl=63, n_cpu_moe=0, spec_type="draft-mtp",
                            spec_n_max=1, spec_draft="", spec_conf_min=0.0,
                            spec_p_min=0.0, flash_attn="on")
    return SimpleNamespace(model_slug="qwen38-ara-v5",
                           gguf=SimpleNamespace(quant="Q3_K_M [mtp]",
                                                file="RVN-Q3_K_M-mtp.gguf"),
                           backend=backend, flags=flags)


def test_config_of_carries_the_measured_engine_identity(monkeypatch):
    # The fix is only real if the field is POPULATED from the shared source; a
    # new FINGERPRINT_FIELDS entry that nobody fills in would hash to the
    # "unknown" sentinel forever and detect nothing.
    from rigma import server_ops
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b9867+152d337fa")
    cfg = kvcache.config_of(_plan(), "C:/engines/b9867/vulkan/llama-server.exe")
    assert cfg["engine_version"] == "b9867+152d337fa"


def test_a_different_engine_build_changes_the_launch_fingerprint(monkeypatch):
    from rigma import server_ops
    rp = _plan()
    exe = "C:/engines/b9867/vulkan/llama-server.exe"
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b9867+152d337fa")
    same_build = kvcache.launch_fingerprint(rp, exe)
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b10709+9a9394a89")
    fork = kvcache.launch_fingerprint(rp, exe)
    assert same_build != fork


def test_a_cache_from_another_engine_version_is_refused(tmp_path, monkeypatch):
    # Runtime A wrote the cache; runtime B (a different build string) must not
    # be handed it. The refusal is by name, so the HTTP restore is never even
    # attempted — which is what the failing monkeypatch pins.
    a = {**BASE, "engine_version": "b9867+152d337fa"}
    b = {**BASE, "engine_version": "b10709+9a9394a89"}
    assert kvcache.fingerprint(a) != kvcache.fingerprint(b)
    (tmp_path / kvcache.cache_name(kvcache.fingerprint(a))).write_bytes(b"x")
    monkeypatch.setattr(kvcache, "slot_action",
                        lambda *a, **k: pytest.fail(
                            "runtime B was handed runtime A's cache"))
    restored, note = kvcache.restore(1, tmp_path, kvcache.fingerprint(b))
    assert restored is False
    assert note is None


def test_the_same_engine_version_still_matches(tmp_path, monkeypatch):
    # No false invalidation: the identical configuration still offers its cache
    # to the engine. A fingerprint that changed on every call would silently
    # disable restore, which is the failure the empty-fingerprint bug caused.
    cfg = {**BASE, "engine_version": "b9867+152d337fa"}
    fp = kvcache.fingerprint(cfg)
    assert fp == kvcache.fingerprint(dict(cfg))
    (tmp_path / kvcache.cache_name(fp)).write_bytes(b"x")
    offered = []
    monkeypatch.setattr(kvcache, "slot_action",
                        lambda *a, **k: offered.append(a) or None)
    restored, note = kvcache.restore(1, tmp_path, fp)
    assert restored is True
    assert note is None
    assert offered, "the matching cache must actually be offered to the engine"


def test_a_legacy_entry_without_the_engine_version_is_invalid_not_a_match(
        tmp_path, monkeypatch):
    # An on-disk cache written before this field existed. Its name was hashed
    # over the OLD field list, so the new fingerprint asks for a different file:
    # "unknown" reads as invalid, never as a match, and nothing raises. The cost
    # is a re-prefill — the safe direction.
    legacy = {k: v for k, v in BASE.items() if k != "engine_version"}
    modern = {**BASE, "engine_version": "b9867+152d337fa"}
    assert kvcache.fingerprint(legacy) != kvcache.fingerprint(modern)
    # And a genuinely unknown identity ("") is its own value, distinct from the
    # absent sentinel — neither can collide with a real build string.
    assert kvcache.fingerprint(legacy) \
        != kvcache.fingerprint({**modern, "engine_version": ""})
    (tmp_path / kvcache.cache_name(kvcache.fingerprint(legacy))).write_bytes(b"x")
    monkeypatch.setattr(kvcache, "slot_action",
                        lambda *a, **k: pytest.fail(
                            "a legacy cache was silently accepted"))
    restored, note = kvcache.restore(1, tmp_path, kvcache.fingerprint(modern))
    assert restored is False
    assert note is None


# --- A13b: a REGISTERED engine's identity is its own binary, not the pin's ----
#
# `engine_binary_for` prefers a registered engine at an arbitrary path (this
# machine has prism-b10743-vulkan and prism-b10743-hip). The fingerprint recorded
# that path but the version of the PIN — `engine_identity(backend)` looks under
# ~/.rigma/engines/<manifest version>/<backend> — so an in-place swap of a
# registered engine kept the same name and a stale cache was reused. A path is
# not a build, and `launch_fingerprint` already receives the exe that will run.

LEGACY = ("llama-server.exe : version: 9867 (152d337fa)\n"
          "built with Clang 20.1.8 for Windows x86_64\n")
# The modern scheme, from the hand-installed PrismML fork. Its identity differs
# from the pin's in BOTH the build number and the commit.
MODERN = ("llama-server.exe : version: 0.2.0-dev (build 10743, commit 9a9394a89)\n"
          "built with Clang 21.0.0 for Windows AMD64\n")


def _registered_exe(tmp_path):
    """A registered engine at its own path — the shape `engine_binary_for`
    returns for `prism-b10743-vulkan`, not the pinned directory layout."""
    p = tmp_path / "prism-b10743-vulkan" / "llama-server.exe"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    return p


def test_a_registered_engine_is_identified_by_its_own_binary(tmp_path,
                                                             monkeypatch):
    """The pin answers for the backend; the launched file is what will run. A
    fingerprint that records the pin's version next to a registered path cannot
    see that engine change."""
    from rigma import engine_build, server_ops
    exe = _registered_exe(tmp_path)
    engine_build._BUILD_CACHE.clear()
    monkeypatch.setattr(engine_build, "read_build",
                        lambda *a, **k: engine_build.parse_version(MODERN))
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b9867+152d337fa")
    rp = _plan()
    got = kvcache.launch_fingerprint(rp, exe)
    measured = engine_build.parse_version(MODERN).identity
    assert got == kvcache.fingerprint(
        kvcache.config_of(rp, str(exe), engine_version=measured))
    assert got != kvcache.fingerprint(
        kvcache.config_of(rp, str(exe), engine_version="b9867+152d337fa")), \
        "the registered engine was identified as the pin"


def test_swapping_a_registered_engine_in_place_changes_the_fingerprint(
        tmp_path, monkeypatch):
    """The failure this closes: the same registered path, a different build, and
    a cache that would otherwise be offered to an engine that never wrote it."""
    from rigma import engine_build, server_ops
    exe = _registered_exe(tmp_path)
    state = {"modern": False}
    engine_build._BUILD_CACHE.clear()
    monkeypatch.setattr(engine_build, "read_build",
                        lambda *a, **k: engine_build.parse_version(
                            MODERN if state["modern"] else LEGACY))
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b9867+152d337fa")
    rp = _plan()
    before = kvcache.launch_fingerprint(rp, exe)
    state["modern"] = True
    exe.write_bytes(b"a much longer replacement binary")   # same path, new build
    after = kvcache.launch_fingerprint(rp, exe)
    assert before != after


def test_a_pinned_launch_keeps_the_fingerprint_it_had(tmp_path, monkeypatch):
    """The fix must not move the hash for an ordinary pinned launch: every
    existing cache would be orphaned — a four-minute re-prefill — for a change
    that has nothing to do with it. Here the pin's own directory is measured, so
    the identity is the same one `engine_identity(backend)` returns."""
    from rigma import engine_build, runtime
    home = tmp_path / "home"
    exe = home / "engines" / "b9867" / "vulkan" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"x")
    engine_build._BUILD_CACHE.clear()
    monkeypatch.setattr(runtime, "rigma_home", lambda: home)
    monkeypatch.setattr(runtime, "_engines_manifest", lambda: {"version": "b9867"})
    monkeypatch.setattr(engine_build, "read_build",
                        lambda *a, **k: engine_build.parse_version(LEGACY))
    rp = _plan()
    # what the fingerprint was BEFORE this change: engine_identity(backend)
    old = kvcache.fingerprint(kvcache.config_of(rp, str(exe)))
    assert old == kvcache.launch_fingerprint(rp, exe)


# --- DR7: an UNMEASURABLE engine is still identified by its own file ----------
#
# A13b keyed on the launched binary's measured identity, but only while that
# binary answered `--version`. When it did not, the identity fell back to
# `engine_identity(backend)` — the PIN directory's identity — so two
# unmeasurable builds swapped at one registered path hashed alike and the old
# build's cache was restored under the new one. The fallback now folds in the
# same (size, nanosecond mtime) key the build memo is built on (A13d).


def _unmeasurable(monkeypatch):
    """Make every `--version` fail, as a timeout or a broken runtime does."""
    from rigma import engine_build, server_ops
    engine_build._BUILD_CACHE.clear()
    monkeypatch.setattr(engine_build, "read_build",
                        lambda *a, **k: engine_build.EngineBuild(
                            reason="could not run llama-server.exe --version: timed out"))
    monkeypatch.setattr(server_ops, "engine_version",
                        lambda backend="": "b9867+152d337fa")


def test_an_unmeasurable_registered_engine_is_not_identified_as_the_pin(
        tmp_path, monkeypatch):
    """The pin answers for the backend; the launched file is what will run, and
    an unreadable file is still its own file."""
    _unmeasurable(monkeypatch)
    exe = _registered_exe(tmp_path)
    ident = kvcache.launched_engine_identity(_plan(), exe)
    assert ident != "b9867+152d337fa", \
        "an unmeasurable registered engine was identified as the pin"
    assert ident.startswith("b9867+152d337fa+file:")


def test_an_unmeasurable_swap_at_one_path_changes_the_fingerprint(
        tmp_path, monkeypatch):
    """DR7: one registered path, two builds, neither measurable. The same file
    must still key the same cache (the memo keeps working); replacing it must
    not — or the old build's cache is restored under the new one."""
    import os
    _unmeasurable(monkeypatch)
    exe = _registered_exe(tmp_path)
    rp = _plan()
    before = kvcache.launch_fingerprint(rp, exe)
    assert before == kvcache.launch_fingerprint(rp, exe), \
        "the same file must key the same cache, or restore is silently disabled"
    # A SAME-SIZE replacement written later in the same second: size alone would
    # miss it, so this also pins that the nanosecond mtime is what is folded in.
    exe.write_bytes(b"y")
    st = exe.stat()
    os.utime(exe, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    after = kvcache.launch_fingerprint(rp, exe)
    assert before != after, \
        "an unmeasurable build swapped in place reused the old build's cache"
