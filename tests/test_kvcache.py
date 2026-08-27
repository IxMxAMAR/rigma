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
    "engine": "b9867",
    "ctx": 122880,
    "cache_type_k": "q5_1",
    "cache_type_v": "q5_1",
    "ngl": 63,
    "n_cpu_moe": 0,
    "spec_type": "draft-mtp",
    "spec_n_max": 1,
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
    ("engine", "b9900"),
    ("ctx", 131072),
    ("cache_type_k", "q4_0"),
    ("cache_type_v", "q4_0"),
    ("ngl", 62),
    ("n_cpu_moe", 18),
    ("spec_type", ""),
    ("spec_n_max", 2),
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
