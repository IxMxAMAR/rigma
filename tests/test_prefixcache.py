"""Prefix keying, which is what decides whether a conversation resumes from the
right place or from someone else's history.

The failure this guards against is not a crash. Restoring a snapshot taken
under a different configuration produces fluent, plausible text generated from
a state that does not describe the conversation on screen — and nothing
downstream can tell.
"""
import os

from rigma.prefixcache import (PrefixPoint, available, best_match, evict,
                               prefix_keys, should_snapshot, snapshot_name)


FP = "cfg0123456789ab"
CHAT = [
    {"role": "system", "content": "You are a writing assistant."},
    {"role": "user", "content": "Draft chapter one."},
    {"role": "assistant", "content": "Once upon a time..."},
    {"role": "user", "content": "Now chapter two."},
]


def test_one_key_per_boundary_in_order():
    pts = prefix_keys(CHAT, FP)

    assert [p.n_messages for p in pts] == [1, 2, 3, 4]
    assert len({p.key for p in pts}) == 4          # every depth distinct


def test_a_shared_opening_shares_its_keys():
    # Two chats that begin the same way must agree on the keys for the part
    # they share, or a snapshot of the common opening is never reusable.
    other = CHAT[:2] + [{"role": "assistant", "content": "A different reply."}]
    a, b = prefix_keys(CHAT, FP), prefix_keys(other, FP)

    assert [p.key for p in a[:2]] == [p.key for p in b[:2]]
    assert a[2].key != b[2].key                    # and diverge where they do


def test_a_different_configuration_shares_nothing():
    # THE safety property: a snapshot taken at another ctx / quant / cache type
    # must be unreachable, because its KV values mean something else entirely.
    other = prefix_keys(CHAT, "cfgFFFFFFFFFFFF")

    assert not ({p.key for p in prefix_keys(CHAT, FP)}
                & {p.key for p in other})


def test_editing_an_early_message_invalidates_everything_after_it():
    edited = [dict(CHAT[0]), {**CHAT[1], "content": "Draft chapter ONE."},
              *CHAT[2:]]
    a, b = prefix_keys(CHAT, FP), prefix_keys(edited, FP)

    assert a[0].key == b[0].key                    # untouched opening
    assert [p.key for p in a[1:]] != [p.key for p in b[1:]]


def test_the_role_is_part_of_the_key():
    swapped = [{**CHAT[0], "role": "user"}, *CHAT[1:]]

    assert prefix_keys(swapped, FP)[0].key != prefix_keys(CHAT, FP)[0].key


def test_metadata_that_never_reaches_the_model_does_not_fragment_the_cache():
    # ids and timestamps ride on rigma's messages and change constantly. If
    # they entered the key, every key would be unique and nothing would ever
    # hit.
    noisy = [{**m, "id": f"msg-{i}", "ts": 12345 + i}
             for i, m in enumerate(CHAT)]

    assert [p.key for p in prefix_keys(noisy, FP)] \
        == [p.key for p in prefix_keys(CHAT, FP)]


def test_vision_parts_are_read_as_their_text():
    parts = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    plain = [{"role": "user", "content": "hello"}]

    assert prefix_keys(parts, FP)[0].key == prefix_keys(plain, FP)[0].key


def test_best_match_takes_the_deepest_available_snapshot():
    pts = prefix_keys(CHAT, FP)
    # snapshots exist at boundaries 1 and 3; 3 saves strictly more work
    got = best_match(pts, {pts[0].key, pts[2].key})

    assert got is not None and got.n_messages == 3


def test_best_match_returns_nothing_for_a_new_conversation():
    assert best_match(prefix_keys(CHAT, FP), set()) is None
    assert best_match([], {"anything"}) is None


def test_snapshots_are_spaced_by_growth_not_by_turns():
    # Every turn would cost more in write time and disk than the prefill saved.
    first = PrefixPoint(n_messages=4, key="a", approx_tokens=5000)
    soon = PrefixPoint(n_messages=5, key="b", approx_tokens=5500)
    later = PrefixPoint(n_messages=9, key="c", approx_tokens=12000)

    assert should_snapshot(first, None, min_growth=4096) is True
    assert should_snapshot(soon, first, min_growth=4096) is False
    assert should_snapshot(later, first, min_growth=4096) is True


def test_a_short_conversation_is_not_worth_snapshotting():
    tiny = PrefixPoint(n_messages=2, key="a", approx_tokens=200)

    assert should_snapshot(tiny, None, min_growth=4096) is False


def test_a_shallower_point_never_replaces_a_deeper_snapshot():
    deep = PrefixPoint(n_messages=20, key="a", approx_tokens=40000)
    shallow = PrefixPoint(n_messages=3, key="b", approx_tokens=50000)

    assert should_snapshot(shallow, deep, min_growth=4096) is False


def test_available_reads_the_keys_on_disk(tmp_path):
    (tmp_path / snapshot_name("aaaa")).write_bytes(b"x")
    (tmp_path / snapshot_name("bbbb")).write_bytes(b"x")
    (tmp_path / "kv-cccc.bin").write_bytes(b"x")       # the other cache's files

    assert available(tmp_path) == {"aaaa", "bbbb"}


def test_available_on_a_missing_directory_is_empty(tmp_path):
    assert available(tmp_path / "nope") == set()


def test_eviction_keeps_the_most_recently_used_within_budget(tmp_path):
    for i, key in enumerate(["old", "mid", "new"]):
        p = tmp_path / snapshot_name(key)
        p.write_bytes(b"x" * 100)
        (tmp_path / f"pfx-{key}.json").write_text("{}")
        os.utime(p, (1_000_000 + i * 60, 1_000_000 + i * 60))

    removed = evict(tmp_path, budget_bytes=250)       # room for two

    assert removed == [snapshot_name("old")]
    assert available(tmp_path) == {"mid", "new"}
    assert not (tmp_path / "pfx-old.json").exists()   # metadata follows


def test_eviction_within_budget_removes_nothing(tmp_path):
    (tmp_path / snapshot_name("a")).write_bytes(b"x" * 10)

    assert evict(tmp_path, budget_bytes=10_000) == []
