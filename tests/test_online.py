import json
import time

import numpy as np

from mci import online


def ev(t, track="a", ctx="pl1", ts=None):
    return {"type": t, "track_id": track, "context": ctx, "ts": ts or time.time()}


def test_events_flow_from_stream_to_counters(r):
    online.ensure_group(r)
    online.publish(r, [ev("complete"), ev("skip"), ev("save"), ev("skip", track="b")])
    stats = online.RedisStats(r)
    assert online.consume_once(r, stats, "learner-1", block_ms=10) == 4
    n, s = stats.get("pl1", ["a", "b", "never"])
    assert n.tolist() == [3, 1, 0] and s.tolist() == [2, 0, 0]
    assert online.streams_last_hour(r, "a") == 2
    # acknowledged: nothing pending, nothing re-applied
    assert r.xpending(online.STREAM, online.GROUP)["pending"] == 0
    assert online.consume_once(r, stats, "learner-1", block_ms=10) == 0


def test_events_of_a_crashed_learner_are_reclaimed(r):
    online.ensure_group(r)
    online.publish(r, [ev("complete"), ev("complete")])
    r.xreadgroup(online.GROUP, "dead-pod", {online.STREAM: ">"}, count=10)  # read, never acked
    stats = online.RedisStats(r)
    assert online.consume_once(r, stats, "live-pod", block_ms=10, claim_idle_ms=0) == 2
    assert stats.get("pl1", ["a"])[0].tolist() == [2]


def test_malformed_events_are_dropped_not_fatal(r):
    online.ensure_group(r)
    r.xadd(online.STREAM, {"e": json.dumps({"type": "explode", "track_id": "a", "context": "pl1"})})
    r.xadd(online.STREAM, {"e": "not json"})
    online.publish(r, [ev("complete")])
    stats = online.RedisStats(r)
    assert online.consume_once(r, stats, "l", block_ms=10) == 1
    assert r.xpending(online.STREAM, online.GROUP)["pending"] == 0


def test_two_learners_share_the_stream_without_double_counting(r):
    online.ensure_group(r)
    online.publish(r, [ev("complete", track=f"t{i % 5}") for i in range(200)])
    stats = online.RedisStats(r)
    a = online.consume_once(r, stats, "pod-a", count=120, block_ms=10)
    b = online.consume_once(r, stats, "pod-b", count=120, block_ms=10)
    assert a + b == 200
    assert stats.get("pl1", [f"t{i}" for i in range(5)])[0].sum() == 200


def test_thompson_prior_then_evidence():
    stats = online.MemoryStats()
    pol = online.ThompsonPolicy(stats, strength=20, rng=np.random.default_rng(0))
    tracks = ["good", "hyped"]
    prior = np.array([0.3, 0.7])  # the model prefers "hyped"
    assert pol.mean("c", tracks, prior)[1] > pol.mean("c", tracks, prior)[0]
    stats.add("c", "good", 200, 160)  # listeners disagree
    stats.add("c", "hyped", 200, 40)
    assert pol.mean("c", tracks, prior)[0] > pol.mean("c", tracks, prior)[1]
    picks = pol.choose_each("c", tracks, prior, 500)
    assert (picks == 0).mean() > 0.95


def test_event_validation():
    import pytest

    with pytest.raises(ValueError):
        online.validate({"type": "like", "track_id": "a", "context": "c"})
    with pytest.raises(ValueError):
        online.validate({"type": "skip", "track_id": "", "context": "c"})
