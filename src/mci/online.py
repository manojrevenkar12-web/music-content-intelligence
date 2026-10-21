"""Learning from listening as it happens.

The offline model ranks new releases before anyone has heard them. Once they are played, real
listening beats any prediction. This module closes that loop:

    POST /v1/events ──XADD──► Redis Stream `mci:events` ──XREADGROUP──► learner(s) ──► counters
                                                                                       │
    GET /v1/contexts/{c}/promote ◄── Thompson sampling: model prior + counters ◄───────┘

- **Events** are impressions with an outcome: `complete` (listened past 30 s), `save`, or `skip`.
- **Learners** form a consumer group, so pods share the stream. Each event is applied once
  (XACK after the counter update), and events from a crashed learner are reclaimed (XAUTOCLAIM).
- **The policy** keeps a Beta posterior per (context, track). The prior comes from the offline model:
  a track the model likes starts with a higher mean, worth `strength` imaginary plays. Real plays take
  over as they arrive. Thompson sampling balances trying new tracks with playing what works.
- **Live counters** for each track, in minute buckets with a TTL, give "streams in the last hour"
  as a real-time feature.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("mci.online")

STREAM = "mci:events"
GROUP = "learners"
SUCCESS = {"complete", "save"}
OUTCOMES = SUCCESS | {"skip"}


# ------------------------------------------------------------------------------------- stores
class MemoryStats:
    """Same interface as RedisStats; used by the simulator and tests."""

    def __init__(self):
        self.n: dict[tuple[str, str], int] = defaultdict(int)
        self.s: dict[tuple[str, str], int] = defaultdict(int)

    def add(self, context: str, track: str, plays: int, successes: int) -> None:
        self.n[(context, track)] += plays
        self.s[(context, track)] += successes

    def get(self, context: str, tracks: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.n[(context, t)] for t in tracks], float), np.array([self.s[(context, t)] for t in tracks], float)


class RedisStats:
    def __init__(self, r):
        self.r = r

    def add(self, context: str, track: str, plays: int, successes: int) -> None:
        p = self.r.pipeline(transaction=False)
        p.hincrby(f"mci:ctx:{context}", f"{track}:n", plays)
        if successes:
            p.hincrby(f"mci:ctx:{context}", f"{track}:s", successes)
        p.execute()

    def get(self, context: str, tracks: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not tracks:
            return np.zeros(0), np.zeros(0)
        vals = self.r.hmget(f"mci:ctx:{context}", [f"{t}:{k}" for t in tracks for k in ("n", "s")])
        v = np.array([int(x or 0) for x in vals], float).reshape(-1, 2)
        return v[:, 0], v[:, 1]


# ------------------------------------------------------------------------------------- policy
@dataclass
class ThompsonPolicy:
    stats: object
    strength: float = 20.0  # how many plays the model's opinion is worth
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def posterior(self, context: str, tracks: list[str], prior_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, s = self.stats.get(context, tracks)
        m = np.clip(prior_mean, 0.01, 0.99)
        return m * self.strength + s, (1 - m) * self.strength + (n - s)

    def choose(self, context: str, tracks: list[str], prior_mean: np.ndarray, k: int = 1) -> list[int]:
        a, b = self.posterior(context, tracks, prior_mean)
        draw = self.rng.beta(a, b)
        return list(np.argsort(-draw)[:k])

    def choose_each(self, context: str, tracks: list[str], prior_mean: np.ndarray, plays: int) -> np.ndarray:
        """One independent Thompson draw per play (the learner applies outcomes between batches)."""
        a, b = self.posterior(context, tracks, prior_mean)
        return np.argmax(self.rng.beta(a, b, size=(plays, len(tracks))), axis=1)

    def mean(self, context: str, tracks: list[str], prior_mean: np.ndarray) -> np.ndarray:
        a, b = self.posterior(context, tracks, prior_mean)
        return a / (a + b)


def prior_from_scores(scores: np.ndarray, base: float = 0.4, spread: float = 0.12) -> np.ndarray:
    """Model scores for the candidates → prior listen-through rates around the platform average."""
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    return np.clip(base + spread * z, 0.02, 0.98)


# ------------------------------------------------------------------------------------- events
def validate(e: dict) -> dict:
    out = {"type": str(e["type"]), "track_id": str(e["track_id"]), "context": str(e["context"])}
    if out["type"] not in OUTCOMES:
        raise ValueError(f"type must be one of {sorted(OUTCOMES)}")
    if not (0 < len(out["track_id"]) <= 64 and 0 < len(out["context"]) <= 64):
        raise ValueError("track_id and context must be 1-64 characters")
    out["ts"] = float(e.get("ts") or time.time())
    if "user" in e:
        out["user"] = str(e["user"])[:64]
    return out


def publish(r, events: list[dict], maxlen: int = 1_000_000) -> int:
    p = r.pipeline(transaction=False)
    for e in events:
        p.xadd(STREAM, {"e": json.dumps(e)}, maxlen=maxlen, approximate=True)
    p.execute()
    return len(events)


def live_key(track: str, minute: int) -> str:
    return f"mci:live:{track}:{minute}"


def streams_last_hour(r, track: str, now: float | None = None) -> int:
    m = int((now or time.time()) // 60)
    vals = r.mget([live_key(track, m - i) for i in range(60)])
    return sum(int(v or 0) for v in vals)


def apply(r, stats: RedisStats, events: list[dict]) -> None:
    """Aggregate a batch, then one round trip of counter updates."""
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    live: dict[str, int] = defaultdict(int)
    for e in events:
        a = agg[(e["context"], e["track_id"])]
        a[0] += 1
        a[1] += e["type"] in SUCCESS
        if e["type"] in SUCCESS:
            live[live_key(e["track_id"], int(e["ts"] // 60))] += 1
    for (c, t), (n, s) in agg.items():
        stats.add(c, t, n, s)
    if live:
        p = r.pipeline(transaction=False)
        for k, v in live.items():
            p.incrby(k, v)
            p.expire(k, 3700)
        p.execute()


def ensure_group(r) -> None:
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as e:  # BUSYGROUP: another learner created it first
        if "BUSYGROUP" not in str(e):
            raise


def consume_once(r, stats: RedisStats, consumer: str, count: int = 500, block_ms: int = 1000, claim_idle_ms: int = 60_000) -> int:
    """Reclaim events a dead learner left pending, then read new ones. Returns events applied."""
    batch = []
    claimed = r.xautoclaim(STREAM, GROUP, consumer, min_idle_time=claim_idle_ms, start_id="0-0", count=count)
    batch.extend(claimed[1])
    resp = r.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=count, block=block_ms)
    for _, msgs in resp or []:
        batch.extend(msgs)
    if not batch:
        return 0
    events, ids = [], []
    for mid, fields in batch:
        if not fields:  # trimmed away while pending
            ids.append(mid)
            continue
        raw = fields.get(b"e") or fields.get("e")
        try:
            events.append(validate(json.loads(raw)))
        except (ValueError, KeyError, TypeError):
            log.warning("dropping malformed event %s", mid)
        ids.append(mid)
    apply(r, stats, events)
    r.xack(STREAM, GROUP, *ids)
    from mci.serving.metrics import EVENTS_APPLIED

    EVENTS_APPLIED.inc(len(events))
    return len(events)


def redis_client():
    import redis

    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def run_learner() -> None:
    from prometheus_client import start_http_server

    from mci.serving.metrics import STREAM_LAG

    r = redis_client()
    ensure_group(r)
    stats = RedisStats(r)
    consumer = os.environ.get("HOSTNAME", socket.gethostname())
    port = int(os.environ.get("MCI_METRICS_PORT", "9100"))
    if port:
        start_http_server(port)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    log.info("learner %s consuming %s", consumer, STREAM)
    while not stop["now"]:
        consume_once(r, stats, consumer)
        info = r.xinfo_groups(STREAM)
        for g in info:
            if g.get("name") in (GROUP, GROUP.encode()):
                STREAM_LAG.set(g.get("lag") or 0)
    log.info("learner stopped")
