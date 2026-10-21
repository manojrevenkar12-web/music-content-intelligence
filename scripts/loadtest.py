"""Load test for the scoring API: latency, throughput and cost per million predictions.

    python scripts/loadtest.py --base http://localhost:8000 --users 32 --seconds 20 --out report.json

Each simulated client sends one new release per request (the worst case for batching) as fast as
the server answers. Cost per million predictions is derived from measured throughput and an
assumed price per vCPU-hour (--vcpu-hour-usd, --vcpus): change them to your provider's.
With --rollout-check the script exits non-zero on ANY failed request, which is how CI checks that
a model rollout drops nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import httpx
import numpy as np

TITLES = ["Night Drive", "Corazón Roto", "Summer Lights", "Baila Conmigo", "Blue Hour", "Fuego", "Lost in the City", "Midnight Remix"]
ARTISTS = ["Dua Lipa", "Bad Bunny", "a debut artist", "Martin Garrix", "J Balvin", "Billie Eilish"]


def track(rng: random.Random) -> dict:
    return {
        "title": rng.choice(TITLES), "artist": rng.choice(ARTISTS), "danceability": rng.random(), "energy": rng.random(), "loudness": rng.uniform(-14, -2),
        "speechiness": rng.random() * 0.3, "acousticness": rng.random(), "instrumentalness": rng.random() ** 4, "liveness": rng.random() * 0.4,
        "valence": rng.random(), "tempo": rng.uniform(70, 180), "duration_ms": rng.randint(120_000, 300_000), "mode": rng.randint(0, 1), "key": rng.randint(0, 11),
    }  # fmt: skip


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--token", default="")
    ap.add_argument("--vcpus", type=float, default=1.0, help="vCPUs the server under test used")
    ap.add_argument("--vcpu-hour-usd", type=float, default=0.04, help="assumed on-demand price per vCPU-hour")
    ap.add_argument("--tracks-per-request", type=int, default=1, help="1 = interactive scoring; e.g. 64 = bulk catalogue scoring")
    ap.add_argument("--rollout-check", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    lat: list[float] = []
    errors: dict[str, int] = {}
    versions: dict[str, int] = {}
    stop = time.monotonic() + a.seconds
    limits = httpx.Limits(max_connections=a.users, max_keepalive_connections=a.users)
    async with httpx.AsyncClient(base_url=a.base, headers=headers, timeout=10, limits=limits) as c:
        await c.post("/v1/tracks/score", json={"tracks": [track(random.Random(0))]})  # warm up

        async def user(seed: int):
            rng = random.Random(seed)
            while time.monotonic() < stop:
                t0 = time.perf_counter()
                try:
                    r = await c.post("/v1/tracks/score", json={"tracks": [track(rng) for _ in range(a.tracks_per_request)], "top_k": 10})
                    code = str(r.status_code)
                    if r.status_code == 200:
                        v = r.headers.get("x-model-version", "?")
                        versions[v] = versions.get(v, 0) + 1
                except httpx.HTTPError as e:
                    code = type(e).__name__
                lat.append((time.perf_counter() - t0) * 1000)
                if code != "200":
                    errors[code] = errors.get(code, 0) + 1

        t0 = time.monotonic()
        await asyncio.gather(*(user(i) for i in range(a.users)))
        elapsed = time.monotonic() - t0
    ok = len(lat) - sum(errors.values())
    rps = ok / elapsed
    tps = rps * a.tracks_per_request
    report = {
        "users": a.users, "tracks_per_request": a.tracks_per_request, "seconds": round(elapsed, 1), "requests": len(lat), "errors": errors,
        "throughput_rps": round(rps, 1), "tracks_per_second": round(tps, 1),
        "latency_ms": {p: round(float(np.percentile(lat, q)), 2) for p, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "model_versions_served": versions,
        "cost": {"assumed_vcpu_hour_usd": a.vcpu_hour_usd, "vcpus": a.vcpus, "usd_per_million_predictions": round(a.vcpu_hour_usd * a.vcpus / (tps * 3600) * 1e6, 4) if tps else None},
    }  # fmt: skip
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    if a.rollout_check and errors:
        print(f"rollout dropped requests: {errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
