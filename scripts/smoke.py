"""Smoke test against a running deployment (standard library only).

    python scripts/smoke.py --base http://localhost:8000 [--token ...]

Scores a new release, checks the answer's shape and model version, sends listening events,
waits for the learner to apply them, and checks that promotion follows the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

TRACK = {"title": "Night Drive", "artist": "a debut artist", "danceability": 0.62, "energy": 0.81, "loudness": -5.2, "speechiness": 0.05, "acousticness": 0.04,
         "instrumentalness": 0.0, "liveness": 0.12, "valence": 0.48, "tempo": 118.0, "duration_ms": 201_000, "mode": 1, "key": 7}  # fmt: skip


def call(base, path, body=None, token=""):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None, headers=h, method="POST" if body is not None else "GET"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--context", help="a playlist id from the bundle (defaults to the first suggested)")
    a = ap.parse_args()
    m = call(a.base, "/v1/model", token=a.token)
    print("model", m["version"], m["precision"], "variant:", m.get("variant"))
    res = call(a.base, "/v1/tracks/score", {"tracks": [{**TRACK, "track_id": "smoke-1"}], "top_k": 5}, a.token)["results"][0]
    assert 0 <= res["popularity"] <= 100 and len(res["playlists"]) == 5, res
    print("score", res["popularity"], [p["name"] for p in res["playlists"]])
    ctx = a.context or res["playlists"][0]["id"]
    cands = [{**TRACK, "track_id": f"cand-{i}", "title": f"Candidate {i}", "energy": i / 10} for i in range(6)]
    events = [{"type": "complete", "track_id": "cand-5", "context": ctx}] * 200 + [
        {"type": "skip", "track_id": f"cand-{i}", "context": ctx} for i in range(5) for _ in range(40)
    ]
    assert call(a.base, "/v1/events", {"events": events}, a.token)["accepted"] == len(events)
    for _ in range(60):  # the learner applies the stream asynchronously
        live = call(a.base, "/v1/tracks/cand-5/live", token=a.token)["completions_last_hour"]
        if live >= 200:
            break
        time.sleep(1)
    else:
        print("learner did not apply events within 60 s", file=sys.stderr)
        return 1
    top = call(a.base, f"/v1/contexts/{ctx}/promote", {"candidates": cands, "slots": 1}, a.token)["promote"][0]
    assert top["track_id"] == "cand-5" and top["plays_seen"] >= 200, top
    print("promote", top)
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
