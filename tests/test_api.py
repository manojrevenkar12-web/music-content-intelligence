import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mci.serving.app import create_app
from mci.serving.engine import Engine

from .conftest import api_track


@pytest.fixture()
def client(tiny_bundle, r, monkeypatch):
    monkeypatch.setenv("MCI_API_TOKEN", "secret")
    app = create_app(tiny_bundle, redis_client=r)
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer secret"
        yield c


def test_bundle_serves_int8_and_fp32_consistently(tiny_bundle):
    a, b = Engine.load(tiny_bundle, "fp32"), Engine.load(tiny_bundle, "int8")
    items = [api_track(title=f"song {i}", energy=i / 10) for i in range(10)]
    va, pa = a.run(a.featurise(items))
    vb, pb = b.run(b.featurise(items))
    assert np.abs(va - vb).max() < 0.1 and np.abs(pa - pb).max() < 5
    assert len(a.popularity_sessions) == 2  # ensemble: LightGBM + network


def test_score(client):
    res = client.post(
        "/v1/tracks/score", json={"tracks": [api_track(track_id="x1"), api_track(track_id="x2", title="Corazón", artist="new artist")], "top_k": 3}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["model"] == "mci-test/int8" and res.headers["X-Model-Version"] == "mci-test/int8"
    assert [x["track_id"] for x in body["results"]] == ["x1", "x2"]
    for x in body["results"]:
        assert 0 <= x["popularity"] <= 100 and len(x["playlists"]) == 3
        assert x["playlists"][0]["score"] >= x["playlists"][-1]["score"]


def test_concurrent_requests_are_batched_and_answered_in_order(tiny_bundle, r):
    app = create_app(tiny_bundle, redis_client=r)

    async def go():
        import httpx

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                reqs = [c.post("/v1/tracks/score", json={"tracks": [api_track(track_id=f"t{i}", energy=(i % 10) / 10)]}) for i in range(40)]
                return await asyncio.gather(*reqs)

    res = asyncio.run(go())
    assert all(x.status_code == 200 for x in res)
    assert [x.json()["results"][0]["track_id"] for x in res] == [f"t{i}" for i in range(40)]
    single = Engine.load(tiny_bundle, "int8").score([api_track(energy=0.3)])[0]["popularity"]
    assert res[3].json()["results"][0]["popularity"] == pytest.approx(single, abs=0.11)


def test_validation_and_auth(client):
    assert client.post("/v1/tracks/score", json={"tracks": [api_track(energy=3)]}).status_code == 422
    assert client.post("/v1/tracks/score", json={"tracks": []}).status_code == 422
    assert client.post("/v1/tracks/score", json={"tracks": [api_track()]}, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_events_promote_and_live(client, r):
    from mci import online

    online.ensure_group(r)
    cands = [api_track(track_id=f"c{i}", title=f"cand {i}", energy=i / 10) for i in range(8)]
    first = client.post("/v1/contexts/pl3/promote", json={"candidates": cands, "slots": 3}).json()
    assert first["known_playlist"] and len(first["promote"]) == 3
    events = [{"type": "complete", "track_id": "c7", "context": "pl3"}] * 300 + [
        {"type": "skip", "track_id": f"c{i}", "context": "pl3"} for i in range(7) for _ in range(40)
    ]
    for i in range(0, len(events), 1000):
        assert client.post("/v1/events", json={"events": events[i : i + 1000]}).status_code == 202
    while online.consume_once(r, online.RedisStats(r), "test", block_ms=10):
        pass
    after = client.post("/v1/contexts/pl3/promote", json={"candidates": cands, "slots": 1}).json()
    assert after["promote"][0]["track_id"] == "c7" and after["promote"][0]["plays_seen"] == 300
    assert client.get("/v1/tracks/c7/live").json()["completions_last_hour"] == 300
    assert client.post("/v1/events", json={"events": [{"type": "like", "track_id": "a", "context": "b"}]}).status_code == 422


def test_model_and_ops_endpoints(client):
    assert client.get("/v1/model").json()["version"] == "mci-test"
    assert client.get("/readyz").json()["status"] == "ready"
    for _ in range(3):
        client.post("/v1/tracks/score", json={"tracks": [api_track()] * 100})
    m = client.get("/metrics").text
    assert "mci_inference_batch_size_bucket" in m and 'mci_feature_psi{feature="energy"}' in m and "mci_model_info" in m


def test_drain_closes_connections_and_fails_readiness(client):
    assert client.post("/internal/drain").status_code == 403  # only the pod itself (preStop) may drain
    client.app.state.draining = True
    assert client.get("/readyz").status_code == 503
    res = client.post("/v1/tracks/score", json={"tracks": [api_track()]})
    assert res.status_code == 200 and res.headers["connection"] == "close"
