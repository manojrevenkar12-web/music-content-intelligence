"""The inference API.

    POST /v1/tracks/score              new releases → predicted popularity + best-fitting playlists
    POST /v1/events                    listening events → Redis Stream (the learner applies them)
    POST /v1/contexts/{id}/promote     which candidates to play now in a playlist (Thompson sampling)
    GET  /v1/tracks/{id}/live          plays in the last hour (real-time feature)
    GET  /v1/model                     the serving model's manifest: version, data, metrics, gate

Scoring requests are batched opportunistically: whatever requests are already queued when the
model becomes free go to ONNX Runtime together (up to MCI_MAX_BATCH tracks). MCI_BATCH_WAIT_MS
adds a collection window. For this model the call is ~0.1 ms of a ~2 ms request, so neither
matters much for one-track requests (reports/serving.json). What matters is bulk requests: 64
tracks in one call cost ~23 us each instead of ~113 us.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from mci import drift, online
from mci.serving import metrics as M
from mci.serving.engine import Engine

log = logging.getLogger("mci.api")


# ------------------------------------------------------------------------------------- schemas
class Track(BaseModel):
    track_id: str | None = Field(default=None, max_length=64)
    title: str = Field(min_length=1, max_length=300)
    artist: str = Field(min_length=1, max_length=200)
    album: str = Field(default="", max_length=300)
    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    loudness: float = Field(ge=-60, le=5)
    speechiness: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    liveness: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    tempo: float = Field(ge=0, le=300)
    duration_ms: int = Field(ge=1_000, le=3_600_000)
    mode: int = Field(ge=0, le=1)
    key: int = Field(ge=-1, le=11)
    artist_prior_tracks: int | None = Field(default=None, ge=0, description="omit to use the catalogue's history for this artist")
    artist_prior_pop: float | None = Field(default=None, ge=0, le=100)
    age_days: int = Field(default=0, ge=0, le=36500)


class ScoreRequest(BaseModel):
    tracks: list[Track] = Field(min_length=1, max_length=256)
    top_k: int = Field(default=10, ge=1, le=50)


class Event(BaseModel):
    type: str
    track_id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=64)
    user: str | None = Field(default=None, max_length=64)
    ts: float | None = None


class EventsRequest(BaseModel):
    events: list[Event] = Field(min_length=1, max_length=1000)


class PromoteRequest(BaseModel):
    candidates: list[Track] = Field(min_length=1, max_length=500)
    slots: int = Field(default=5, ge=1, le=50)


# ------------------------------------------------------------------------------------- batching
class Batcher:
    def __init__(self, engine: Engine, max_batch: int, wait_ms: float, monitor: drift.Monitor | None):
        self.engine, self.max_batch, self.wait = engine, max_batch, wait_ms / 1000.0
        self.inline_max = 16
        self.queue: asyncio.Queue = asyncio.Queue()
        self.monitor = monitor
        self.task: asyncio.Task | None = None

    async def start(self):
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task:
            self.task.cancel()

    async def score(self, items: list[dict], top_k: int) -> list[dict]:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((items, top_k, fut))
        return await fut

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            jobs, n = [first], len(first[0])
            deadline = loop.time() + self.wait
            while n < self.max_batch:
                if not self.queue.empty():  # opportunistic: take what is already waiting, never wait for it
                    job = self.queue.get_nowait()
                else:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        break
                    try:
                        job = await asyncio.wait_for(self.queue.get(), timeout)
                    except TimeoutError:
                        break
                jobs.append(job)
                n += len(job[0])
            items = [it for j in jobs for it in j[0]]
            top_k = max(j[1] for j in jobs)
            t0 = time.perf_counter()
            try:
                # small batches run inline (~0.1-0.5 ms; a thread hop costs as much); big ones off the loop
                results = self._run(items, top_k) if len(items) <= self.inline_max else await loop.run_in_executor(None, self._run, items, top_k)
            except Exception as e:  # one bad batch must not stop the loop
                for j in jobs:
                    if not j[2].done():
                        j[2].set_exception(e)
                continue
            M.INFERENCE.observe(time.perf_counter() - t0)
            M.BATCH_SIZE.observe(len(items))
            i = 0
            for its, k, fut in jobs:
                if not fut.done():
                    fut.set_result([{**r, "playlists": r["playlists"][:k]} for r in results[i : i + len(its)]])
                i += len(its)

    def _run(self, items, top_k):
        x = self.engine.featurise(items)
        res = self.engine.score_features(x, top_k)
        if self.monitor is not None:
            self.monitor.observe(x["audio"])
        return res


# ------------------------------------------------------------------------------------- app
def create_app(bundle: Path | None = None, redis_client=None) -> FastAPI:
    bundle = Path(bundle or os.environ.get("MCI_BUNDLE", "bundles/current"))
    precision = os.environ.get("MCI_PRECISION", "int8")
    token = os.environ.get("MCI_API_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = Engine.load(bundle, precision=precision)
        ref_path = bundle / "drift_reference.json"
        monitor = drift.Monitor(__import__("json").loads(ref_path.read_text())) if ref_path.exists() else None
        app.state.engine, app.state.monitor = eng, monitor
        app.state.batcher = Batcher(eng, int(os.environ.get("MCI_MAX_BATCH", "64")), float(os.environ.get("MCI_BATCH_WAIT_MS", "0")), monitor)
        app.state.redis = redis_client if redis_client is not None else (online.redis_client() if os.environ.get("REDIS_URL") else None)
        app.state.policy = online.ThompsonPolicy(online.RedisStats(app.state.redis)) if app.state.redis is not None else None
        M.MODEL_INFO.labels(eng.manifest["version"], precision).set(1)
        await app.state.batcher.start()
        log.info("serving %s", eng.version)
        yield
        await app.state.batcher.stop()

    app = FastAPI(title="Music Content Intelligence", version="1.0.0", lifespan=lifespan)

    app.state.draining = False

    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.app.state.draining:  # shutting down: finish this request, then make the client reconnect elsewhere
            response.headers["Connection"] = "close"
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        M.REQUESTS.labels(path, request.method, str(response.status_code)).inc()
        M.LATENCY.labels(path).observe(time.perf_counter() - t0)
        response.headers["X-Model-Version"] = getattr(request.app.state, "engine", None).version if hasattr(request.app.state, "engine") else ""
        return response

    def auth(request: Request):
        if token:
            got = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(got, token):
                raise HTTPException(401, "missing or invalid token")

    def redis_or_503(request: Request):
        if request.app.state.redis is None:
            raise HTTPException(503, "real-time features need REDIS_URL")
        return request.app.state.redis

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}

    @app.post("/internal/drain", include_in_schema=False)
    def drain(request: Request):
        """Called by the pod's preStop hook: fail readiness and close keep-alive connections, so
        a rollout or scale-down drops no request. Only reachable from inside the pod."""
        if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(403, "local only")
        request.app.state.draining = True
        return {"draining": True}

    @app.get("/readyz", include_in_schema=False)
    def readyz(request: Request):
        if request.app.state.draining:
            return JSONResponse({"status": "draining"}, 503)
        if not hasattr(request.app.state, "engine"):
            return JSONResponse({"status": "loading"}, 503)
        if request.app.state.redis is not None:
            try:
                request.app.state.redis.ping()
            except Exception:
                return JSONResponse({"status": "redis unavailable"}, 503)
        return {"status": "ready", "model": request.app.state.engine.version}

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        mon = request.app.state.monitor
        rep = mon.report() if mon else None
        for f, v in (rep or {}).items():
            M.FEATURE_PSI.labels(f).set(v)
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", dependencies=[Depends(auth)])
    def model(request: Request):
        m = request.app.state.engine.manifest
        return {
            "version": m["version"],
            "precision": precision,
            **{k: m.get(k) for k in ("created", "variant", "history_until", "dataset_sha256", "metrics", "gate", "int8", "mlflow_run_id")},
        }

    @app.post("/v1/tracks/score", dependencies=[Depends(auth)])
    async def score(body: ScoreRequest, request: Request):
        eng = request.app.state.engine
        items = [t.model_dump() for t in body.tracks]
        res = await request.app.state.batcher.score(items, body.top_k)
        M.PREDICTIONS.labels(eng.version).inc(len(items))
        for r in res:
            M.PREDICTED_POPULARITY.observe(r["popularity"])
        return {
            "model": eng.version,
            "results": [{"track_id": it["track_id"], "popularity": r["popularity"], "playlists": r["playlists"]} for it, r in zip(items, res, strict=True)],
        }

    @app.post("/v1/events", status_code=202, dependencies=[Depends(auth)])
    def events(body: EventsRequest, r=Depends(redis_or_503)):
        try:
            evs = [online.validate(e.model_dump(exclude_none=True)) for e in body.events]
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        n = online.publish(r, evs)
        M.EVENTS_INGESTED.inc(n)
        return {"accepted": n}

    @app.post("/v1/contexts/{context}/promote", dependencies=[Depends(auth)])
    async def promote(context: str, body: PromoteRequest, request: Request, r=Depends(redis_or_503)):
        eng: Engine = request.app.state.engine
        items = [t.model_dump() for t in body.candidates]
        if any(not it["track_id"] for it in items):
            raise HTTPException(422, "every candidate needs a track_id")
        res = await request.app.state.batcher.score(items, 1)
        vec = np.stack([x["vector"] for x in res])
        pop = np.array([x["popularity"] for x in res])
        j = eng.playlist_index.get(context)
        fit = vec @ eng.playlist_vectors[j] if j is not None else np.zeros(len(items))
        prior = online.prior_from_scores(_z(fit) + _z(pop))
        ids = [it["track_id"] for it in items]
        pol: online.ThompsonPolicy = request.app.state.policy
        pick = pol.choose(context, ids, prior, k=min(body.slots, len(ids)))
        mean = pol.mean(context, ids, prior)
        n, _ = pol.stats.get(context, ids)
        return {"context": context, "known_playlist": j is not None, "model": eng.version,
                "promote": [{"track_id": ids[i], "expected_completion": round(float(mean[i]), 4), "plays_seen": int(n[i])} for i in pick]}  # fmt: skip

    @app.get("/v1/tracks/{track_id}/live", dependencies=[Depends(auth)])
    def live(track_id: str, r=Depends(redis_or_503)):
        return {"track_id": track_id, "completions_last_hour": online.streams_last_hour(r, track_id)}

    return app


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else np.zeros_like(x)
