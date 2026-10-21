# Architecture

```
                         offline (batch, retrained on a schedule)
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │ mci fetch ─► point-in-time features ─► baselines ─► candidates ─► select on validation      │
 │  (pinned       (artist history before    (heuristics)   (two-tower variants,              │
 │   sha256)       release, track age)                      LightGBM, network, ensemble)     │
 │                                                     ─► score test once ─► gate ─► bundle   │
 │                                                        (bootstrap CIs)  (ONNX fp32 + int8, │
 │                                                                          manifest, MLflow) │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
                                         │ bundle baked into the image; ConfigMap selects it
                                         ▼
 online (Kubernetes)
 ┌──────────────── api (2-10 pods, HPA) ───────────────┐        ┌──────── learner (1-n) ───────┐
 │ POST /v1/tracks/score ─► micro-batcher ─► ONNX RT    │        │ XREADGROUP mci:events        │
 │ POST /v1/events ───────────────────────────► XADD ───┼──► Redis ◄─ HINCRBY counters, XACK   │
 │ POST /v1/contexts/{id}/promote ◄─ Thompson sampling ◄┼─── counters + live minute buckets    │
 │ /metrics: latency, batch size, PSI drift, versions   │        │ XAUTOCLAIM a dead pod's work │
 └──────────────────────────────────────────────────────┘        └──────────────────────────────┘
```

## Decisions

**Two tasks, one champion each.** Playlist matching and popularity are different problems. The
multimodal two-tower model wins playlist matching, because words in titles, artist names and
playlist names carry what audio descriptors cannot. A gradient-boosted tree on audio plus artist
history wins popularity, as it usually does on tabular signals. Forcing one network to do both
made each worse (an earlier run: [MODEL_CARD.md](MODEL_CARD.md)). Each task's champion is
chosen on the validation quarter. The test releases are scored once.

**Point-in-time features or nothing.** The split is by release date, and every input is what was
knowable on a track's release day: the artist's earlier releases, the track's age, and its
audio and text. A random split would leak an artist's future hits into their past and make
every number in the report look better than it is.

**A bundle is the unit of deployment.** A bundle is one directory with the ONNX graphs, the
featuriser, the playlist vectors, the artist-history table, drift references and a manifest. The
manifest records the dataset hash, the MLflow run, the metrics and the gate. The serving image
contains no PyTorch and no pandas. Training and serving share `features.py`, and a test checks
that they produce identical inputs.

**The model is configuration.** `MCI_BUNDLE` and `MCI_PRECISION` live in a kustomize-generated
ConfigMap. Changing either changes the ConfigMap's hash, which rolls the Deployment. Rollouts
use `maxUnavailable: 0`, readiness gates and a drain step (below). CI performs one under load and
fails on any dropped request.

**Draining, because rolling updates drop requests by default.** Kube-proxy keeps routing a
keep-alive connection to its old pod. The preStop hook does three things before the container
gets SIGTERM:
- marks the pod as draining;
- fails readiness;
- answers every remaining request with `Connection: close`.

Clients reconnect, and their new connections reach ready pods.

**Batching where it pays.** Measured (`reports/serving.json`): one model call costs 0.11 ms for one
track, and 1.5 ms for 64 tracks (23 µs each). A one-track HTTP request costs about 2 ms in all,
mostly HTTP, JSON and validation. So batching concurrent single-track requests changes little,
and the batcher is opportunistic: it merges whatever is already queued, never waits.
Batching pays for **bulk scoring**: one request with 64 tracks reaches 4,594 tracks/s per pod, about
9× cheaper per prediction than one-track requests. Catalogue re-scoring should use that path.

**int8 for size, not speed.** Dynamic quantisation makes the track model four times smaller
(13.3 → 3.3 MB) with the same recall (0.3399 vs 0.3396). It is *not* faster for one track
(0.113 vs 0.112 ms), because the activations are quantised at run time. It is 23% faster at 64
tracks per call (1.47 vs 1.90 ms). It is the default because image size, pull time and memory
per pod are the costs that scale with replicas, and bulk scoring is where compute goes.

**Events through a stream, not the request path.** `POST /v1/events` only appends to a Redis
Stream. The learners are a consumer group, so pods share the work, and every event is acked
after its counters are written. A learner that dies leaves its pending events to be reclaimed
by the others. Counters are idempotent at the batch level, but not exactly-once across a crash
between write and ack. That can count a batch twice. The trade-off is accepted for engagement
counters, and the docs say so.

**Thompson sampling with a model prior.** The offline model knows a track before anyone has heard
it. Listening data knows it better after a few hundred plays. A Beta prior centred on the model's
score, worth 20 plays, gets both. `reports/online_learning.json` measures what that buys.

## Scaling notes

- The API is stateless. Scale it with the HPA on CPU, one uvicorn worker per pod and one ONNX
  thread per worker. That avoids thread oversubscription, and batching uses the core.
- Redis holds counters: O(contexts × candidates) hash fields, a few bytes each. The stream is
  capped at one million events (`MAXLEN ~`). Production would use a managed Redis with a replica
  and AOF. Past one primary's throughput, events would move to Kafka and the learner's
  `consume_once` would become a Kafka consumer; the API contract would not change.
- The catalogue here is 28k tracks and 471 playlists, so playlist scoring is a 471×64 matrix
  product. At millions of playlists it becomes an approximate nearest-neighbour index (FAISS
  or ScaNN) over the same vectors. The two-tower split exists so that this is possible.
- Retraining is one command, `mci train`, and needs about 15 minutes on 2 CPU cores. It would
  run as a scheduled Kubernetes Job writing bundles to object storage, with the image
  referencing bundles by URL, not baked in.
