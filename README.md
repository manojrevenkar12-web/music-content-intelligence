# Music Content Intelligence

A new track is about to be released. Which playlists should it be pitched to, how well will it
do, and, once people start listening, what should each playlist play next?

This repository answers those three questions as a production ML system:

- **a multimodal model** that reads a release's title, artist and audio descriptors, and ranks 471
  playlists for it;
- **a popularity model**;
- **an online learner** that updates from listening events in real time;
- **a serving layer** on Kubernetes, with the training pipeline, release gate, MLflow registry,
  drift monitoring and a CI job that rolls out a model under load and fails if a single request
  drops.

It trains on **real data**: 28,352 Spotify tracks and their placements in 471 playlists. The
split is by release date, so every number below is measured on the 3,587 newest releases, which
the models never saw.

## Results

All numbers come from the files in [`reports/`](reports). Method and limits are in
[docs/MODEL_CARD.md](docs/MODEL_CARD.md).

**Playlist matching** (recall@10: the share of a release's real playlists that appear in the top 10)

| | all releases | debut artists | artists with history |
|---|---:|---:|---:|
| strongest heuristic: the artist's past playlists, then similar-sounding recent tracks | 0.293 | 0.177 | 0.409 |
| **multimodal two-tower model (served)** | **0.340** | **0.238** | **0.442** |
| gain, 95% bootstrap CI | +0.047 [0.030, 0.062] | **+0.061 [0.038, 0.084]** | +0.033 [0.010, 0.058] |

Half of the new releases come from artists with no earlier track in the catalogue. That is where
the heuristic has nothing to go on, and where the model adds most. The ablation shows the gain
comes from **pretrained word vectors**:

| Variant | recall@10 |
|---|---:|
| audio + context | 0.317 |
| + text, no pretrained vectors | 0.307 |
| + text with pretrained vectors | 0.340–0.347 |

**Popularity.** LightGBM was chosen on the validation quarter:
- MAE 10.46, against 13.46 for the artist-history heuristic (gate: 3.0 points better, CI [2.7, 3.3]);
- hit AUC 0.86.

The neural network trained for popularity lost (MAE 13.09). It is reported, not shipped.

**Learning from listening** (semi-synthetic: real tracks and real placements, simulated listeners;
20 playlists, 72 hours, 5 seeds)

| Policy | completion rate, 72 h | first 6 hours | regret vs oracle |
|---|---:|---:|---:|
| offline model only, never learns | 58.4% | 58.4% | 92,632 |
| learning only (Thompson sampling, flat prior) | 77.2% | 66.9% | 11,420 |
| **model prior + learning (served)** | **77.8%** | **72.9%** | **8,842 (−23%)** |
| oracle | 79.9% | 79.9% | 0 |

Learning from live plays is what gets close to the oracle. The offline model's prior is what
makes the first hours of a release good.

**Serving** (one pod's shape: 1 worker, 1 ONNX thread; 2-vCPU machine shared with the load generator)

| | throughput | p99 | cost per million predictions* |
|---|---:|---:|---:|
| interactive, 4 clients | 506 req/s | **15.5 ms** | $0.022 |
| bulk catalogue scoring, 64 tracks per request | **4,594 tracks/s** | 97 ms per request | **$0.0024** |
| model call alone, int8 | 1 track 0.11 ms, 64 tracks 1.5 ms | | |

<sub>*At an assumed $0.04 per vCPU-hour; the load-test script takes your price.</sub>

int8 quantisation shrinks the model 4× (13.3 → 3.3 MB) with the same recall (0.3399 vs 0.3396).
Batching single-track requests changes little, because HTTP costs 20× the model call. Bulk
requests are where batching pays ([ARCHITECTURE.md](docs/ARCHITECTURE.md)).

## How it works

```
 offline   fetch (pinned sha256) → point-in-time features → heuristics + candidates → select on validation
           → score the test releases once → release gate (bootstrap CIs) → bundle (ONNX fp32 + int8) → MLflow

 online    POST /v1/tracks/score ──► opportunistic batching ──► ONNX Runtime ──► playlists + popularity
           POST /v1/events ──► Redis Stream ──► learner consumer group ──► per-playlist counters
           POST /v1/contexts/{id}/promote ◄── Thompson sampling: model prior + live counters
```

- **Point-in-time features.** An artist's history only counts releases before the track's own
  release date. Track age is an input, because new releases go to new-music playlists. A random
  split would leak an artist's future into their past.
- **Two-tower model**, so playlist scoring is one matrix product today and an ANN index at a
  million playlists:
  - the track tower reads words (spaCy vectors, fine-tuned), character trigrams for the
    Spanish and Portuguese titles, audio descriptors, and artist history and age;
  - the playlist tower reads the playlist's name and identity. The identity is dropped in
    training, so a new playlist works from its name alone.
- **One champion per task, chosen on validation; the test set is scored once.** The gate
  promotes only if both tasks beat the strongest heuristic with 95% confidence, and p99 is
  under 50 ms.
- **A bundle is the unit of deployment.** It holds the ONNX graphs, featuriser, playlist vectors,
  artist history, drift references, and a manifest with the data hash, MLflow run, metrics and
  gate. The serving image has no PyTorch or pandas. A test proves that training and serving
  featurise identically.
- **Real-time:**
  - events are appended to a Redis Stream;
  - learners form a consumer group, so pods share the stream;
  - each event is acked after its counter update;
  - a dead pod's pending events are reclaimed with `XAUTOCLAIM`.
- **Kubernetes:**
  - the API runs 2–10 pods (HPA) with probes, a PDB, a NetworkPolicy, non-root containers
    and a read-only filesystem;
  - the model is a ConfigMap, so changing it rolls the Deployment;
  - a preStop drain fails readiness and sends `Connection: close`, so a rollout drops nothing.
    CI proves it.
- **Monitoring:** Prometheus metrics for latency, batch size, predicted-popularity distribution,
  per-feature drift (PSI against training deciles), stream lag and the serving model's version.
  [RUNBOOK.md](docs/RUNBOOK.md) says what to do when each one alerts.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt && pip install -e .
mci fetch                         # 7.7 MB, verified against a pinned sha256
mci train                         # ≈15 min on 2 CPU cores → bundles/<version>, reports/training.json, MLflow
REDIS_URL=redis://localhost:6379/0 mci serve & mci learner &
python scripts/smoke.py           # score → events → learner → promotion
```

```bash
curl -s localhost:8000/v1/tracks/score -H 'content-type: application/json' -d '{"tracks": [{
  "title": "Corazón de Neón", "artist": "a debut artist", "danceability": 0.78, "energy": 0.71, "loudness": -5.1,
  "speechiness": 0.09, "acousticness": 0.12, "instrumentalness": 0, "liveness": 0.1, "valence": 0.66,
  "tempo": 96, "duration_ms": 194000, "mode": 0, "key": 1}]}'
```

On Kubernetes (the same steps CI runs on a kind cluster):

```bash
docker build -t mci:ci . && kind create cluster --config deploy/kind/cluster.yaml && kind load docker-image mci:ci
kubectl apply -k deploy/k8s/overlays/kind && python scripts/smoke.py --base http://localhost:30080
```

Benchmarks: `mci simulate`, `python scripts/bench_serving.py`, `python scripts/cold_start.py`.

## CI

| Job | What it proves |
|---|---|
| **pipeline** | ruff; 26 tests, including identical training/serving features, point-in-time artist history, stream redelivery after a crash, Thompson sampling, the API and int8/fp32 agreement; the full training pipeline end to end on a slice |
| **kubernetes** | strict schema validation of both overlays; the image builds; Redis, API ×2 and learner deploy to kind; the smoke test passes through the Service; a load test runs; then the model is rolled out *under load*, and the job fails if any request drops or the new version is not observed serving |

## Limits

- One January 2020 snapshot of Spotify data. Popularity is Spotify's index, not streams or revenue.
- "Multimodal" here means text plus Spotify's audio *descriptors*. The dataset has no waveforms,
  so the model never hears the audio.
- The online-learning benchmark simulates listeners. Its numbers show how the policies compare,
  not what a real launch will get.
- Redis runs as a single instance on the demo cluster. Production needs a managed Redis with a
  replica, or Kafka at higher event rates ([ARCHITECTURE.md](docs/ARCHITECTURE.md)).

MIT licence. Data: TidyTuesday 2020-01-21 (CC0). Word vectors: spaCy `en_core_web_md` (MIT).
