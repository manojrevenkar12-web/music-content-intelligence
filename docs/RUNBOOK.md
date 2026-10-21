# Runbook

## Service levels

| Objective | Target | Metric |
|---|---|---|
| Scoring latency | p99 < 50 ms at the HPA's target load (measured: 15.5 ms at 506 req/s per pod) | `mci_http_request_duration_seconds{route="/v1/tracks/score"}` |
| Availability | 99.9% non-5xx | `mci_http_requests_total` |
| Event freshness | learner lag < 10 000 events, for under 5 minutes | `mci_event_stream_lag` |
| Model rollout | zero failed requests | CI `kubernetes` job: rollout under load |

## Roll out a new model

1. Train: `mci train`. Read `reports/training.json`. The gate must say `"promote": true`, and the
   bundle's `manifest.json` has the numbers.
2. Add the bundle directory to `bundles/`, then build and push the image.
3. Change `MCI_BUNDLE` in the overlay's ConfigMap, then `kubectl apply -k`. Pods roll one at a
   time and drain before they stop.
4. Watch `mci_model_info`, latency, and `mci_predicted_popularity`. A sudden shift in the predicted
   distribution means the new model disagrees with the old one on live traffic. That needs a
   look before anyone relies on it.

**Rollback:** `kubectl -n mci rollout undo deployment/api`, or set `MCI_BUNDLE` back. Bundles are
immutable directories, and the old one is still in the image.

## Alerts

### Input drift (`mci_feature_psi` > 0.25 on any feature for an hour)

The tracks being scored no longer look like the training catalogue. Typical causes:
- a new genre is being signed;
- an upstream change in how audio descriptors are computed;
- a client sending defaults (all zeros).

1. Check which feature drifted. If one feature jumped and the others did not, suspect the client
   or the upstream extractor, not the music.
2. If the music really changed, retrain with recent releases included (`mci train`). The release
   gate decides whether the new model is better.

### Learner lag (`mci_event_stream_lag` growing)

1. Check whether the learner pods are running: `kubectl -n mci get pods -l app=learner`.
2. If they are running but slow, scale them (`kubectl -n mci scale deployment/learner --replicas=3`).
   The consumer group shares the stream automatically.
3. If a learner crashed mid-batch, its pending events are reclaimed after 60 s. Nothing needs
   replaying by hand.

### Latency (p99 over target)

1. Check `mci_inference_batch_size`. Big batches at high load are expected. If p99 is still over
   target, the HPA is behind: check `kubectl -n mci get hpa api`.
2. Check `MCI_THREADS` (1) and `WEB_CONCURRENCY` (1). More threads per pod on a 1-CPU limit
   makes latency worse, not better.
3. Precision makes no difference to latency here (`reports/serving.json`). If the model call
   itself is slow, check that bulk callers send many tracks per request, not one each.

### Redis unavailable

`/readyz` fails, and pods leave the Service. Scoring stops too, because readiness covers Redis.
That is deliberate: promotion without counters would silently fall back to the prior. If you
would rather keep scoring during a Redis outage, remove `REDIS_URL` from the ConfigMap. The API
then serves scoring only and answers 503 on the real-time routes.
