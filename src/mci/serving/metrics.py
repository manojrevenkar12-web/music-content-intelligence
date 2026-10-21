"""Prometheus metrics shared by the API and the learner."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("mci_http_requests_total", "HTTP requests", ["route", "method", "code"])
LATENCY = Histogram("mci_http_request_duration_seconds", "HTTP latency", ["route"], buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
BATCH_SIZE = Histogram("mci_inference_batch_size", "Tracks per model call", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256))
INFERENCE = Histogram("mci_inference_seconds", "Model call duration", buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
PREDICTIONS = Counter("mci_predictions_total", "Tracks scored", ["model"])
PREDICTED_POPULARITY = Histogram(
    "mci_predicted_popularity", "Distribution of predicted popularity (watch for shifts)", buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90)
)
FEATURE_PSI = Gauge("mci_feature_psi", "Population stability index of live inputs vs training", ["feature"])
EVENTS_INGESTED = Counter("mci_events_ingested_total", "Listening events accepted")
EVENTS_APPLIED = Counter("mci_events_applied_total", "Listening events applied by learners")
STREAM_LAG = Gauge("mci_event_stream_lag", "Events in the stream not yet read by the learner group")
MODEL_INFO = Gauge("mci_model_info", "The model this process serves", ["version", "precision"])
