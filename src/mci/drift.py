"""Input drift: are the tracks being scored today like the ones the model learned from?

Population stability index (PSI) per audio descriptor, against training deciles stored in the
bundle. Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 the model is extrapolating (see
docs/RUNBOOK.md)."""

from __future__ import annotations

from collections import deque

import numpy as np

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration",
    "mode",
    "key_sin",
    "key_cos",
]
CONTINUOUS = FEATURES[:10]


def reference(audio: np.ndarray) -> dict:
    """Decile edges of each continuous feature on the training data (already standardised), and the
    share of training tracks in each bin (not exactly 10% where values repeat, e.g. zeros)."""
    out = {}
    for f in CONTINUOUS:
        x = audio[:, FEATURES.index(f)]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)[1:-1])).round(5)
        expected = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1) / len(x)
        out[f] = {"edges": edges.tolist(), "expected": expected.round(5).tolist()}
    return out


def psi(ref: dict, values: np.ndarray) -> float:
    edges, expected = np.array(ref["edges"]), np.array(ref["expected"])
    actual = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1) / max(1, len(values))
    e, a = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


class Monitor:
    def __init__(self, ref: dict, window: int = 2000):
        self.ref = ref
        self.buf: deque = deque(maxlen=window)

    def observe(self, audio: np.ndarray) -> None:
        self.buf.extend(audio)

    def report(self, min_count: int = 200) -> dict[str, float] | None:
        if len(self.buf) < min_count:
            return None
        x = np.array(self.buf)
        return {f: round(psi(self.ref[f], x[:, FEATURES.index(f)]), 4) for f in self.ref}
