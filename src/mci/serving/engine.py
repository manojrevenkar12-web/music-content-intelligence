"""Inference without PyTorch: a bundle directory → ONNX Runtime session + playlist vectors.

One `Engine` per process. `run` takes a batch of featurised tracks; `score` takes API payloads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mci.features import Featurizer, request_frame

INPUTS = ["audio", "context", "title_words", "title_tris", "artist_words", "artist_tris"]


@dataclass
class Engine:
    session: ort.InferenceSession
    featurizer: Featurizer
    playlist_vectors: np.ndarray
    playlists: list[dict]
    artists: dict[str, list]
    manifest: dict
    precision: str
    popularity_sessions: list  # [(kind, session)]; empty = the track model's own popularity head

    @classmethod
    def load(cls, bundle: Path, precision: str = "int8", threads: int | None = None) -> Engine:
        bundle = Path(bundle)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("MCI_THREADS", "1"))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        name = "model.int8.onnx" if precision == "int8" else "model.onnx"
        sess = ort.InferenceSession(str(bundle / name), opts, providers=["CPUExecutionProvider"])
        manifest = json.loads((bundle / "manifest.json").read_text())
        kind = manifest.get("popularity_model", "neural")
        pops = []
        if kind in ("lightgbm", "ensemble"):
            pops.append(("lightgbm", ort.InferenceSession(str(bundle / "popularity_lgbm.onnx"), opts, providers=["CPUExecutionProvider"])))
        if kind in ("neural-pop", "ensemble"):
            nn_name = "popularity_nn.int8.onnx" if precision == "int8" else "popularity_nn.onnx"
            pops.append(("neural-pop", ort.InferenceSession(str(bundle / nn_name), opts, providers=["CPUExecutionProvider"])))
        return cls(
            session=sess,
            featurizer=Featurizer.load(bundle / "featurizer.json"),
            playlist_vectors=np.load(bundle / "playlist_vectors.npy"),
            playlists=json.loads((bundle / "playlists.json").read_text()),
            artists=json.loads((bundle / "artists.json").read_text()),
            manifest=manifest,
            precision=precision,
            popularity_sessions=pops,
        )

    @property
    def version(self) -> str:
        return f"{self.manifest['version']}/{self.precision}"

    @property
    def scale(self) -> float:
        return float(self.manifest.get("scale", 10.0))

    def run(self, x: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        feeds = {k: x[k] for k in INPUTS}
        vec, pop = self.session.run(None, feeds)
        if self.popularity_sessions:
            preds = []
            for kind, sess in self.popularity_sessions:
                if kind == "lightgbm":
                    preds.append(sess.run(None, {"tabular": np.hstack([x["audio"], x["context"]]).astype(np.float32)})[0].ravel())
                else:
                    preds.append(sess.run(None, feeds)[1])
            pop = np.clip(np.mean(preds, axis=0), 0, 100)
        return vec, pop

    def featurise(self, items: list[dict]) -> dict[str, np.ndarray]:
        cols = request_frame(items)
        # point-in-time artist context: the caller's numbers if given, else the bundle's history
        for i, it in enumerate(items):
            if it.get("artist_prior_tracks") is None and it.get("artist") in self.artists:
                cols["artist_prior_tracks"][i], cols["artist_prior_pop"][i] = self.artists[it["artist"]]
        return self.featurizer.tracks(cols)

    @property
    def playlist_index(self) -> dict[str, int]:
        if not hasattr(self, "_pidx"):
            self._pidx = {p["playlist_id"]: i for i, p in enumerate(self.playlists)}
        return self._pidx

    def score(self, items: list[dict], top_k: int = 10) -> list[dict]:
        return self.score_features(self.featurise(items), top_k)

    def score_features(self, x: dict[str, np.ndarray], top_k: int = 10) -> list[dict]:
        vec, pop = self.run(x)
        logits = self.scale * vec @ self.playlist_vectors.T
        out = []
        for v, lg, p in zip(vec, logits, pop, strict=True):
            top = np.argpartition(-lg, top_k)[:top_k]
            top = top[np.argsort(-lg[top])]
            prob = np.exp(lg - lg.max())
            prob /= prob.sum()
            out.append(
                {
                    "popularity": round(float(p), 1),
                    "playlists": [
                        {
                            "id": self.playlists[j]["playlist_id"],
                            "name": self.playlists[j]["playlist_name"],
                            "genre": self.playlists[j]["playlist_genre"],
                            "score": round(float(prob[j]), 4),
                        }
                        for j in top
                    ],  # fmt: skip
                    "vector": v,
                }
            )
        return out
