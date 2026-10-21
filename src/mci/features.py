"""Turn a track into model inputs. Pure NumPy: training and serving run exactly this code, and the
serving image needs neither PyTorch nor spaCy.

Three inputs per track:
  - audio: Spotify's 12 audio descriptors → 13 standardised numbers (key as a point on a circle);
  - context: what was known about the artist before release (zeros for a debut) and the track's
    age in days (0 for a release scored on its release day);
  - text: title + album and artist name, as word ids (pretrained vectors, fine-tuned) plus hashed
    character trigrams, which cover the Spanish, Portuguese and invented words the English vectors
    miss (much of the catalogue is Latin music).
"""

from __future__ import annotations

import json
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORD_RE = re.compile(r"\w+", re.UNICODE)
TRI_BUCKETS = 1 << 15
MAX_WORDS = {"title": 24, "artist": 6, "playlist": 12}
MAX_TRIS = {"title": 96, "artist": 32, "playlist": 64}
N_AUDIO = 13
N_CONTEXT = 4


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return WORD_RE.findall(text)


def trigrams(words: list[str]) -> list[int]:
    out = []
    for w in words:
        w = f"<{w}>"
        out.extend(zlib.crc32(w[i : i + 3].encode()) % (TRI_BUCKETS - 1) + 1 for i in range(len(w) - 2))
    return out


def _pad(ids: list[int], n: int) -> np.ndarray:
    a = np.zeros(n, np.int64)
    a[: min(n, len(ids))] = ids[:n]
    return a


@dataclass
class Featurizer:
    """Fitted on training tracks only; saved with the model so serving featurises identically."""

    vocab: dict[str, int] = field(default_factory=dict)  # word -> id (0 is padding, 1 is unknown)
    audio_mean: list[float] = field(default_factory=list)
    audio_std: list[float] = field(default_factory=list)

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, train, playlist_names: list[str], has_vector=lambda w: False, min_count: int = 3) -> Featurizer:
        """Vocabulary: every word with a pretrained vector, plus words seen `min_count` times."""
        counts: dict[str, int] = {}
        texts = [*train["track_name"], *train["track_album_name"], *train["track_artist"], *playlist_names]
        for t in texts:
            for w in tokens(t):
                counts[w] = counts.get(w, 0) + 1
        words = sorted(w for w, c in counts.items() if c >= min_count or has_vector(w))
        f = cls(vocab={w: i + 2 for i, w in enumerate(words)})
        raw = f._audio_raw(train)
        f.audio_mean = raw.mean(0).round(6).tolist()
        f.audio_std = (raw.std(0) + 1e-6).round(6).tolist()
        return f

    # ---------------------------------------------------------------- transforms
    @staticmethod
    def _audio_raw(df) -> np.ndarray:
        """`df` is a DataFrame (training) or a dict of columns (serving): both index by name."""

        def col(c):
            return np.asarray(df[c], dtype=float)

        key = col("key")
        cols = [
            col("danceability"), col("energy"), col("loudness"), col("speechiness"), np.sqrt(np.clip(col("acousticness"), 0, None)),
            np.sqrt(np.clip(col("instrumentalness"), 0, None)), col("liveness"), col("valence"), col("tempo"),
            np.log(np.clip(col("duration_ms"), 10_000, None) / 1000.0), col("mode"),
            np.where(key >= 0, np.sin(2 * np.pi * key / 12), 0.0), np.where(key >= 0, np.cos(2 * np.pi * key / 12), 0.0),
        ]  # fmt: skip
        return np.stack(cols, axis=1)

    def audio(self, df) -> np.ndarray:
        return ((self._audio_raw(df) - np.array(self.audio_mean)) / np.array(self.audio_std)).astype(np.float32)

    @staticmethod
    def context(df) -> np.ndarray:
        n = np.nan_to_num(np.asarray(df["artist_prior_tracks"], dtype=float))
        pop = np.asarray(df["artist_prior_pop"], dtype=float)
        age = np.asarray(df["age_days"], dtype=float) if "age_days" in df else np.zeros(len(n))
        seen = n > 0
        cols = [np.log1p(n) / 3.0, np.where(seen, np.nan_to_num(pop) / 100.0, 0.0), seen.astype(float), np.log1p(np.clip(np.nan_to_num(age), 0, None)) / 7.0]
        return np.stack(cols, axis=1).astype(np.float32)

    def text(self, texts, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """(word ids [n, MAX_WORDS], trigram ids [n, MAX_TRIS]) for an iterable of strings."""
        w_rows, t_rows = [], []
        for t in texts:
            ws = tokens(t)
            w_rows.append(_pad([self.vocab.get(w, 1) for w in ws], MAX_WORDS[kind]))
            t_rows.append(_pad(trigrams(ws), MAX_TRIS[kind]))
        return np.stack(w_rows), np.stack(t_rows)

    def tracks(self, df) -> dict[str, np.ndarray]:
        """All inputs of the track tower, named as in the ONNX graph."""
        title = [f"{a} {b}" for a, b in zip(df["track_name"], df["track_album_name"], strict=True)]
        tw, tt = self.text(title, "title")
        aw, at = self.text([str(a) for a in df["track_artist"]], "artist")
        return {"audio": self.audio(df), "context": self.context(df), "title_words": tw, "title_tris": tt, "artist_words": aw, "artist_tris": at}

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"vocab": self.vocab, "audio_mean": self.audio_mean, "audio_std": self.audio_std}))

    @classmethod
    def load(cls, path: Path) -> Featurizer:
        d = json.loads(path.read_text())
        return cls(vocab=d["vocab"], audio_mean=d["audio_mean"], audio_std=d["audio_std"])


def request_frame(items: list[dict]) -> dict[str, list]:
    """API payloads → the columns `Featurizer.tracks` reads. A dict of lists, not a DataFrame:
    building a DataFrame per request cost 20x the model call itself."""
    cols = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "mode",
        "key",
    ]
    out: dict[str, list] = {c: [it[c] for it in items] for c in cols}
    out["track_name"] = [str(it["title"]) for it in items]
    out["track_artist"] = [str(it["artist"]) for it in items]
    out["track_album_name"] = [str(it.get("album") or "") for it in items]
    out["artist_prior_tracks"] = [it.get("artist_prior_tracks") or 0 for it in items]
    out["artist_prior_pop"] = [np.nan if it.get("artist_prior_pop") is None else it["artist_prior_pop"] for it in items]
    out["age_days"] = [it.get("age_days") or 0 for it in items]
    return out
