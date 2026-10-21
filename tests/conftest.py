from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(os.environ.get("MCI_DATA", "data/spotify_songs.csv"))


def track_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    words = ["night", "drive", "corazón", "amor", "fuego", "dance", "lights", "remix", "summer", "blue", "baila", "love"]
    return pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(n)],
            "track_name": [" ".join(rng.choice(words, 2)) for _ in range(n)],
            "track_album_name": [" ".join(rng.choice(words, 1)) for _ in range(n)],
            "track_artist": [f"artist {i % 7}" for i in range(n)],
            "track_popularity": rng.integers(0, 100, n),
            "release_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
            "danceability": rng.random(n), "energy": rng.random(n), "loudness": rng.uniform(-20, 0, n), "speechiness": rng.random(n) * 0.3,
            "acousticness": rng.random(n), "instrumentalness": rng.random(n) ** 3, "liveness": rng.random(n), "valence": rng.random(n),
            "tempo": rng.uniform(70, 180, n), "duration_ms": rng.integers(120_000, 300_000, n), "mode": rng.integers(0, 2, n), "key": rng.integers(-1, 12, n),
            "artist_prior_tracks": rng.integers(0, 5, n), "artist_prior_pop": rng.uniform(10, 80, n), "age_days": rng.integers(0, 400, n),
        }
    )  # fmt: skip


def api_track(**kw) -> dict:
    t = {"title": "Night Drive", "artist": "artist 1", "danceability": 0.6, "energy": 0.7, "loudness": -6.0, "speechiness": 0.05, "acousticness": 0.1,
         "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.5, "tempo": 120.0, "duration_ms": 200_000, "mode": 1, "key": 5}  # fmt: skip
    return {**t, **kw}


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> Path:
    """A real bundle from a briefly trained model on random tracks: exercises export, quantisation,
    LightGBM conversion and serving, not model quality."""
    pytest.importorskip("torch")
    from mci import data, train

    df = track_rows(400)
    n_pl = 12
    playlists = pd.DataFrame({"idx": range(n_pl), "playlist_id": [f"pl{i}" for i in range(n_pl)], "playlist_name": [f"playlist {w}" for w in ["chill", "party", "latin", "rock", "indie", "rap", "edm", "pop", "r&b", "summer", "new", "throwback"]],
                              "playlist_genre": ["pop"] * n_pl, "playlist_subgenre": ["x"] * n_pl})  # fmt: skip
    membership = {t: [i % n_pl] for i, t in enumerate(df["track_id"])}
    df["split"] = "train"
    df.attrs["snapshot"] = "2020-01-01"
    cat = data.Catalogue(df, playlists, membership)
    cfg = train.Config(epochs=2, use_pretrained=False)
    fitted = train.fit(cfg, cat, df, None, epochs=2)
    lg = train.lightgbm_fit(fitted.featurizer, df, None, n_estimators=20)
    out = tmp_path_factory.mktemp("bundle") / "mci-test"
    train.write_bundle(fitted, cat, df, out, "mci-test", {"test": True}, {"variant": "tiny"}, popularity="ensemble", lgbm=lg, pop_fitted=fitted)
    return out


@pytest.fixture(scope="session")
def redis_url():
    """REDIS_URL if set (CI service), else a throwaway redis-server on a free port."""
    if os.environ.get("REDIS_URL"):
        yield os.environ["REDIS_URL"]
        return
    exe = shutil.which("redis-server")
    if not exe:
        pytest.skip("no redis-server and no REDIS_URL")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([exe, "--port", str(port), "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    import redis

    for _ in range(50):
        try:
            redis.Redis.from_url(url).ping()
            break
        except redis.ConnectionError:
            time.sleep(0.1)
    yield url
    proc.terminate()


@pytest.fixture()
def r(redis_url):
    import redis

    client = redis.Redis.from_url(redis_url)
    client.flushdb()
    return client
