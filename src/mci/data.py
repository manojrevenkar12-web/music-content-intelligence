"""The catalogue: 28,356 Spotify tracks placed in 471 editorial and user playlists.

Source: the TidyTuesday 2020-01-21 release of Spotify Web API data (collected with spotifyr,
January 2020). Each row is a (track, playlist) placement with Spotify's audio descriptors, the
track's popularity at collection time, and album release date.

Everything here is *point in time*: the split is by release date, and every feature a model sees
for a track is computable on the day that track is released. That is the question a label actually
faces: a new track arrives; where should it be pitched, and how well will it do?
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
SHA256 = "9902920cfe4940f0914ebb6716801a42f7f19ac88fbd6693eaae7e2822b16845"

# release-date split: train on the past, tune on the next quarter, test on the newest releases
VALID_FROM = pd.Timestamp("2019-07-01")
TEST_FROM = pd.Timestamp("2019-10-01")

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "mode", "key"]
HIT_QUANTILE = 0.8  # a "hit" is a track in the top 20% of training-period popularity


def fetch(path: Path) -> Path:
    """Download the CSV once and verify it, so every run trains on exactly the same bytes."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} does not match the pinned dataset")
    return path


@dataclass
class Catalogue:
    tracks: pd.DataFrame  # one row per track, with split, point-in-time artist features and labels
    playlists: pd.DataFrame  # one row per playlist: index, id, name, genre, subgenre
    membership: dict[str, list[int]]  # track_id -> playlist indices

    def split(self, name: str) -> pd.DataFrame:
        return self.tracks[self.tracks["split"] == name]

    def history(self) -> pd.DataFrame:
        """Everything released before the test period: what the final models and baselines learn
        from when they score the test releases."""
        return self.tracks[self.tracks["split"] != "test"]


def _artist_history(tracks: pd.DataFrame) -> pd.DataFrame:
    """For each track, what was known about its artist *before* its release date: how many earlier
    releases, and their mean popularity. Tracks released the same day do not see each other.

    Caveat, stated in docs/MODEL_CARD.md: popularity is a January 2020 snapshot, so an earlier track's
    popularity includes plays it collected after the later track came out. It is the best
    point-in-time proxy this dataset allows; a production system would snapshot it daily."""
    t = tracks.sort_values("release_date").copy()
    rows = []
    for _, g in t.groupby("track_artist", sort=False):
        dates = g["release_date"].to_numpy()
        pops = g["track_popularity"].to_numpy(float)
        # releases strictly before each date
        before = np.searchsorted(dates, dates, side="left")
        csum = np.concatenate([[0.0], np.cumsum(pops)])
        n = before
        mean = np.where(n > 0, csum[before] / np.maximum(n, 1), np.nan)
        rows.append(pd.DataFrame({"track_id": g["track_id"].to_numpy(), "artist_prior_tracks": n, "artist_prior_pop": mean}))
    return tracks.merge(pd.concat(rows), on="track_id")


def load(path: Path) -> Catalogue:
    raw = pd.read_csv(fetch(path))
    raw = raw.dropna(subset=["track_name", "track_artist"])
    raw["release_date"] = pd.to_datetime(raw["track_album_release_date"], errors="coerce", format="mixed")
    raw = raw.dropna(subset=["release_date"])

    playlists = raw.drop_duplicates("playlist_id")[["playlist_id", "playlist_name", "playlist_genre", "playlist_subgenre"]]
    playlists = playlists.sort_values("playlist_id").reset_index(drop=True)
    playlists.insert(0, "idx", np.arange(len(playlists)))
    pidx = dict(zip(playlists["playlist_id"], playlists["idx"], strict=True))
    membership = raw.groupby("track_id")["playlist_id"].apply(lambda s: sorted({pidx[p] for p in s})).to_dict()

    tracks = raw.drop_duplicates("track_id")[
        ["track_id", "track_name", "track_artist", "track_album_name", "track_popularity", "release_date", *AUDIO]
    ].reset_index(drop=True)
    tracks["track_album_name"] = tracks["track_album_name"].fillna("")
    tracks = _artist_history(tracks)
    # The data was collected once, in January 2020. A track's age at collection is what lets a model
    # learn that new releases go to new-music playlists and are scored on fewer weeks of plays.
    snapshot = tracks["release_date"].max()
    tracks["age_days"] = (snapshot - tracks["release_date"]).dt.days.clip(lower=0)
    tracks.attrs["snapshot"] = str(snapshot.date())
    tracks["split"] = np.where(tracks["release_date"] >= TEST_FROM, "test", np.where(tracks["release_date"] >= VALID_FROM, "valid", "train"))
    hit_threshold = float(tracks.loc[tracks["split"] == "train", "track_popularity"].quantile(HIT_QUANTILE))
    tracks["hit"] = (tracks["track_popularity"] >= hit_threshold).astype(int)
    tracks.attrs["hit_threshold"] = hit_threshold
    return Catalogue(tracks.sort_values(["release_date", "track_id"]).reset_index(drop=True), playlists, membership)
