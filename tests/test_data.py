import numpy as np
import pytest

from mci import data

from .conftest import DATA


@pytest.fixture(scope="module")
def cat():
    if not DATA.exists():
        pytest.skip("dataset not downloaded: run `mci fetch`")
    return data.load(DATA)


def test_dataset_is_the_pinned_one(cat):
    assert len(cat.tracks) == 28_352 and len(cat.playlists) == 471


def test_split_is_by_release_date(cat):
    t = cat.tracks
    assert t.loc[t.split == "train", "release_date"].max() < data.VALID_FROM
    assert t.loc[t.split == "test", "release_date"].min() >= data.TEST_FROM
    assert set(cat.history()["split"]) == {"train", "valid"}


def test_artist_history_only_sees_earlier_releases(cat):
    t = cat.tracks
    for artist in t["track_artist"].value_counts().index[:25]:
        g = t[t.track_artist == artist].sort_values("release_date")
        for row in g.itertuples():
            earlier = g[g.release_date < row.release_date]
            assert row.artist_prior_tracks == len(earlier)
            if len(earlier):
                assert row.artist_prior_pop == pytest.approx(earlier.track_popularity.mean())
            else:
                assert np.isnan(row.artist_prior_pop)


def test_hit_threshold_comes_from_training_period(cat):
    tr = cat.split("train")
    assert cat.tracks.attrs["hit_threshold"] == pytest.approx(tr.track_popularity.quantile(data.HIT_QUANTILE))
